import os
import argparse
import time
from datetime import datetime

import anndata as ad
import lmdb
import pandas as pd
import pickle
import scipy.sparse as sparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsets.pbmc.pbmc_dataset import PBMC_Dataset
from models import MInterface


start = time.time()
end = time.time()
print("Import耗时: ", end - start, "秒")


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VAR_DIMS_PATH = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "arcinstitute",
    "collections",
    "ST_Parse",
    "var_dims.pkl",
)
PREDICTION_CUTOFF = 0.05
INFERENCE_REPEAT_COUNT = 3
INFERENCE_REPEAT_SEED_STRIDE = 1000003


def set_dataset_sampling_seed(dataset, seed):
    if hasattr(dataset, "set_sampling_seed"):
        dataset.set_sampling_seed(seed)


def seed_dataset_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    base_seed = getattr(dataset, "sampling_seed", getattr(dataset, "random_seed", 0))
    set_dataset_sampling_seed(dataset, int(base_seed) + worker_id)


def apply_pbmc_env_overrides(conf):
    overrides = [
        ("PBMC_EXPERIMENT_GROUP", "experiment.group", str),
        ("PBMC_BATCH_SIZE", "dataset.batch_size", int),
        ("PBMC_NUM_WORKERS", "dataset.num_workers", int),
        ("PBMC_MEGATRON_CKPT", "prepare_model.megatron_ckpt", str),
    ]
    applied = []
    for env_name, key_path, caster in overrides:
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == "":
            continue
        value = caster(raw_value)
        OmegaConf.update(conf, key_path, value, merge=False)
        applied.append(f"{key_path}={value}")
    if applied:
        print("PBMC config overrides:", ", ".join(applied))


def batch_to_cuda(batch):
    if isinstance(batch, tuple) and len(batch) == 3:
        batch = batch[0]

    def to_cuda(x):
        if isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        return x

    return {k: to_cuda(v) for k, v in batch.items()}


def apply_prediction_cutoff(tgt_pred, cutoff=PREDICTION_CUTOFF):
    return tgt_pred.masked_fill(tgt_pred < cutoff, 0)


def build_anndata(cell_chunks, var):
    obs = []
    xs = []
    for cell_chunk in cell_chunks:
        donor, cell_type, cytokine = cell_chunk["cartesian_key"]
        ob = {
            "donor": donor,
            "celltype": cell_type,
            "cell_type": cell_type,
            "cytokine": cytokine,
        }
        cell_matrix = cell_chunk["cell_matrix"]
        obs.extend([ob] * cell_matrix.shape[0])
        xs.append(sparse.csr_matrix(cell_matrix))

    obs = pd.DataFrame(obs)
    xs = sparse.vstack(xs)
    return ad.AnnData(X=xs, obs=obs, var=var)


def get_split_name(test_lmdb_path):
    lmdb_name = os.path.basename(os.path.normpath(test_lmdb_path))
    for prefix in ("pbmc_train_", "pbmc_val_", "pbmc_test_"):
        if lmdb_name.startswith(prefix):
            return lmdb_name[len(prefix) :]
    return lmdb_name


def prepare_var(n_genes):
    if os.path.exists(VAR_DIMS_PATH):
        with open(VAR_DIMS_PATH, "rb") as f:
            var_dims = pickle.load(f)
        gene_names = list(var_dims["gene_names"])
    else:
        gene_names = [f"gene_{i}" for i in range(n_genes)]
    return pd.DataFrame(index=gene_names)


def run(conf, dataset_name):
    conf = OmegaConf.to_container(conf, resolve=True)
    dataset_conf = conf["dataset"]

    num_workers = dataset_conf.get("num_workers", 4)
    batch_size = dataset_conf["batch_size"]
    control_pert = dataset_conf["control_pert"]
    pred_cell_chunks = []
    real_cell_chunks = []

    test_lmdb_path = dataset_conf["test_lmdb_path"]
    control_lmdb_path = dataset_conf["control_lmdb_path"]

    print(f"Using PBMC test LMDB: {test_lmdb_path}")
    print(f"Using PBMC control LMDB: {control_lmdb_path}")

    test_env = lmdb.open(
        test_lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )
    control_env = lmdb.open(
        control_lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )

    with open(dataset_conf["global_keys_path"], "rb") as f:
        global_keys = pickle.load(f)

    ds = PBMC_Dataset(dataset_conf, test_env, control_env, global_keys, mode="test")
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_dataset_worker,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
    inference_dataloader = DataLoader(ds, **dataloader_kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_interface = MInterface(conf)
    model = model_interface.model
    if hasattr(model, "module"):
        model.module.to(device)
    else:
        model.to(device)
    model.eval()

    print("[1/4] Start Inference...")
    with torch.no_grad():
        for repeat_idx in range(INFERENCE_REPEAT_COUNT):
            repeat_seed = int(dataset_conf["random_seed"]) + repeat_idx * INFERENCE_REPEAT_SEED_STRIDE
            set_dataset_sampling_seed(ds, repeat_seed)
            desc = f"Inference pass {repeat_idx + 1}/{INFERENCE_REPEAT_COUNT}"
            print(f"[1/4] {desc} sampling seed: {repeat_seed}")
            for batch in tqdm(inference_dataloader, desc=desc):
                batch = batch_to_cuda(batch)
                tgt_real = batch["pert_cell_emb"]
                tgt_ctrl = batch["ctrl_cell_emb"]
                tgt_pred = model.predict_step(batch)
                tgt_pred = apply_prediction_cutoff(tgt_pred)

                cartesian_keys = zip(*batch["pert_cartesian_key"])
                for idx, cartesian_key in enumerate(cartesian_keys):
                    donor, cell_type, _ = cartesian_key
                    ctrl_cartesian_key = (donor, cell_type, control_pert)
                    ctrl_cell_chunk = {
                        "cartesian_key": ctrl_cartesian_key,
                        "cell_matrix": tgt_ctrl[idx].cpu().numpy(),
                    }

                    pred_cell_chunks.append(
                        {
                            "cartesian_key": cartesian_key,
                            "cell_matrix": tgt_pred[idx].cpu().numpy(),
                        }
                    )
                    real_cell_chunks.append(
                        {
                            "cartesian_key": cartesian_key,
                            "cell_matrix": tgt_real[idx].cpu().numpy(),
                        }
                    )
                    pred_cell_chunks.append(ctrl_cell_chunk)
                    real_cell_chunks.append(ctrl_cell_chunk)

    # with control_env.begin(write=False) as txn:
    #     control_keys = pickle.loads(txn.get(b"__keys__"))
    #     for key in control_keys:
    #         value = txn.get(str(key).encode())
    #         control_cell_chunk = {
    #             "cartesian_key": key,
    #             "cell_matrix": pickle.loads(value),
    #         }
    #         pred_cell_chunks.append(control_cell_chunk)
    #         real_cell_chunks.append(control_cell_chunk)

    print("[2/4] Prepare adata.var...")
    n_genes = pred_cell_chunks[0]["cell_matrix"].shape[-1]
    var = prepare_var(n_genes)

    print("[3/4] Build Anndata (real and pred)...")
    real_adata = build_anndata(real_cell_chunks, var)
    pred_adata = build_anndata(pred_cell_chunks, var)

    print("[4/4] Save Anndata (real and pred)...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = os.environ.get("PBMC_WORKSPACE_NAME") or (
        f"{timestamp}_{conf['model_name']}_{conf['dataset_name']}"
    )
    save_path = os.path.join(
        PROJECT_ROOT,
        "benchmark",
        "workspace",
        workspace_name,
    )
    os.makedirs(save_path, exist_ok=True)

    split_name = get_split_name(test_lmdb_path)
    real_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_real_{split_name}.h5ad"))
    pred_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_pred_{split_name}.h5ad"))

    path_file = os.environ.get("PBMC_WORKSPACE_PATH_FILE")
    if path_file:
        with open(path_file, "w") as f:
            f.write(save_path + "\n")

    print("Inference and Anndata building completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PBMC inference.")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "prior_llm_pbmc.yaml"),
        help="Path to the OmegaConf YAML config.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    apply_pbmc_env_overrides(conf)
    run(conf, dataset_name="pbmc")
