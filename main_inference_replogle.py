import argparse
import os
import re
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

from dsets.replogle.replogle_dataset import Replogle_Dataset
from models import MInterface


start = time.time()
end = time.time()
print("Import耗时: ", end - start, "秒")


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")
DEFAULT_REPLOGLE_ROOT = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "arcinstitute",
    "datasets",
    "State_Replogle_Filtered",
)
VAR_DIMS_PATH = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "arcinstitute",
    "collections",
    "ST_Replogle",
    "var_dims.pkl",
)
AVG_DELTA_PATH = os.path.join(
    PROJECT_ROOT,
    "dsets",
    "replogle",
    "replogle_train_avg_delta.pkl",
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
        ob = {
            "celltype": cell_chunk["cartesian_key"][1],
            "gene": cell_chunk["cartesian_key"][2],
        }
        cell_matrix = cell_chunk["cell_matrix"]
        obs.extend([ob] * cell_matrix.shape[0])
        xs.append(sparse.csr_matrix(cell_matrix))

    obs = pd.DataFrame(obs)
    xs = sparse.vstack(xs)
    return ad.AnnData(X=xs, obs=obs, var=var)


def load_avg_delta(path=AVG_DELTA_PATH):
    with open(path, "rb") as f:
        avg_delta = pickle.load(f)
    print(
        "Loaded avg delta:",
        path,
        "shape=",
        avg_delta["delta"].shape,
    )
    return avg_delta


def to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
    return x


def iter_cartesian_keys(batch_value):
    return [tuple(to_python_scalar(v) for v in key) for key in zip(*batch_value)]


def add_avg_delta(tgt_pred, cartesian_keys, avg_delta):
    # Add one precomputed perturbation-level delta vector to every predicted cell.
    out = tgt_pred.clone()
    applied = 0
    missing = 0
    for idx, cartesian_key in enumerate(cartesian_keys):
        gene = str(cartesian_key[-1])
        delta_idx = avg_delta["pert_to_idx"].get(gene)
        if delta_idx is None:
            missing += 1
            continue
        delta = torch.as_tensor(
            avg_delta["delta"][delta_idx],
            device=out.device,
            dtype=out.dtype,
        )
        if delta.shape[-1] != out.shape[-1]:
            raise ValueError(
                f"avg_delta dim mismatch for gene={gene}: "
                f"{delta.shape[-1]} vs pred dim {out.shape[-1]}"
            )
        out[idx] = out[idx] + delta
        applied += 1
    return out, applied, missing


def get_fewshot_name(test_lmdb_path):
    lmdb_name = os.path.basename(os.path.normpath(test_lmdb_path))
    if lmdb_name.startswith("replogle_test_"):
        return lmdb_name[len("replogle_test_") :]
    return lmdb_name


def set_fewshot_paths(conf, cell_line, replogle_root=DEFAULT_REPLOGLE_ROOT):
    cell_line = cell_line.lower()
    if cell_line not in CELL_LINES:
        raise ValueError(f"Unsupported cell line {cell_line!r}; expected one of {CELL_LINES}")

    base = os.path.join(replogle_root, "few_shot", cell_line)
    paths = {
        "global_keys_path": os.path.join(replogle_root, "global_keys.pkl"),
        "train_lmdb_path": os.path.join(base, f"replogle_train_{cell_line}"),
        "val_lmdb_path": os.path.join(base, f"replogle_val_{cell_line}"),
        "test_lmdb_path": os.path.join(base, f"replogle_test_{cell_line}"),
        "control_lmdb_path": os.path.join(base, f"replogle_control_{cell_line}"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Replogle path for {cell_line}: {path}")
        OmegaConf.update(conf, f"dataset.{key}", path, merge=True)


def cell_lines_from_text(text):
    if text is None:
        return []
    text = str(text).lower()
    matches = []
    for cell_line in CELL_LINES:
        pattern = rf"(?<![a-z0-9]){re.escape(cell_line)}(?![a-z0-9])"
        if re.search(pattern, text):
            matches.append(cell_line)
    return matches


def infer_cell_line_from_config(conf, config_path=None):
    sources = []
    if config_path:
        sources.append(("config_path", config_path))

    for key_path in (
        "dataset.test_lmdb_path",
        "dataset.control_lmdb_path",
        "dataset.val_lmdb_path",
        "dataset.train_lmdb_path",
        "experiment.group",
        "prepare_model.megatron_ckpt",
    ):
        value = OmegaConf.select(conf, key_path, default=None)
        if value:
            sources.append((key_path, value))

    candidates = {}
    for source_name, value in sources:
        for cell_line in cell_lines_from_text(value):
            candidates.setdefault(cell_line, []).append(source_name)

    if not candidates:
        raise ValueError(
            "Unable to infer Replogle cell line from config. "
            "Use --cell-line or include one of "
            f"{CELL_LINES} in dataset paths, experiment.group, checkpoint path, or config filename."
        )

    if len(candidates) > 1:
        detail = ", ".join(
            f"{cell_line} from {sources}" for cell_line, sources in sorted(candidates.items())
        )
        raise ValueError(
            "Ambiguous Replogle cell line in config. "
            f"Candidates: {detail}. Use --cell-line to override explicitly."
        )

    return next(iter(candidates))


def prepare_var():
    with open(VAR_DIMS_PATH, "rb") as f:
        var_dims = pickle.load(f)
    gene_names = list(var_dims["gene_names"])
    return pd.DataFrame(index=gene_names)


def run(
    conf,
    dataset_name,
    cell_line_hint=None,
    out_dir=None,
    timestamp=None,
    replogle_root=DEFAULT_REPLOGLE_ROOT,
    config_path=None,
):
    cell_line = cell_line_hint or infer_cell_line_from_config(conf, config_path=config_path)
    print(f"Using Replogle cell line: {cell_line}")
    set_fewshot_paths(conf, cell_line, replogle_root)
    conf = OmegaConf.to_container(conf, resolve=True)
    dataset_conf = conf["dataset"]

    num_workers = dataset_conf.get("num_workers", 4)
    batch_size = dataset_conf["batch_size"]
    control_pert = dataset_conf["control_pert"]
    pred_cell_chunks = []
    real_cell_chunks = []

    test_lmdb_path = dataset_conf["test_lmdb_path"]
    control_lmdb_path = dataset_conf["control_lmdb_path"]

    test_env = lmdb.open(test_lmdb_path, readonly=True, lock=False)
    control_env = lmdb.open(control_lmdb_path, readonly=True, lock=False)

    with open(dataset_conf["global_keys_path"], "rb") as f:
        global_keys = pickle.load(f)

    ds = Replogle_Dataset(
        dataset_conf,
        test_env,
        control_env,
        global_keys,
        mode="test")

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
    avg_delta = load_avg_delta()
    avg_delta_applied = 0
    avg_delta_missing = 0

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
                cartesian_keys = iter_cartesian_keys(batch["pert_cartesian_key"])
                tgt_pred, applied, missing = add_avg_delta(tgt_pred, cartesian_keys, avg_delta)
                tgt_pred = apply_prediction_cutoff(tgt_pred)
                avg_delta_applied += applied
                avg_delta_missing += missing

                for idx, cartesian_key in enumerate(cartesian_keys):
                    batch_name, cell_type, _ = cartesian_key
                    ctrl_cartesian_key = (batch_name, cell_type, control_pert)
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

    print(f"Avg delta applied to {avg_delta_applied} groups; missing={avg_delta_missing}.")

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
    var = prepare_var()

    print("[3/4] Build Anndata (real and pred)...")
    real_adata = build_anndata(real_cell_chunks, var)
    pred_adata = build_anndata(pred_cell_chunks, var)

    print("[4/4] Save Anndata (real and pred)...")
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp
        save_path = os.path.join(
            PROJECT_ROOT,
            "benchmark",
            "workspace",
            (
                f"{timestamp}_{conf['model_name']}_{conf['dataset_name']}"
                if timestamp
                else f"{conf['model_name']}_{conf['dataset_name']}"
            ),
        )
    elif timestamp == "":
        save_path = out_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp
        save_path = os.path.join(
            out_dir,
            f"{timestamp}_{conf['model_name']}_{conf['dataset_name']}",
        )
    os.makedirs(save_path, exist_ok=True)

    fewshot_name = get_fewshot_name(test_lmdb_path)
    real_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_real_{fewshot_name}.h5ad"))
    pred_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_pred_{fewshot_name}.h5ad"))

    print("Inference and Anndata building completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Replogle inference and build h5ad outputs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the OmegaConf YAML config.",
    )
    parser.add_argument(
        "--cell-line",
        choices=CELL_LINES,
        default=None,
        help="Optional override. If omitted, infer from the config.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--timestamp",
        default=None,
        help='Use "" to write directly into --out-dir.',
    )
    parser.add_argument("--replogle-root", default=DEFAULT_REPLOGLE_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    conf = OmegaConf.load(args.config)
    run(
        conf,
        dataset_name="replogle",
        cell_line_hint=args.cell_line,
        out_dir=args.out_dir,
        timestamp=args.timestamp,
        replogle_root=args.replogle_root,
        config_path=args.config,
    )
