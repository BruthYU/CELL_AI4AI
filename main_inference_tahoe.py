# Evaluation on single rank based on cell-eval
import argparse
import os 
import time
from datetime import datetime
start = time.time()
from omegaconf import OmegaConf, DictConfig
import torch
from models import MInterface
from omegaconf import  OmegaConf
import lmdb
import pickle
import gc
from dsets.tahoe.tahoe_dataset import Tahoe_Dataset
from torch.utils.data import DataLoader, Dataset
import ast
from tqdm import tqdm
import scipy.sparse as sparse
import pandas as pd
import anndata as ad
end = time.time()
print("Import耗时: ", end - start, "秒")


# row = {
#     "plate_celltype": f"{plate}_{celltype}",
#     "drugname_drugconc": drug,
# }
# rows_df1.extend([row.copy() for _ in range(512)])

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
            _batch = batch[0]
        else:
            _batch = batch
        def to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            else:
                return x
        _batch = {k: to_cuda(v) for k, v in _batch.items()}
        return _batch

def apply_prediction_cutoff(tgt_pred, cutoff=PREDICTION_CUTOFF):
    return tgt_pred.masked_fill(tgt_pred < cutoff, 0)

def cuda_sync_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def build_anndata(cell_chunks, var):
    """
    将 cell_chunks 列表构建为 AnnData 对象
    
    Args:
        cell_chunks: list of dict, 每个 dict 包含 'cartesian_key' 和 'tgt_real'/'tgt_pred'
        var: pd.DataFrame, 基因/特征信息
    
    Returns:
        ad.AnnData
    """
    obs = []
    Xs = []
    for cell_chunk in cell_chunks:
        ob = {
            "celltype": f"{cell_chunk['cartesian_key'][0]}_{cell_chunk['cartesian_key'][1]}",
            "drugname_drugconc": cell_chunk['cartesian_key'][2],
        }
        
        # 支持 'tgt_real' 或 'tgt_pred' 键
        cell_matrix = cell_chunk['cell_matrix']
        obs.extend([ob] * cell_matrix.shape[0])
        Xs.append(sparse.csr_matrix(cell_matrix))
    
    obs = pd.DataFrame(obs)
    Xs = sparse.vstack(Xs)
    return ad.AnnData(X=Xs, obs=obs, var=var)



def run(conf, dataset_name):
    run_start_time = time.perf_counter()
    conf = OmegaConf.to_container(conf, resolve=True)   # DictConf → dict, to support Megatron serialisation
    pred_cell_chunks = []
    real_cell_chunks = []
    # dataset
    dataset_conf = conf["dataset"]
    control_pert = dataset_conf["control_pert"]
    batch_size = dataset_conf.get("batch_size", 64)
    num_workers = dataset_conf.get("num_workers", 4)

    # test_lmdb_path = dataset_conf["test_lmdb_path"]
    # test_env = lmdb.open(test_lmdb_path, readonly=True, lock=False)
    test_lmdb_path = dataset_conf["test_lmdb_path"]
    test_env = lmdb.open(test_lmdb_path, readonly=True, lock=False)

    control_lmdb_path = dataset_conf["control_lmdb_path"]
    control_env = lmdb.open(control_lmdb_path, readonly=True, lock=False)



    with open(dataset_conf["global_keys_path"], "rb") as f:
            global_keys = pickle.load(f)

    fingerprint_map = None
    fp_path = dataset_conf.get("fingerprint_path")
    if fp_path:
        with open(fp_path, "rb") as f:
            fp_data = pickle.load(f)
        fingerprint_map = fp_data["fingerprints"]
        fp_dim = fp_data.get("meta", {}).get("fp_dim", "unknown")
        print(f"Loaded {len(fingerprint_map)} drug fingerprints (dim={fp_dim}) from {fp_path}")

    ds = Tahoe_Dataset(
        dataset_conf,
        test_env,
        control_env,
        global_keys,
        mode="test",
        fingerprint_map=fingerprint_map,
    )
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_dataset_worker,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
    inference_dataloader = DataLoader(ds, **dataloader_kwargs)
    # Model Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model from ckpt
    model_interface = MInterface(conf)
    model = model_interface.model
    if hasattr(model, "module"):
        model.module.to(device)
    else:
        model.to(device)
    model.eval()


    print("[1/4] Start Inference...")
    # 获取real和pred的中间信息
    total_batches = 0
    total_cells = 0
    model_forward_time = 0.0
    cuda_sync_if_available()
    inference_stage_start_time = time.perf_counter()
    with torch.no_grad():
        for repeat_idx in range(INFERENCE_REPEAT_COUNT):
            repeat_seed = int(dataset_conf["random_seed"]) + repeat_idx * INFERENCE_REPEAT_SEED_STRIDE
            set_dataset_sampling_seed(ds, repeat_seed)
            desc = f"Inference pass {repeat_idx + 1}/{INFERENCE_REPEAT_COUNT}"
            print(f"[1/4] {desc} sampling seed: {repeat_seed}")
            for i, batch in enumerate(tqdm(inference_dataloader, desc=desc, unit="batch")):

                batch = batch_to_cuda(batch)
                tgt_real = batch["pert_cell_emb"]
                tgt_ctrl = batch["ctrl_cell_emb"]

                cuda_sync_if_available()
                model_forward_start_time = time.perf_counter()
                tgt_pred = model.predict_step(batch)
                tgt_pred = apply_prediction_cutoff(tgt_pred)
                cuda_sync_if_available()
                model_forward_time += time.perf_counter() - model_forward_start_time
                total_batches += 1
                total_cells += tgt_pred.shape[0]
                
                cartesian_keys = zip(*batch["pert_cartesian_key"])
                for idx, cartesian_key in enumerate(cartesian_keys):
                    plate, cell_line, _ = cartesian_key
                    ctrl_cartesian_key = (plate, cell_line, control_pert)
                    ctrl_cell_chunk = {
                        'cartesian_key': ctrl_cartesian_key,
                        'cell_matrix': tgt_ctrl[idx].cpu().numpy(),
                    }

                    pred_cell_chunk = {
                        'cartesian_key': cartesian_key,
                        'cell_matrix': tgt_pred[idx].cpu().numpy(),
                    }
                    pred_cell_chunks.append(pred_cell_chunk)

                    real_cell_chunk = {
                        'cartesian_key': cartesian_key,
                        'cell_matrix': tgt_real[idx].cpu().numpy(),
                    }
                    real_cell_chunks.append(real_cell_chunk)
                    pred_cell_chunks.append(ctrl_cell_chunk)
                    real_cell_chunks.append(ctrl_cell_chunk)
    cuda_sync_if_available()
    inference_stage_time = time.perf_counter() - inference_stage_start_time
    print(f"[1/4] Inference stats: batches={total_batches}, cells={total_cells}")
    print(f"[1/4] Total inference stage time: {inference_stage_time:.2f} s")
    print(f"[1/4] Model forward time: {model_forward_time:.2f} s")
    if total_batches > 0:
        print(f"[1/4] Avg forward time per batch: {model_forward_time / total_batches:.4f} s")
        print(f"[1/4] Throughput: {total_batches / inference_stage_time:.2f} batches/s, {total_cells / inference_stage_time:.2f} cells/s")

    # pred和real保存相同的control信息
    # with control_env.begin(write=False) as txn:
    #     control_keys = txn.get(b"__keys__")
    #     control_keys = pickle.loads(control_keys)
    #     for key in control_keys:  # 已知的 key 列表
    #         value = txn.get(str(key).encode())
    #         control_cell_chunk = {
    #             'cartesian_key': key,
    #             'cell_matrix': pickle.loads(value),
    #         }
    #         pred_cell_chunks.append(control_cell_chunk)
    #         real_cell_chunks.append(control_cell_chunk)

    print("[2/4] Prepare adata.var...")
    pkl_path = dataset_conf.get(
        "var_dims_path",
        os.path.join(PROJECT_ROOT, "preprocessing", "tahoe", "group", "var_dims_tahoe.pkl"),
    )
    with open(pkl_path, "rb") as f:
        var_dims = pickle.load(f)
    gene_names = [name for name in var_dims['gene_names']]
    var = pd.DataFrame(index=gene_names)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        PROJECT_ROOT,
        "benchmark",
        "workspace",
        f"{timestamp}_{conf['model_name']}_{conf['dataset_name']}",
    )
    os.makedirs(save_path, exist_ok=True)

    print("[3/4] Build and save Anndata (real)...")
    real_adata = build_anndata(real_cell_chunks, var)
    real_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_real.h5ad"))
    del real_adata, real_cell_chunks
    gc.collect()

    print("[4/4] Build and save Anndata (pred)...")
    pred_adata = build_anndata(pred_cell_chunks, var)
    pred_adata.write_h5ad(os.path.join(save_path, f"{dataset_name}_pred.h5ad"))
    del pred_adata, pred_cell_chunks
    gc.collect()

    total_run_time = time.perf_counter() - run_start_time
    print(f"Total run time: {total_run_time:.2f} s")
    print("Inference and Anndata building completed!")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Tahoe inference and build h5ad outputs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the OmegaConf YAML config.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Using config: {args.config}")
    conf = OmegaConf.load(args.config)
    run(conf, dataset_name="tahoe100m")
