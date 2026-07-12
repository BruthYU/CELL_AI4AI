from cell_eval import MetricsEvaluator
from cell_eval.data import build_random_anndata, downsample_cells
from cell_eval.utils import split_anndata_on_celltype
import anndata as ad
import argparse
import gc
import h5py
import numpy as np
from scipy.sparse import issparse
import scipy.sparse as sp
import os


# 当前 step 的测试目录，结果会保存在该目录下，避免与其它 step 覆盖
DEFAULT_STEP_DIR = "20260615_055828_JIT_FP"
DEFAULT_OUT_SUBFOLDER = "results_calibrate"
DEFAULT_WORKSPACE = os.path.join(os.path.dirname(__file__), "workspace")
DEFAULT_REAL_H5AD = "tahoe100m_real.h5ad"
DEFAULT_PRED_H5AD = "tahoe100m_pred.h5ad"


def infer_perturbation_config(real_adata, pred_adata):
    common_obs = set(real_adata.obs.columns) & set(pred_adata.obs.columns)
    candidates = [
        ("drugname_drugconc", "[('DMSO_TF', 0.0, 'uM')]"),
        ("gene", "non-targeting"),
        ("cytokine", "PBS"),
    ]

    for pert_col, control_pert in candidates:
        if pert_col not in common_obs:
            continue

        real_perts = set(real_adata.obs[pert_col].astype(str).unique().tolist())
        pred_perts = set(pred_adata.obs[pert_col].astype(str).unique().tolist())
        if control_pert in real_perts and control_pert in pred_perts:
            return pert_col, control_pert

    raise ValueError(
        "Unable to infer perturbation config from AnnData obs columns. "
        f"real obs={list(real_adata.obs.columns)}, pred obs={list(pred_adata.obs.columns)}"
    )


def clip_adata(adata):
    if issparse(adata.X):
        adata.X.data = np.clip(adata.X.data, 0, 14.99)
    else:
        adata.X = np.clip(adata.X, 0., 14.99)


def resolve_celltype_col(real_adata, pred_adata, requested_col):
    real_cols = set(real_adata.obs.columns)
    pred_cols = set(pred_adata.obs.columns)
    if requested_col in real_cols and requested_col in pred_cols:
        return requested_col
    if requested_col == "celltype" and "cell_type" in real_cols and "cell_type" in pred_cols:
        print("celltype column 'celltype' not found in both AnnData obs; using 'cell_type'.")
        return "cell_type"
    raise ValueError(
        f"AnnData is missing celltype column {requested_col!r}. "
        f"Available real obs columns: {list(real_adata.obs.columns)}; "
        f"available pred obs columns: {list(pred_adata.obs.columns)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tahoe h5ad predictions.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--step-dir", default=DEFAULT_STEP_DIR)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing Tahoe real/pred h5ad files. Overrides workspace/step-dir.",
    )
    parser.add_argument("--real-h5ad", default=DEFAULT_REAL_H5AD)
    parser.add_argument("--pred-h5ad", default=DEFAULT_PRED_H5AD)
    parser.add_argument("--out-subfolder", default=DEFAULT_OUT_SUBFOLDER)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--celltype-col", default="celltype")
    parser.add_argument(
        "--only-celltypes",
        default="",
        help="Comma-separated exact celltype labels to evaluate. Empty means all.",
    )
    parser.add_argument(
        "--only-celltype-contains",
        default="",
        help="Comma-separated substrings; evaluate celltypes containing any substring. Empty means all.",
    )
    parser.add_argument(
        "--skip-metrics",
        default="pearson_edistance,clustering_agreement",
        help="Comma-separated metrics to skip.",
    )
    return parser.parse_args()


def parse_skip_metrics(value):
    if not value:
        return []
    return [metric.strip() for metric in value.split(",") if metric.strip()]


def parse_csv_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def has_celltype_filter(args):
    return bool(parse_csv_list(args.only_celltypes) or parse_csv_list(args.only_celltype_contains))


def select_celltypes(all_celltypes, args):
    celltypes = sorted([str(celltype) for celltype in all_celltypes], key=str)
    only_celltypes = set(parse_csv_list(args.only_celltypes))
    contains_filters = parse_csv_list(args.only_celltype_contains)
    if only_celltypes:
        missing = sorted(only_celltypes - set(celltypes), key=str)
        if missing:
            raise ValueError(f"--only-celltypes values not found: {missing}")
        celltypes = [celltype for celltype in celltypes if celltype in only_celltypes]
    if contains_filters:
        celltypes = [
            celltype
            for celltype in celltypes
            if any(pattern in str(celltype) for pattern in contains_filters)
        ]
    if not celltypes:
        raise ValueError("No celltypes selected for evaluation.")
    return celltypes


def load_anndata_pair(real_path, pred_path, args):
    has_filter = bool(parse_csv_list(args.only_celltypes) or parse_csv_list(args.only_celltype_contains))
    if not has_filter:
        return ad.read_h5ad(real_path), ad.read_h5ad(pred_path), None

    print("Loading filtered backed AnnData subset...")
    real_backed = ad.read_h5ad(real_path, backed="r")
    pred_backed = ad.read_h5ad(pred_path, backed="r")
    try:
        args.celltype_col = resolve_celltype_col(real_backed, pred_backed, args.celltype_col)
        real_celltypes = set(real_backed.obs[args.celltype_col].astype(str).unique().tolist())
        pred_celltypes = set(pred_backed.obs[args.celltype_col].astype(str).unique().tolist())
        if real_celltypes != pred_celltypes:
            missing_in_pred = sorted(real_celltypes - pred_celltypes, key=str)
            missing_in_real = sorted(pred_celltypes - real_celltypes, key=str)
            raise ValueError(
                "Real and predicted AnnData have different celltypes. "
                f"missing_in_pred={missing_in_pred}, missing_in_real={missing_in_real}"
            )

        selected_celltypes = select_celltypes(real_celltypes, args)
        selected_set = set(selected_celltypes)
        real_mask = real_backed.obs[args.celltype_col].astype(str).isin(selected_set).to_numpy()
        pred_mask = pred_backed.obs[args.celltype_col].astype(str).isin(selected_set).to_numpy()
        real_adata = real_backed[real_mask].to_memory()
        pred_adata = pred_backed[pred_mask].to_memory()
        return real_adata, pred_adata, selected_celltypes
    finally:
        real_backed.file.close()
        pred_backed.file.close()


def contiguous_runs(indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, indices.size]
    return [(int(indices[start]), int(indices[stop - 1])) for start, stop in zip(starts, stops)]


def read_x_rows(h5, row_indices):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    x = h5["X"]
    if isinstance(x, h5py.Dataset):
        n_vars = int(x.shape[1])
        out = np.empty((len(row_indices), n_vars), dtype=x.dtype)
        out_pos = 0
        for start, end in contiguous_runs(row_indices):
            block = x[start : end + 1, :]
            out[out_pos : out_pos + block.shape[0], :] = block
            out_pos += block.shape[0]
        return out

    encoding = x.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding != "csr_matrix":
        raise ValueError(f"Unsupported backed X encoding: {encoding!r}")

    shape = tuple(int(v) for v in x.attrs["shape"])
    indptr_ds = x["indptr"]
    data_ds = x["data"]
    indices_ds = x["indices"]
    total_nnz = 0
    runs = contiguous_runs(row_indices)
    for start, end in runs:
        total_nnz += int(indptr_ds[end + 1] - indptr_ds[start])

    data = np.empty(total_nnz, dtype=data_ds.dtype)
    indices = np.empty(total_nnz, dtype=indices_ds.dtype)
    indptr = np.empty(len(row_indices) + 1, dtype=indptr_ds.dtype)
    indptr[0] = 0
    row_pos = 0
    data_pos = 0
    for start, end in runs:
        old_indptr = indptr_ds[start : end + 2]
        span_start = int(old_indptr[0])
        span_stop = int(old_indptr[-1])
        span_nnz = span_stop - span_start
        data[data_pos : data_pos + span_nnz] = data_ds[span_start:span_stop]
        indices[data_pos : data_pos + span_nnz] = indices_ds[span_start:span_stop]
        shifted = old_indptr - span_start + data_pos
        n_rows = end - start + 1
        indptr[row_pos + 1 : row_pos + n_rows + 1] = shifted[1:]
        row_pos += n_rows
        data_pos += span_nnz
    return sp.csr_matrix((data, indices, indptr), shape=(len(row_indices), shape[1]))


def make_adata_from_h5_rows(h5, obs, var, row_indices):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    return ad.AnnData(
        X=read_x_rows(h5, row_indices),
        obs=obs.iloc[row_indices].copy(),
        var=var.copy(),
    )


def run_filtered_backed(real_path, pred_path, outdir, args):
    print("Loading filtered backed AnnData subset...")
    real_backed = ad.read_h5ad(real_path, backed="r")
    pred_backed = ad.read_h5ad(pred_path, backed="r")
    real_h5 = h5py.File(real_path, "r")
    pred_h5 = h5py.File(pred_path, "r")
    try:
        args.celltype_col = resolve_celltype_col(real_backed, pred_backed, args.celltype_col)

        real_celltypes = set(real_backed.obs[args.celltype_col].astype(str).unique().tolist())
        pred_celltypes = set(pred_backed.obs[args.celltype_col].astype(str).unique().tolist())
        if real_celltypes != pred_celltypes:
            missing_in_pred = sorted(real_celltypes - pred_celltypes, key=str)
            missing_in_real = sorted(pred_celltypes - real_celltypes, key=str)
            raise ValueError(
                "Real and predicted AnnData have different celltypes. "
                f"missing_in_pred={missing_in_pred}, missing_in_real={missing_in_real}"
            )

        pert_col, control_pert = infer_perturbation_config(real_backed, pred_backed)
        print(f"Using perturbation config: pert_col={pert_col}, control_pert={control_pert}")
        print(f"[3/3] Evaluating by {args.celltype_col} with DE metrics...")
        skip_metrics = parse_skip_metrics(args.skip_metrics)
        celltypes = select_celltypes(real_celltypes, args)
        print(f"Selected {len(celltypes)} celltypes: {celltypes}")

        real_labels = real_backed.obs[args.celltype_col].astype(str).to_numpy()
        pred_labels = pred_backed.obs[args.celltype_col].astype(str).to_numpy()
        for idx, celltype in enumerate(celltypes, start=1):
            print(f"[{idx}/{len(celltypes)}] Evaluating celltype={celltype}")
            real_rows = np.flatnonzero(real_labels == celltype)
            pred_rows = np.flatnonzero(pred_labels == celltype)
            real_cell = make_adata_from_h5_rows(real_h5, real_backed.obs, real_backed.var, real_rows)
            pred_cell = make_adata_from_h5_rows(pred_h5, pred_backed.obs, pred_backed.var, pred_rows)
            clip_adata(real_cell)
            clip_adata(pred_cell)
            evaluator = MetricsEvaluator(
                adata_pred=pred_cell,
                adata_real=real_cell,
                control_pert=control_pert,
                pert_col=pert_col,
                num_threads=args.num_threads,
                outdir=outdir,
                allow_discrete=False,
                prefix=str(celltype),
            )
            evaluator.compute(
                profile="full",
                skip_metrics=skip_metrics,
                basename="results.csv",
            )
            del evaluator, real_cell, pred_cell
            gc.collect()
    finally:
        real_h5.close()
        pred_h5.close()
        real_backed.file.close()
        pred_backed.file.close()


def main():
    args = parse_args()

    step_path = args.input_dir or os.path.join(args.workspace, args.step_dir)
    outdir = os.path.join(step_path, args.out_subfolder)
    os.makedirs(outdir, exist_ok=True)

    real_path = os.path.join(step_path, args.real_h5ad)
    pred_path = os.path.join(step_path, args.pred_h5ad)

    print("[1/3] Loading data...")
    print(f"Real h5ad: {real_path}")
    print(f"Pred h5ad: {pred_path}")
    print(f"Output dir: {outdir}")
    if has_celltype_filter(args):
        run_filtered_backed(real_path, pred_path, outdir, args)
        print(f"Results saved to: {outdir}")
        return

    real_adata, pred_adata, selected_from_backed = load_anndata_pair(real_path, pred_path, args)

    print("[2/3] Clipping data...")
    clip_adata(real_adata)
    clip_adata(pred_adata)

    pert_col, control_pert = infer_perturbation_config(real_adata, pred_adata)
    print(f"Using perturbation config: pert_col={pert_col}, control_pert={control_pert}")

    args.celltype_col = resolve_celltype_col(real_adata, pred_adata, args.celltype_col)
    print(f"[3/3] Evaluating by {args.celltype_col} with DE metrics...")
    real_split = split_anndata_on_celltype(real_adata, args.celltype_col)
    pred_split = split_anndata_on_celltype(pred_adata, args.celltype_col)

    real_celltypes = set(real_split.keys())
    pred_celltypes = set(pred_split.keys())
    if real_celltypes != pred_celltypes:
        missing_in_pred = sorted(real_celltypes - pred_celltypes, key=str)
        missing_in_real = sorted(pred_celltypes - real_celltypes, key=str)
        raise ValueError(
            "Real and predicted AnnData have different celltypes. "
            f"missing_in_pred={missing_in_pred}, missing_in_real={missing_in_real}"
        )

    skip_metrics = parse_skip_metrics(args.skip_metrics)
    celltypes = selected_from_backed or select_celltypes(real_celltypes, args)
    print(f"Selected {len(celltypes)} celltypes: {celltypes}")
    for idx, celltype in enumerate(celltypes, start=1):
        print(f"[{idx}/{len(celltypes)}] Evaluating celltype={celltype}")
        evaluator = MetricsEvaluator(
            adata_pred=pred_split[celltype].copy(),
            adata_real=real_split[celltype].copy(),
            control_pert=control_pert,
            pert_col=pert_col,
            num_threads=args.num_threads,
            outdir=outdir,
            allow_discrete=False,
            prefix=str(celltype),
        )

        evaluator.compute(
            profile="full",
            skip_metrics=skip_metrics,
            basename="results.csv",
        )

    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
