import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PBMC real/pred h5ad outputs.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing pbmc_real_<split>.h5ad and pbmc_pred_<split>.h5ad.",
    )
    parser.add_argument(
        "--out-subfolder",
        default="results_calibrate",
        help="Subfolder under input-dir for evaluation outputs.",
    )
    parser.add_argument(
        "--celltype-col",
        default="celltype",
        help="AnnData obs column used to split PBMC cell types.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of threads for cell_eval metrics.",
    )
    parser.add_argument(
        "--skip-metrics",
        default="pearson_edistance,clustering_agreement",
        help="Comma-separated metrics to skip. Use an empty string to skip none.",
    )
    return parser.parse_args()


def split_suffix(path, prefix):
    name = os.path.basename(path)
    if not name.startswith(prefix) or not name.endswith(".h5ad"):
        return None
    return name[len(prefix) : -len(".h5ad")]


def infer_pbmc_pair(input_dir):
    real_by_split = {}
    pred_by_split = {}

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue

        real_split = split_suffix(path, "pbmc_real_")
        if real_split:
            real_by_split[real_split] = path
            continue

        pred_split = split_suffix(path, "pbmc_pred_")
        if pred_split:
            pred_by_split[pred_split] = path

    split_names = sorted(set(real_by_split) & set(pred_by_split))
    if not split_names:
        raise FileNotFoundError(
            "No matching PBMC real/pred h5ad pair found. Expected files like "
            f"pbmc_real_<split>.h5ad and pbmc_pred_<split>.h5ad in: {input_dir}"
        )
    if len(split_names) > 1:
        raise ValueError(
            "Multiple PBMC split pairs found in input-dir; keep one split per directory. "
            f"split_names={split_names}"
        )

    split_name = split_names[0]
    return split_name, real_by_split[split_name], pred_by_split[split_name]


def parse_skip_metrics(value):
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def clip_adata(adata):
    import numpy as np
    from scipy.sparse import issparse

    if issparse(adata.X):
        adata.X.data = np.clip(adata.X.data, 0, 14.99)
    else:
        adata.X = np.clip(adata.X, 0, 14.99)


def main():
    args = parse_args()
    import anndata as ad
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input dir does not exist: {input_dir}")

    split_name, real_path, pred_path = infer_pbmc_pair(input_dir)
    outdir = os.path.join(input_dir, args.out_subfolder)
    os.makedirs(outdir, exist_ok=True)

    print("[1/3] Loading data...")
    print(f"Input dir: {input_dir}")
    print(f"Split: {split_name}")
    print(f"Real h5ad: {real_path}")
    print(f"Pred h5ad: {pred_path}")
    real_adata = ad.read_h5ad(real_path)
    pred_adata = ad.read_h5ad(pred_path)

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
    celltypes = sorted(real_celltypes, key=str)
    for idx, celltype in enumerate(celltypes, start=1):
        celltype_name = str(celltype).replace("/", "-")
        celltype_outdir = os.path.join(outdir, celltype_name)
        os.makedirs(celltype_outdir, exist_ok=True)

        print(f"[{idx}/{len(celltypes)}] Evaluating celltype={celltype}")
        evaluator = MetricsEvaluator(
            adata_pred=pred_split[celltype].copy(),
            adata_real=real_split[celltype].copy(),
            control_pert=control_pert,
            pert_col=pert_col,
            num_threads=args.num_threads,
            outdir=celltype_outdir,
            allow_discrete=False,
        )

        evaluator.compute(profile="full", skip_metrics=skip_metrics)

    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
