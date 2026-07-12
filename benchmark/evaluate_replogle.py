from cell_eval import MetricsEvaluator
from cell_eval.data import build_random_anndata, downsample_cells
from cell_eval.utils import split_anndata_on_celltype
import anndata as ad
import numpy as np
from scipy.sparse import issparse
import argparse
import os


DEFAULT_STEP_DIR = "20260615_105743_JIT_LLM_REPLOGLE_V3_STATEALIGN"
DEFAULT_WORKSPACE = os.path.join(os.path.dirname(__file__), "workspace")
DEFAULT_SKIP_METRICS = ["pearson_edistance", "clustering_agreement"]
REPLOGLE_CELL_LINES = ["rpe1", "hepg2", "jurkat", "k562"]


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
        adata.X.data = np.clip(adata.X.data, 0, None)
    else:
        adata.X = np.clip(adata.X, 0, None)


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
    parser = argparse.ArgumentParser(description="Evaluate Replogle h5ad predictions.")
    parser.add_argument("--step-dir", default=DEFAULT_STEP_DIR)
    parser.add_argument(
        "--cell-line",
        default=None,
        choices=REPLOGLE_CELL_LINES,
        help="Optional. If omitted, infer from replogle_real/pred_<cell_line>.h5ad in step-dir.",
    )
    parser.add_argument(
        "--job-name",
        default="",
        help="Optional job name used only to disambiguate when step-dir contains multiple cell lines.",
    )
    parser.add_argument("--out-subfolder", default="results_calibrate")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--celltype-col", default=os.environ.get("CELLTYPE_COL", "celltype"))
    parser.add_argument(
        "--num-threads",
        type=int,
        default=int(os.environ.get("CELL_EVAL_NUM_THREADS", "32")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("CELL_EVAL_BATCH_SIZE", "100")),
        help="Batch size for parallel differential expression.",
    )
    parser.add_argument(
        "--skip-metrics",
        default=os.environ.get("CELL_EVAL_SKIP_METRICS", ",".join(DEFAULT_SKIP_METRICS)),
        help='Comma-separated metric names to skip. Use "" to run the full profile without skips.',
    )
    parser.add_argument(
        "--de-source-subfolder",
        default=os.environ.get("CELL_EVAL_DE_SOURCE_SUBFOLDER", ""),
        help=(
            "Optional output subfolder containing existing per-celltype real_de.csv "
            "and pred_de.csv files to reuse instead of recomputing DE."
        ),
    )
    return parser.parse_args()


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input does not exist: {path}")


def split_suffix(path, prefix):
    name = os.path.basename(path)
    if not name.startswith(prefix) or not name.endswith(".h5ad"):
        return None
    return name[len(prefix) : -len(".h5ad")]


def infer_replogle_pair(step_path, cell_line, job_name=""):
    if cell_line is not None:
        real_path = os.path.join(step_path, f"replogle_real_{cell_line}.h5ad")
        pred_path = os.path.join(step_path, f"replogle_pred_{cell_line}.h5ad")
        require_file(real_path)
        require_file(pred_path)
        return cell_line, real_path, pred_path

    real_by_cell = {}
    pred_by_cell = {}
    for name in os.listdir(step_path):
        path = os.path.join(step_path, name)
        if not os.path.isfile(path):
            continue

        real_cell = split_suffix(path, "replogle_real_")
        if real_cell:
            real_by_cell[real_cell] = path
            continue

        pred_cell = split_suffix(path, "replogle_pred_")
        if pred_cell:
            pred_by_cell[pred_cell] = path

    cell_lines = sorted(set(real_by_cell) & set(pred_by_cell))
    if not cell_lines:
        raise FileNotFoundError(
            "No matching Replogle real/pred h5ad pair found. Expected files like "
            f"replogle_real_<cell_line>.h5ad and replogle_pred_<cell_line>.h5ad in: {step_path}"
        )
    if len(cell_lines) > 1:
        job_name_lower = job_name.lower()
        matches = [candidate for candidate in cell_lines if candidate.lower() in job_name_lower]
        if len(matches) == 1:
            matched_cell_line = matches[0]
            return matched_cell_line, real_by_cell[matched_cell_line], pred_by_cell[matched_cell_line]

        raise ValueError(
            "Multiple Replogle cell-line pairs found in step-dir; pass --cell-line explicitly "
            "or include exactly one cell line in --job-name. "
            f"cell_lines={cell_lines}"
        )

    inferred_cell_line = cell_lines[0]
    return inferred_cell_line, real_by_cell[inferred_cell_line], pred_by_cell[inferred_cell_line]


def main():
    args = parse_args()
    step_path = os.path.abspath(os.path.join(args.workspace, args.step_dir))
    outdir = os.path.join(step_path, args.out_subfolder)
    os.makedirs(outdir, exist_ok=True)

    cell_line, real_path, pred_path = infer_replogle_pair(
        step_path,
        args.cell_line,
        job_name=args.job_name,
    )

    print("[1/3] Loading data...")
    print(f"step_path={step_path}")
    print(f"cell_line={cell_line}")
    real_adata = ad.read_h5ad(real_path)
    pred_adata = ad.read_h5ad(pred_path)

    print("[2/3] Clipping data...")
    clip_adata(real_adata)
    clip_adata(pred_adata)

    pert_col, control_pert = infer_perturbation_config(real_adata, pred_adata)
    print(f"Using perturbation config: pert_col={pert_col}, control_pert={control_pert}")

    args.celltype_col = resolve_celltype_col(real_adata, pred_adata, args.celltype_col)
    print(f"[3/3] Evaluating by {args.celltype_col} with DE metrics...")
    skip_metrics = [x.strip() for x in args.skip_metrics.split(",") if x.strip()]
    print(f"Cell-eval batch_size={args.batch_size}")
    print(f"Cell-eval skip_metrics={skip_metrics}")
    print(f"Cell-eval de_source_subfolder={args.de_source_subfolder or None}")
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

    celltypes = sorted(real_celltypes, key=str)
    for idx, celltype in enumerate(celltypes, start=1):
        celltype_name = str(celltype).replace("/", "-")
        celltype_outdir = os.path.join(outdir, celltype_name)
        os.makedirs(celltype_outdir, exist_ok=True)

        de_real = None
        de_pred = None
        if args.de_source_subfolder:
            de_source_dir = os.path.join(step_path, args.de_source_subfolder, celltype_name)
            de_real = os.path.join(de_source_dir, "real_de.csv")
            de_pred = os.path.join(de_source_dir, "pred_de.csv")
            require_file(de_real)
            require_file(de_pred)
            print(f"Reusing real DE from: {de_real}")
            print(f"Reusing pred DE from: {de_pred}")

        print(f"[{idx}/{len(celltypes)}] Evaluating celltype={celltype}")
        evaluator = MetricsEvaluator(
            adata_pred=pred_split[celltype].copy(),
            adata_real=real_split[celltype].copy(),
            de_pred=de_pred,
            de_real=de_real,
            control_pert=control_pert,
            pert_col=pert_col,
            num_threads=args.num_threads,
            batch_size=args.batch_size,
            outdir=celltype_outdir,
            allow_discrete=False,
        )

        evaluator.compute(
            profile="full",
            skip_metrics=skip_metrics or None,
        )

    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
