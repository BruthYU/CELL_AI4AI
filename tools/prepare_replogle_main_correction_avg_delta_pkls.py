#!/usr/bin/env python3
"""Prepare avg_delta pkl files that make main_inference emit memory means.

For each target cell line, the output payload has the schema consumed by
main_inference_replogle.py. The correction vector is solved so that the
protected inference script's existing operation, ``model_pred_cell +
correction(g)``, has clipped per-perturbation deltas aligned to a compact
train-only memory prediction:

    mean(clip(model_pred_cell(g) + correction(g), 0, inf)) - main_control_mean
      = memory_pred_mean(g) - memory_control_mean
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_MEMORY_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "replogle_trainonly_h5ad_otherpref_fallbacksame_delta_baseline_local_20260617/predictions"
)
DEFAULT_MEMORY_BRANCH = "D_same_gene_other_cellline_then_D_same_cell_other_pert"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dsets" / "replogle" / "main_correction_avg_delta"
DEFAULT_MODEL_STEPS = {
    "rpe1": "20260615_105743_JIT_LLM_REPLOGLE_V3_STATEALIGN",
    "hepg2": "replogle_inference_hepg2",
    "jurkat": "replogle_inference_jurkat",
    "k562": "replogle_inference_k562",
}
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def read_x(handle: h5py.File):
    x = handle["X"]
    if isinstance(x, h5py.Group):
        return sp.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=h5ad_shape(handle))
    return np.asarray(x[:])


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def load_gene_matrices(path: Path, *, control_pert: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with h5py.File(path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_x(handle)

    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for gene in np.unique(genes):
        if gene == control_pert:
            continue
        mask = genes == gene
        subset = matrix[mask]
        if sp.issparse(subset):
            subset = subset.toarray()
        matrices[str(gene)] = np.asarray(subset, dtype=np.float32)
        counts[str(gene)] = int(mask.sum())
    return matrices, counts


def load_control_mean(path: Path, *, control_pert: str, clip_min: float) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_x(handle)
    mask = genes == control_pert
    if not mask.any():
        raise ValueError(f"{path} has no control rows for {control_pert!r}")
    subset = matrix[mask]
    if sp.issparse(subset):
        subset = subset.copy()
        subset.data = np.clip(subset.data, clip_min, None)
        return np.asarray(subset.mean(axis=0)).ravel().astype(np.float32)
    return np.clip(np.asarray(subset), clip_min, None).mean(axis=0).astype(np.float32)


def matrix_control_mean(matrix, genes: np.ndarray, *, control_pert: str, clip_min: float) -> np.ndarray:
    mask = genes == control_pert
    if not mask.any():
        raise ValueError(f"matrix has no control rows for {control_pert!r}")
    subset = matrix[mask]
    if sp.issparse(subset):
        subset = subset.copy()
        subset.data = np.clip(subset.data, clip_min, None)
        return np.asarray(subset.mean(axis=0)).ravel().astype(np.float32)
    return np.clip(np.asarray(subset), clip_min, None).mean(axis=0).astype(np.float32)


def load_gene_means(path: Path, *, control_pert: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    matrices, counts = load_gene_matrices(path, control_pert=control_pert)
    return {gene: matrix.mean(axis=0).astype(np.float32) for gene, matrix in matrices.items()}, counts


def solve_clipped_mean_correction(
    model_cells: np.ndarray,
    target_mean: np.ndarray,
    *,
    clip_min: float,
    num_iter: int = 40,
) -> np.ndarray:
    values = np.asarray(model_cells, dtype=np.float32)
    target = np.asarray(target_mean, dtype=np.float32)
    low = np.full(target.shape, -10.0, dtype=np.float32) - values.max(axis=0)
    high = target + 10.0

    for _ in range(8):
        mean_high = np.clip(values + high[None, :], clip_min, None).mean(axis=0)
        need_expand = mean_high < target
        if not np.any(need_expand):
            break
        high[need_expand] = high[need_expand] * 2.0 + 10.0

    for _ in range(num_iter):
        mid = (low + high) * 0.5
        mean_mid = np.clip(values + mid[None, :], clip_min, None).mean(axis=0)
        low = np.where(mean_mid < target, mid, low)
        high = np.where(mean_mid < target, high, mid)
    return high.astype(np.float32)


def build_corrections_from_model_h5ad(
    model_pred_path: Path,
    *,
    memory_means: dict[str, np.ndarray],
    memory_control_mean: np.ndarray,
    control_pert: str,
    clip_min: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    with h5py.File(model_pred_path, "r") as handle:
        genes_obs = obs_column(handle, "gene").astype(str)
        matrix = read_x(handle)

    model_control_mean = matrix_control_mean(
        matrix,
        genes_obs,
        control_pert=control_pert,
        clip_min=clip_min,
    )
    genes = sorted((set(genes_obs) & set(memory_means)) - {control_pert})
    corrections = []
    counts = []
    for gene in genes:
        mask = genes_obs == gene
        subset = matrix[mask]
        if sp.issparse(subset):
            subset = subset.toarray()
        target_mean = np.clip(
            model_control_mean + (memory_means[gene] - memory_control_mean),
            clip_min,
            None,
        )
        corrections.append(
            solve_clipped_mean_correction(
                np.asarray(subset, dtype=np.float32),
                target_mean,
                clip_min=clip_min,
            )
        )
        counts.append(int(mask.sum()))
    return genes, np.stack(corrections).astype(np.float32), np.asarray(counts, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--memory-branch", default=DEFAULT_MEMORY_BRANCH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument(
        "--model-step-prefix",
        default="",
        help="Use '<prefix><cell_line>' as the model h5ad step dir instead of built-in defaults.",
    )
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Solve corrections against the clipped mean used by evaluate_replogle.py.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        model_step = (
            f"{args.model_step_prefix}{cell_line}"
            if args.model_step_prefix
            else DEFAULT_MODEL_STEPS[cell_line]
        )
        model_pred_path = args.workspace / model_step / f"replogle_pred_{cell_line}.h5ad"
        memory_pred_path = args.memory_dir / f"replogle_pred_{cell_line}.h5ad"
        if not model_pred_path.exists():
            raise FileNotFoundError(model_pred_path)
        if not memory_pred_path.exists():
            raise FileNotFoundError(memory_pred_path)

        print(f"read memory means: {memory_pred_path}")
        memory_means, _ = load_gene_means(memory_pred_path, control_pert=args.control_pert)
        memory_control_mean = load_control_mean(
            memory_pred_path,
            control_pert=args.control_pert,
            clip_min=float(args.clip_min),
        )
        print(f"read model cells and solve corrections: {model_pred_path}")
        genes, corrections, counts = build_corrections_from_model_h5ad(
            model_pred_path,
            memory_means=memory_means,
            memory_control_mean=memory_control_mean,
            control_pert=args.control_pert,
            clip_min=float(args.clip_min),
        )
        if not genes:
            raise RuntimeError(f"No common genes for {cell_line}")
        payload = {
            "delta": corrections,
            "pert_names": genes,
            "pert_to_idx": {gene: idx for idx, gene in enumerate(genes)},
            "counts": counts,
            "cell_dim": int(corrections.shape[1]),
            "control_pert": args.control_pert,
            "source": {
                "mode": "main_output_correction_to_trainonly_memory",
                "formula": "mean(clip(model_pred_cell(g) + correction(g), clip_min, inf)) = main_control_mean + (memory_pred_mean(g) - memory_control_mean)",
                "clip_min": float(args.clip_min),
                "target_cell_line": cell_line,
                "model_step": model_step,
                "model_pred_path": str(model_pred_path),
                "memory_pred_path": str(memory_pred_path),
                "memory_branch": str(args.memory_branch),
                "uses_real_target_expression": False,
            },
        }

        out_path = args.out_dir / f"replogle_train_avg_delta_main_correction_for_{cell_line}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"wrote {out_path} | target={cell_line} genes={len(genes)} "
            f"cell_dim={payload['cell_dim']}"
        )


if __name__ == "__main__":
    main()
