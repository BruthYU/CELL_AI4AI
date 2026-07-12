#!/usr/bin/env python3
"""Estimate PCC for swapping main_inference_replogle avg_delta payloads.

This script does not write h5ad files. It reads existing main_inference outputs
and compact other-cell memory predictions to estimate whether the additive
``main_pred + avg_delta`` path can reach the desired delta PCC.
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
DEFAULT_GLOBAL_AVG_DELTA = PROJECT_ROOT / "dsets" / "replogle" / "replogle_train_avg_delta.pkl"
DEFAULT_MEMORY_DIR = WORKSPACE / "replogle_othercell_model010_memory090_20260622" / "predictions"
DEFAULT_MAIN_STEPS = {
    "rpe1": "20260622_041404_PRIOR_LLM",
    "hepg2": "replogle_prior_avgdelta_hepg2",
    "jurkat": "replogle_prior_avgdelta_jurkat",
    "k562": "replogle_prior_avgdelta_k562",
}
INFERENCE_MAIN_STEPS = {
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


def row_mean(matrix, mask: np.ndarray, *, clip_min: float | None = None) -> np.ndarray:
    subset = matrix[mask]
    if clip_min is None:
        mean = subset.mean(axis=0)
        return np.asarray(mean).ravel().astype(np.float64)
    if sp.issparse(subset):
        subset = subset.copy()
        subset.data = np.clip(subset.data, clip_min, None)
        mean = subset.mean(axis=0)
        return np.asarray(mean).ravel().astype(np.float64)
    return np.clip(np.asarray(subset), clip_min, None).mean(axis=0).astype(np.float64)


def load_means(path: Path, *, control_pert: str, clip_min: float | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_x(handle)
    control_mask = genes == control_pert
    if not control_mask.any():
        raise ValueError(f"{path} has no {control_pert!r} rows")
    control = row_mean(matrix, control_mask, clip_min=clip_min)
    means = {
        str(gene): row_mean(matrix, genes == gene, clip_min=clip_min)
        for gene in np.unique(genes[~control_mask])
    }
    return control, means


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--global-avg-delta", type=Path, default=DEFAULT_GLOBAL_AVG_DELTA)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument(
        "--main-steps",
        choices=["prior-avgdelta", "inference"],
        default="prior-avgdelta",
        help="Existing full-cell main_inference outputs used as the real/model reference.",
    )
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--clip-min", type=float, default=0.0)
    args = parser.parse_args()

    with args.global_avg_delta.open("rb") as handle:
        global_payload = pickle.load(handle)
    global_delta = global_payload["delta"]
    global_idx = global_payload["pert_to_idx"]

    memory_scores: list[float] = []
    additive_scores: list[float] = []
    main_steps = DEFAULT_MAIN_STEPS if args.main_steps == "prior-avgdelta" else INFERENCE_MAIN_STEPS
    for cell_line in args.cell_lines:
        step = main_steps[cell_line]
        step_path = args.workspace / step
        real_path = step_path / f"replogle_real_{cell_line}.h5ad"
        main_pred_path = step_path / f"replogle_pred_{cell_line}.h5ad"
        memory_pred_path = args.memory_dir / f"replogle_pred_{cell_line}.h5ad"
        memory_real_path = args.memory_dir / f"replogle_real_{cell_line}.h5ad"

        real_control_clip, real_means_clip = load_means(real_path, control_pert=args.control_pert, clip_min=args.clip_min)
        main_control_clip, _ = load_means(main_pred_path, control_pert=args.control_pert, clip_min=args.clip_min)
        _, main_means_raw = load_means(main_pred_path, control_pert=args.control_pert, clip_min=None)
        memory_control_clip, memory_means_clip = load_means(memory_pred_path, control_pert=args.control_pert, clip_min=args.clip_min)

        if memory_real_path.exists():
            memory_real_control_clip, _ = load_means(
                memory_real_path,
                control_pert=args.control_pert,
                clip_min=args.clip_min,
            )
        else:
            memory_real_control_clip = real_control_clip

        common = sorted(set(real_means_clip) & set(main_means_raw) & set(memory_means_clip) & set(global_idx))
        memory_cell_scores = []
        additive_cell_scores = []
        for gene in common:
            true_delta = real_means_clip[gene] - real_control_clip
            other_delta = memory_means_clip[gene] - memory_control_clip
            memory_score = pearson(other_delta, true_delta)

            model_mean_raw = main_means_raw[gene] - global_delta[global_idx[gene]]
            additive_expr_clip = np.clip(model_mean_raw + other_delta, args.clip_min, None)
            additive_delta = additive_expr_clip - main_control_clip
            additive_score = pearson(additive_delta, true_delta)

            memory_cell_scores.append(memory_score)
            additive_cell_scores.append(additive_score)

        memory_arr = np.asarray(memory_cell_scores, dtype=np.float64)
        additive_arr = np.asarray(additive_cell_scores, dtype=np.float64)
        memory_mean = float(np.nanmean(memory_arr))
        additive_mean = float(np.nanmean(additive_arr))
        memory_scores.append(memory_mean)
        additive_scores.append(additive_mean)
        print(
            f"{cell_line:6s} memory_control_delta_pcc={memory_mean:.6f} "
            f"main_model_plus_memory_delta_pcc={additive_mean:.6f} "
            f"targets={len(common)}"
        )

    print(f"macro_memory_control_delta_pcc={float(np.nanmean(memory_scores)):.6f}")
    print(f"macro_main_model_plus_memory_delta_pcc={float(np.nanmean(additive_scores)):.6f}")


if __name__ == "__main__":
    main()
