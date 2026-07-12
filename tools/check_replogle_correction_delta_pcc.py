#!/usr/bin/env python3
"""Simulate main_inference avg_delta correction and score direct delta PCC."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_STEPS = {
    "rpe1": "20260615_105743_JIT_LLM_REPLOGLE_V3_STATEALIGN",
    "hepg2": "replogle_inference_hepg2",
    "jurkat": "replogle_inference_jurkat",
    "k562": "replogle_inference_k562",
}
DEFAULT_CORRECTION_DIR = PROJECT_ROOT / "dsets" / "replogle" / "main_correction_avg_delta"


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
        return np.asarray(subset.mean(axis=0)).ravel().astype(np.float64)
    if sp.issparse(subset):
        subset = subset.copy()
        subset.data = np.clip(subset.data, clip_min, None)
        return np.asarray(subset.mean(axis=0)).ravel().astype(np.float64)
    return np.clip(np.asarray(subset), clip_min, None).mean(axis=0).astype(np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def score_cell_line(
    *,
    cell_line: str,
    step_path: Path,
    correction_path: Path,
    control_pert: str,
    clip_min: float,
) -> dict[str, float | int]:
    with correction_path.open("rb") as handle:
        payload = pickle.load(handle)
    corr = payload["delta"]
    corr_idx = payload["pert_to_idx"]

    pred_path = step_path / f"replogle_pred_{cell_line}.h5ad"
    real_path = step_path / f"replogle_real_{cell_line}.h5ad"
    with h5py.File(pred_path, "r") as pred_handle:
        pred_genes = obs_column(pred_handle, "gene").astype(str)
        pred_x = read_x(pred_handle)

    pred_control = row_mean(pred_x, pred_genes == control_pert, clip_min=clip_min)
    corrected_means: dict[str, np.ndarray] = {}
    common_pred_genes = sorted((set(pred_genes) & set(corr_idx)) - {control_pert})
    for gene in common_pred_genes:
        correction = corr[corr_idx[gene]].astype(np.float64)
        pred_subset = pred_x[pred_genes == gene]
        if sp.issparse(pred_subset):
            pred_subset = pred_subset.toarray()
        corrected_means[gene] = np.clip(np.asarray(pred_subset, dtype=np.float64) + correction, clip_min, None).mean(axis=0)

    del pred_x
    del pred_genes

    with h5py.File(real_path, "r") as real_handle:
        real_genes = obs_column(real_handle, "gene").astype(str)
        real_x = read_x(real_handle)

    real_control = row_mean(real_x, real_genes == control_pert, clip_min=clip_min)
    scores = []
    for gene in sorted(set(corrected_means) & set(real_genes)):
        corrected_mean = corrected_means[gene]
        real_mean = row_mean(real_x, real_genes == gene, clip_min=clip_min)
        scores.append(pearson(corrected_mean - pred_control, real_mean - real_control))
    arr = np.asarray(scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "num_scores": int(arr.size),
        "num_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--cell-lines", nargs="+", required=True)
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--expected-mean", type=float, default=0.75)
    parser.add_argument("--tolerance", type=float, default=0.08)
    args = parser.parse_args()

    means = []
    failures = []
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        step_path = args.workspace / DEFAULT_STEPS[cell_line]
        correction_path = args.correction_dir / f"replogle_train_avg_delta_main_correction_for_{cell_line}.pkl"
        if not correction_path.exists():
            failures.append(f"missing correction pkl: {correction_path}")
            continue
        score = score_cell_line(
            cell_line=cell_line,
            step_path=step_path,
            correction_path=correction_path,
            control_pert=args.control_pert,
            clip_min=args.clip_min,
        )
        means.append(float(score["mean"]))
        print(
            f"{cell_line:6s} simulated_corrected_delta_pcc_mean={score['mean']:.6f} "
            f"median={score['median']:.6f} finite={score['num_finite']}/{score['num_scores']}"
        )

    if means:
        macro = float(np.mean(means))
        print(f"macro_simulated_corrected_delta_pcc_mean={macro:.6f}")
        if not (args.expected_mean - args.tolerance <= macro <= args.expected_mean + args.tolerance):
            failures.append(
                f"macro_simulated_corrected_delta_pcc_mean={macro:.6f} outside "
                f"{args.expected_mean:.3f} +/- {args.tolerance:.3f}"
            )
    if failures:
        print("FAILED simulated correction check:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASSED simulated correction check")


if __name__ == "__main__":
    main()
