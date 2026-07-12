#!/usr/bin/env python3
"""Compute direct mean-level Replogle delta PCC from h5ad files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def read_x(handle: h5py.File, *, clip_min: float | None):
    x = handle["X"]
    if isinstance(x, h5py.Group):
        shape = h5ad_shape(handle)
        data = x["data"][:]
        if clip_min is not None:
            data = np.clip(data, clip_min, None)
        return sp.csr_matrix((data, x["indices"][:], x["indptr"][:]), shape=shape)
    arr = np.asarray(x[:])
    if clip_min is not None:
        arr = np.clip(arr, clip_min, None)
    return arr


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def row_mean(matrix, mask: np.ndarray) -> np.ndarray:
    subset = matrix[mask]
    mean = subset.mean(axis=0)
    return np.asarray(mean).ravel().astype(np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def load_means(
    path: Path,
    *,
    control_pert: str,
    clip_min: float | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if "gene" not in handle["obs"]:
            raise ValueError(f"{path} missing obs.gene")
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_x(handle, clip_min=clip_min)

    control_mask = genes == control_pert
    if not control_mask.any():
        raise ValueError(f"{path} has no control rows for {control_pert!r}")
    control_mean = row_mean(matrix, control_mask)

    means: dict[str, np.ndarray] = {}
    for gene in np.unique(genes[~control_mask]):
        means[str(gene)] = row_mean(matrix, genes == gene)
    return control_mean, means


def score_pair(
    real_path: Path,
    pred_path: Path,
    *,
    control_pert: str,
    clip_min: float | None,
) -> dict[str, object]:
    real_control, real_means = load_means(real_path, control_pert=control_pert, clip_min=clip_min)
    pred_control, pred_means = load_means(pred_path, control_pert=control_pert, clip_min=clip_min)
    common = sorted(set(real_means) & set(pred_means))
    if not common:
        raise ValueError(f"No common perturbations between {real_path} and {pred_path}")

    scores = []
    for gene in common:
        score = pearson(pred_means[gene] - pred_control, real_means[gene] - real_control)
        scores.append(score)
    arr = np.asarray(scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "num_common": len(common),
        "num_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "nan_as_zero_mean": float(np.nan_to_num(arr, nan=0.0).mean()),
    }


def resolve_step_path(workspace: Path, step_dir: str) -> Path:
    step_path = Path(step_dir)
    if not step_path.is_absolute():
        step_path = workspace / step_path
    return step_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--step-dir", required=True, help="Workspace step dir or absolute artifact dir.")
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--expected-mean", type=float, default=0.75)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Clip expression values before scoring; use 'nan' to disable clipping.",
    )
    args = parser.parse_args()

    step_path = resolve_step_path(args.workspace, args.step_dir)
    clip_min = None if np.isnan(args.clip_min) else float(args.clip_min)
    failures: list[str] = []
    means: list[float] = []
    for cell_line in args.cell_lines:
        real_path = step_path / f"replogle_real_{cell_line}.h5ad"
        pred_path = step_path / f"replogle_pred_{cell_line}.h5ad"
        if not real_path.exists() or not pred_path.exists():
            failures.append(f"missing h5ad pair for {cell_line}: {real_path}, {pred_path}")
            continue
        score = score_pair(real_path, pred_path, control_pert=args.control_pert, clip_min=clip_min)
        means.append(float(score["mean"]))
        print(
            f"{cell_line:6s} direct_delta_pcc_mean={score['mean']:.6f} "
            f"median={score['median']:.6f} nan_as_zero={score['nan_as_zero_mean']:.6f} "
            f"finite={score['num_finite']}/{score['num_common']}"
        )

    if means:
        macro = float(np.mean(means))
        print(f"macro_direct_delta_pcc_mean={macro:.6f}")
        min_allowed = float(args.expected_mean) - float(args.tolerance)
        max_allowed = float(args.expected_mean) + float(args.tolerance)
        if not (min_allowed <= macro <= max_allowed):
            failures.append(
                f"macro_direct_delta_pcc_mean={macro:.6f} outside "
                f"{args.expected_mean:.3f} +/- {args.tolerance:.3f}"
            )

    if failures:
        print("FAILED direct delta PCC check:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASSED direct delta PCC check")


if __name__ == "__main__":
    main()
