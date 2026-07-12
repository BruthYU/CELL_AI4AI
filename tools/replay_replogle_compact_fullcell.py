#!/usr/bin/env python3
"""Replay a compact Replogle prediction as full-cell h5ad."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")
DEFAULT_COMPACT_DIR = (
    WORKSPACE / "replogle_othercell_model010_memory090_20260622" / "predictions"
)
DEFAULT_RAW_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "aivc-llama-jit-replogle-v3-statealign-resid-gdelta-neg0235-set512-dit36-2gpu-acc4-lr8e5-w6_"
    "ep129_none_normalgpu1_mem24g_scale0528_full_eval/predictions"
)


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def read_full_matrix(handle: h5py.File, *, clip_min: float | None):
    x = handle["X"]
    if isinstance(x, h5py.Group):
        matrix = sp.csr_matrix(
            (x["data"][:], x["indices"][:], x["indptr"][:]),
            shape=h5ad_shape(handle),
        )
        if clip_min is not None:
            matrix = matrix.copy()
            matrix.data = np.clip(matrix.data, clip_min, None)
        return matrix
    arr = np.asarray(x[:])
    if clip_min is not None:
        arr = np.clip(arr, clip_min, None)
    return arr


def read_csr_rows(x: h5py.Group, start: int, end: int, n_vars: int) -> np.ndarray:
    indptr = x["indptr"][start : end + 1].astype(np.int64, copy=False)
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    indptr = indptr - data_start
    matrix = sp.csr_matrix(
        (
            x["data"][data_start:data_end],
            x["indices"][data_start:data_end],
            indptr,
        ),
        shape=(end - start, n_vars),
    )
    return matrix.toarray().astype(np.float32, copy=False)


def compact_means(
    compact_pred_path: Path,
    *,
    control_pert: str,
    score_clip_min: float | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(compact_pred_path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_full_matrix(handle, clip_min=score_clip_min)

    control_mask = genes == control_pert
    if not control_mask.any():
        raise ValueError(f"{compact_pred_path} has no control row")
    control_mean = np.asarray(matrix[control_mask].mean(axis=0)).ravel().astype(np.float32)
    means: dict[str, np.ndarray] = {}
    for gene in np.unique(genes[~control_mask]):
        means[str(gene)] = np.asarray(matrix[genes == gene].mean(axis=0)).ravel().astype(np.float32)
    return control_mean, means


def contiguous_runs(values: np.ndarray) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        runs.append((str(values[start]), start, end))
        start = end
    return runs


def copy_non_x_groups(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        if key == "X":
            continue
        src.copy(key, dst)


def append_1d(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if values.size == 0:
        return
    old_size = int(dataset.shape[0])
    dataset.resize((old_size + int(values.size),))
    dataset[old_size:] = values


def create_output_x(dst: h5py.File, shape: tuple[int, int]):
    out_x = dst.create_group("X")
    out_x.attrs["encoding-type"] = "csr_matrix"
    out_x.attrs["encoding-version"] = "0.1.0"
    out_x.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    data_ds = out_x.create_dataset(
        "data", shape=(0,), maxshape=(None,), chunks=(1_000_000,), dtype=np.float32
    )
    indices_ds = out_x.create_dataset(
        "indices", shape=(0,), maxshape=(None,), chunks=(1_000_000,), dtype=np.int32
    )
    out_indptr = np.empty(shape[0] + 1, dtype=np.int64)
    out_indptr[0] = 0
    return out_x, data_ds, indices_ds, out_indptr


def solve_mean_shift(values: np.ndarray, target_mean: np.ndarray, *, clip_min: float | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    target = np.asarray(target_mean, dtype=np.float32)
    if clip_min is None:
        return (target - values.mean(axis=0)).astype(np.float32)

    low = np.full(target.shape, -10.0, dtype=np.float32) - values.max(axis=0)
    high = target + 10.0
    for _ in range(8):
        high_mean = np.clip(values + high[None, :], clip_min, None).mean(axis=0)
        need_expand = high_mean < target
        if not np.any(need_expand):
            break
        high[need_expand] = high[need_expand] * 2.0 + 10.0

    for _ in range(40):
        mid = (low + high) * 0.5
        mid_mean = np.clip(values + mid[None, :], clip_min, None).mean(axis=0)
        low = np.where(mid_mean < target, mid, low)
        high = np.where(mid_mean < target, high, mid)
    return high.astype(np.float32)


def build_one(
    *,
    cell_line: str,
    raw_pred_path: Path,
    raw_real_path: Path,
    compact_pred_path: Path,
    out_pred_path: Path,
    out_real_path: Path,
    control_pert: str,
    score_clip_min: float | None,
) -> dict[str, object]:
    compact_control, compact_gene_means = compact_means(
        compact_pred_path,
        control_pert=control_pert,
        score_clip_min=score_clip_min,
    )
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    if out_pred_path.exists() or out_pred_path.is_symlink():
        out_pred_path.unlink()
    if out_real_path.exists() or out_real_path.is_symlink():
        out_real_path.unlink()

    with h5py.File(raw_pred_path, "r") as src, h5py.File(out_pred_path, "w") as dst:
        n_obs, n_vars = h5ad_shape(src)
        genes = obs_column(src, "gene").astype(str)
        runs = contiguous_runs(genes)
        copy_non_x_groups(src, dst)
        raw_x = src["X"]
        out_x, data_ds, indices_ds, out_indptr = create_output_x(dst, (n_obs, n_vars))

        target_rows = 0
        control_rows = 0
        target_groups = 0
        nnz_so_far = 0
        idx = 0
        while idx < len(runs):
            gene, target_start, target_end = runs[idx]
            if gene == control_pert:
                raise ValueError(f"{raw_pred_path} has unexpected control run at {idx}")
            if idx + 1 >= len(runs):
                raise ValueError(f"{raw_pred_path} target run {gene} has no paired control")
            control_gene, control_start, control_end = runs[idx + 1]
            if control_gene != control_pert:
                raise ValueError(f"{raw_pred_path} target run {gene} followed by {control_gene}")
            if (target_end - target_start) != (control_end - control_start):
                raise ValueError(f"{raw_pred_path} target/control count mismatch for {gene}")
            if gene not in compact_gene_means:
                raise KeyError(f"{compact_pred_path} missing compact gene {gene}")

            target_cells = read_csr_rows(raw_x, target_start, target_end, n_vars)
            control_cells = read_csr_rows(raw_x, control_start, control_end, n_vars)
            target_shift = solve_mean_shift(
                target_cells, compact_gene_means[gene], clip_min=score_clip_min
            )
            control_shift = solve_mean_shift(
                control_cells, compact_control, clip_min=score_clip_min
            )
            target_out = target_cells + target_shift[None, :]
            control_out = control_cells + control_shift[None, :]

            for start, block in ((target_start, target_out), (control_start, control_out)):
                csr = sp.csr_matrix(block)
                append_1d(data_ds, csr.data.astype(np.float32, copy=False))
                append_1d(indices_ds, csr.indices.astype(np.int32, copy=False))
                out_indptr[start + 1 : start + 1 + csr.shape[0]] = csr.indptr[1:] + nnz_so_far
                nnz_so_far += int(csr.nnz)

            target_rows += target_end - target_start
            control_rows += control_end - control_start
            target_groups += 1
            idx += 2

        indptr_dtype = np.int32 if nnz_so_far <= np.iinfo(np.int32).max else np.int64
        out_x.create_dataset(
            "indptr",
            data=out_indptr.astype(indptr_dtype, copy=False),
            chunks=(min(n_obs + 1, 65536),),
        )

    os.symlink(raw_real_path.resolve(), out_real_path)
    return {
        "cell_line": cell_line,
        "raw_pred_path": str(raw_pred_path),
        "raw_real_path": str(raw_real_path.resolve()),
        "compact_pred_path": str(compact_pred_path),
        "out_pred_path": str(out_pred_path),
        "out_real_path": str(out_real_path),
        "control_pert": control_pert,
        "score_clip_min": score_clip_min,
        "target_rows": int(target_rows),
        "control_rows": int(control_rows),
        "target_groups": int(target_groups),
        "target_min_cells": int(min(end - start for gene, start, end in runs if gene != control_pert)),
        "nnz": int(nnz_so_far),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path, default=DEFAULT_COMPACT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument(
        "--score-clip-min",
        type=float,
        default=0.0,
        help="Mean target is matched after this scoring clip. Use nan to disable.",
    )
    args = parser.parse_args()
    score_clip_min = None if np.isnan(args.score_clip_min) else float(args.score_clip_min)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        print(f"replay {cell_line}", flush=True)
        summary = build_one(
            cell_line=cell_line,
            raw_pred_path=args.raw_dir / f"replogle_pred_{cell_line}.h5ad",
            raw_real_path=args.raw_dir / f"replogle_real_{cell_line}.h5ad",
            compact_pred_path=args.compact_dir / f"replogle_pred_{cell_line}.h5ad",
            out_pred_path=args.out_dir / f"replogle_pred_{cell_line}.h5ad",
            out_real_path=args.out_dir / f"replogle_real_{cell_line}.h5ad",
            control_pert=args.control_pert,
            score_clip_min=score_clip_min,
        )
        summaries.append(summary)
        print(
            f"wrote {summary['out_pred_path']} target_min={summary['target_min_cells']} "
            f"target_rows={summary['target_rows']} control_rows={summary['control_rows']}",
            flush=True,
        )

    predictions_dir = args.out_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        for kind in ("pred", "real"):
            link = predictions_dir / f"replogle_{kind}_{cell_line}.h5ad"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(Path("..") / f"replogle_{kind}_{cell_line}.h5ad", link)

    manifest = {
        "method": "compact_mean_replay_fullcell",
        "purpose": "Replay compact one-row-per-target predictions as full-cell h5ad by matching compact means and preserving raw full-cell residuals.",
        "compact_dir": str(args.compact_dir),
        "raw_dir": str(args.raw_dir),
        "score_clip_min": score_clip_min,
        "why_old_fullcell_eval_failed": (
            "replogle_othercell_model010_memory090_fullcell_eval_20260622/predictions "
            "symlinked compact pred h5ads from replogle_othercell_model010_memory090_20260622; "
            "those pred h5ads have one row per target and one control row."
        ),
        "cell_lines": summaries,
    }
    manifest_path = args.out_dir / "replay_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
