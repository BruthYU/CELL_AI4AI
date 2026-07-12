#!/usr/bin/env python3
"""Apply main_inference-style Replogle correction vectors to full-cell h5ad files."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


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


def apply_correction(
    raw_pred_path: Path,
    out_pred_path: Path,
    correction_path: Path,
    *,
    chunk_rows: int,
    clip_min: float | None,
) -> dict[str, object]:
    with correction_path.open("rb") as handle:
        correction = pickle.load(handle)
    delta = np.asarray(correction["delta"], dtype=np.float32)
    pert_to_idx = dict(correction["pert_to_idx"])

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    if out_pred_path.exists() or out_pred_path.is_symlink():
        out_pred_path.unlink()

    applied_rows = 0
    missing_rows = 0
    with h5py.File(raw_pred_path, "r") as src, h5py.File(out_pred_path, "w") as dst:
        n_obs, n_vars = h5ad_shape(src)
        if int(delta.shape[1]) != n_vars:
            raise ValueError(f"{correction_path} dim={delta.shape[1]} but h5ad vars={n_vars}")
        genes = obs_column(src, "gene").astype(str)
        copy_non_x_groups(src, dst)

        src_x = src["X"]
        if not isinstance(src_x, h5py.Group) or not {"data", "indices", "indptr"} <= set(src_x.keys()):
            raise ValueError(f"{raw_pred_path} X is not a CSR h5ad group")
        src_indptr = src_x["indptr"][:].astype(np.int64, copy=False)
        src_indices = src_x["indices"]
        src_data = src_x["data"]

        out_x = dst.create_group("X")
        out_x.attrs["encoding-type"] = "csr_matrix"
        out_x.attrs["encoding-version"] = "0.1.0"
        out_x.attrs["shape"] = np.asarray([n_obs, n_vars], dtype=np.int64)
        data_ds = out_x.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            chunks=(1_000_000,),
            dtype=np.float32,
        )
        indices_ds = out_x.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            chunks=(1_000_000,),
            dtype=np.int32,
        )
        out_indptr = np.empty(n_obs + 1, dtype=np.int64)
        out_indptr[0] = 0

        nnz_so_far = 0
        for start in range(0, n_obs, chunk_rows):
            end = min(start + chunk_rows, n_obs)
            data_start = int(src_indptr[start])
            data_end = int(src_indptr[end])
            chunk_indptr = src_indptr[start : end + 1] - data_start
            chunk = sp.csr_matrix(
                (
                    src_data[data_start:data_end],
                    src_indices[data_start:data_end],
                    chunk_indptr,
                ),
                shape=(end - start, n_vars),
            )
            dense = chunk.toarray().astype(np.float32, copy=False)
            gene_chunk = genes[start:end]
            for gene in np.unique(gene_chunk):
                delta_idx = pert_to_idx.get(str(gene))
                if delta_idx is None:
                    missing_rows += int(np.count_nonzero(gene_chunk == gene))
                    continue
                mask = gene_chunk == gene
                dense[mask] += delta[int(delta_idx)]
                applied_rows += int(np.count_nonzero(mask))
            if clip_min is not None:
                np.maximum(dense, clip_min, out=dense)

            out_chunk = sp.csr_matrix(dense)
            append_1d(data_ds, out_chunk.data.astype(np.float32, copy=False))
            append_1d(indices_ds, out_chunk.indices.astype(np.int32, copy=False))
            out_indptr[start + 1 : end + 1] = out_chunk.indptr[1:] + nnz_so_far
            nnz_so_far += int(out_chunk.nnz)

        indptr_dtype = np.int32 if nnz_so_far <= np.iinfo(np.int32).max else np.int64
        out_x.create_dataset(
            "indptr",
            data=out_indptr.astype(indptr_dtype, copy=False),
            chunks=(min(n_obs + 1, 65536),),
        )

    return {
        "raw_pred_path": str(raw_pred_path),
        "out_pred_path": str(out_pred_path),
        "correction_path": str(correction_path),
        "rows": int(n_obs),
        "vars": int(n_vars),
        "applied_rows": int(applied_rows),
        "missing_rows": int(missing_rows),
        "nnz": int(nnz_so_far),
        "clip_min": clip_min,
    }


def replace_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--raw-step-prefix", default="replogle_remote_ep129_raw_")
    parser.add_argument("--correction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--clip-min",
        type=float,
        default=float("nan"),
        help="Optional output clipping. Default nan preserves main_inference-style un-clipped output.",
    )
    args = parser.parse_args()

    clip_min = None if np.isnan(args.clip_min) else float(args.clip_min)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        raw_dir = args.workspace / f"{args.raw_step_prefix}{cell_line}"
        raw_pred_path = raw_dir / f"replogle_pred_{cell_line}.h5ad"
        raw_real_path = raw_dir / f"replogle_real_{cell_line}.h5ad"
        correction_path = (
            args.correction_dir / f"replogle_train_avg_delta_main_correction_for_{cell_line}.pkl"
        )
        out_pred_path = args.out_dir / f"replogle_pred_{cell_line}.h5ad"
        out_real_path = args.out_dir / f"replogle_real_{cell_line}.h5ad"
        if not raw_pred_path.exists():
            raise FileNotFoundError(raw_pred_path)
        if not raw_real_path.exists():
            raise FileNotFoundError(raw_real_path)
        if not correction_path.exists():
            raise FileNotFoundError(correction_path)

        print(f"apply correction for {cell_line}: {raw_pred_path} -> {out_pred_path}", flush=True)
        summary = apply_correction(
            raw_pred_path,
            out_pred_path,
            correction_path,
            chunk_rows=int(args.chunk_rows),
            clip_min=clip_min,
        )
        replace_symlink(raw_real_path.resolve(), out_real_path)
        summary["out_real_path"] = str(out_real_path)
        summary["raw_real_path"] = str(raw_real_path.resolve())
        summaries.append(summary)
        print(
            f"wrote {out_pred_path} rows={summary['rows']} applied_rows={summary['applied_rows']} "
            f"missing_rows={summary['missing_rows']} nnz={summary['nnz']}",
            flush=True,
        )

    manifest_path = args.out_dir / "main_correction_full_h5ad_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
