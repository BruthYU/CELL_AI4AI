#!/usr/bin/env python3
"""Build PBMC LMDB shards from raw h5ad using a state TOML split.

This replaces the ad hoc notebook split builder for PBMC. It follows the state
TOML semantics: perturbations listed under ``val`` go only to validation,
perturbations listed under ``test`` go only to test, and everything else goes
to train. Control cells are written to a separate control LMDB.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import lmdb
import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pbmc_delta_memory_calibration import (  # noqa: E402
    _decode,
    _obs_codes,
    _parse_pbmc_split_toml,
    _split_for_group,
    _x_shape,
)


DEFAULT_RAW_H5AD = (
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/"
    "preprocessing/arcinstitute/datasets/State_Parse_Filtered/only_hvg/PBMC_only_hvg.h5ad"
)
DEFAULT_SPLIT_TOML = (
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/state-main/"
    "state_preprint_toml_files/parse_tomls/split_5.toml"
)
DEFAULT_OUT_ROOT = (
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/"
    "preprocessing/arcinstitute/datasets/State_Parse_Filtered/few_shot/split_5_state"
)


class SplitWriter:
    def __init__(self, path: Path, map_size: int, batch_size: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.batch_size = batch_size
        self.env = lmdb.open(
            str(path),
            map_size=map_size,
            subdir=True,
            create=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_dbs=1,
            max_readers=1024,
        )
        self.count = 0
        self.pending: list[tuple[bytes, bytes]] = []

    def put(self, value: Any) -> None:
        key = str(self.count).encode("utf-8")
        self.pending.append((key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)))
        self.count += 1
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        with self.env.begin(write=True) as txn:
            for key, value in self.pending:
                txn.put(key, value)
        self.pending.clear()

    def close(self) -> None:
        self.flush()
        with self.env.begin(write=True) as txn:
            txn.put(b"__len__", str(self.count).encode("utf-8"))
        self.env.sync()
        self.env.close()


class ControlWriter:
    def __init__(self, path: Path, map_size: int, batch_size: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.batch_size = batch_size
        self.env = lmdb.open(
            str(path),
            map_size=map_size,
            subdir=True,
            create=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_dbs=1,
            max_readers=1024,
        )
        self.keys: list[tuple[str, str, str]] = []
        self.pending: list[tuple[bytes, bytes]] = []

    def put(self, key_tuple: tuple[str, str, str], matrix: sp.csr_matrix) -> None:
        self.keys.append(key_tuple)
        self.pending.append(
            (
                str(key_tuple).encode("utf-8"),
                pickle.dumps(matrix, protocol=pickle.HIGHEST_PROTOCOL),
            )
        )
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        with self.env.begin(write=True) as txn:
            for key, value in self.pending:
                txn.put(key, value)
        self.pending.clear()

    def close(self) -> None:
        self.flush()
        with self.env.begin(write=True) as txn:
            txn.put(b"__len__", str(len(self.keys)).encode("utf-8"))
            txn.put(b"__keys__", pickle.dumps(self.keys, protocol=pickle.HIGHEST_PROTOCOL))
        self.env.sync()
        self.env.close()


def _csr_rows_from_h5(
    x_group: h5py.Group,
    row_indices: np.ndarray,
    *,
    n_vars: int,
) -> sp.csr_matrix:
    data_ds = x_group["data"]
    indices_ds = x_group["indices"]
    indptr = x_group["indptr"]
    row_indices = np.asarray(row_indices, dtype=np.int64)
    nnz_per_row = np.empty(len(row_indices), dtype=np.int64)
    for out_idx, row_idx in enumerate(row_indices):
        nnz_per_row[out_idx] = int(indptr[row_idx + 1]) - int(indptr[row_idx])

    out_indptr = np.empty(len(row_indices) + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(nnz_per_row, out=out_indptr[1:])
    total_nnz = int(out_indptr[-1])
    out_data = np.empty(total_nnz, dtype=data_ds.dtype)
    out_indices = np.empty(total_nnz, dtype=indices_ds.dtype)

    cursor = 0
    for row_idx, nnz in zip(row_indices, nnz_per_row, strict=True):
        if nnz == 0:
            continue
        start = int(indptr[row_idx])
        end = start + int(nnz)
        out_data[cursor : cursor + nnz] = data_ds[start:end]
        out_indices[cursor : cursor + nnz] = indices_ds[start:end]
        cursor += int(nnz)

    return sp.csr_matrix((out_data, out_indices, out_indptr), shape=(len(row_indices), n_vars))


def _pad_for_shard_size(matrix: sp.csr_matrix, shard_size: int, rng: np.random.Generator) -> sp.csr_matrix:
    n_rows = matrix.shape[0]
    pad_n = (-n_rows) % shard_size
    if pad_n == 0:
        return matrix
    pad_idx = rng.integers(0, n_rows, size=pad_n)
    return sp.vstack([matrix, matrix[pad_idx]], format="csr")


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_h5ad = Path(args.raw_h5ad)
    split_toml = Path(args.split_toml)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fewshot = _parse_pbmc_split_toml(split_toml)
    rng = np.random.default_rng(args.padding_seed)
    start_time = time.time()

    with h5py.File(raw_h5ad, "r") as f:
        n_rows, n_vars = _x_shape(f)
        x = f["X"]
        if not isinstance(x, h5py.Group) or x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError(f"{raw_h5ad} must store X as csr_matrix")

        donor_codes, donors = _obs_codes(f, "donor")
        cell_codes, celltypes = _obs_codes(f, "cell_type")
        pert_codes, perts = _obs_codes(f, "cytokine")

        group_lookup: dict[tuple[str, str, str], int] = {}
        group_keys: list[tuple[str, str, str]] = []
        group_ids = np.empty(n_rows, dtype=np.int32)
        counts: list[int] = []
        print(f"[build] grouping obs rows: {n_rows:,}", flush=True)
        for row_idx in range(n_rows):
            key = (
                donors[int(donor_codes[row_idx])],
                celltypes[int(cell_codes[row_idx])],
                perts[int(pert_codes[row_idx])],
            )
            group_id = group_lookup.get(key)
            if group_id is None:
                group_id = len(group_keys)
                group_lookup[key] = group_id
                group_keys.append(key)
                counts.append(0)
            group_ids[row_idx] = group_id
            counts[group_id] += 1
            if row_idx and row_idx % args.progress_every == 0:
                print(
                    f"[build] grouped {row_idx:,}/{n_rows:,}, groups={len(group_keys):,}, "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

        counts_arr = np.asarray(counts, dtype=np.int64)
        processed = np.ones(len(group_keys), dtype=bool)
        for group_id, (_, _, cytokine) in enumerate(group_keys):
            if cytokine != args.control_pert and counts_arr[group_id] <= args.min_pert_cells:
                processed[group_id] = False

        split_by_group: list[str] = []
        split_group_counts = {"train": 0, "val": 0, "test": 0, "control": 0, "dropped": 0}
        for group_id, (_, celltype, cytokine) in enumerate(group_keys):
            if not processed[group_id]:
                split = "dropped"
            elif cytokine == args.control_pert:
                split = "control"
            else:
                split = _split_for_group(celltype, cytokine, fewshot)
            split_by_group.append(split)
            split_group_counts[split] = split_group_counts.get(split, 0) + 1

        print(f"[build] split group counts: {split_group_counts}", flush=True)
        order = np.argsort(group_ids, kind="stable")
        boundaries = np.searchsorted(group_ids[order], np.arange(len(group_keys) + 1), side="left")

        map_size = int(args.map_size_gb * (1 << 30))
        writers = {
            "train": SplitWriter(out_root / f"pbmc_train_{args.split_name}", map_size, args.write_batch_size),
            "val": SplitWriter(out_root / f"pbmc_val_{args.split_name}", map_size, args.write_batch_size),
            "test": SplitWriter(out_root / f"pbmc_test_{args.split_name}", map_size, args.write_batch_size),
        }
        control_writer = ControlWriter(
            out_root / f"pbmc_control_{args.split_name}",
            map_size,
            args.write_batch_size,
        )

        try:
            for group_id, key in enumerate(group_keys):
                split = split_by_group[group_id]
                if split == "dropped":
                    continue
                row_start = int(boundaries[group_id])
                row_end = int(boundaries[group_id + 1])
                row_indices = order[row_start:row_end]
                matrix = _csr_rows_from_h5(x, row_indices, n_vars=n_vars)
                if split == "control":
                    control_writer.put(key, matrix)
                else:
                    matrix = _pad_for_shard_size(matrix, args.shard_size, rng)
                    for start in range(0, matrix.shape[0], args.shard_size):
                        writers[split].put(
                            {
                                "cartesian_key": key,
                                "cell_matrix": matrix[start : start + args.shard_size],
                            }
                        )
                if group_id and group_id % args.group_progress_every == 0:
                    print(
                        f"[build] wrote groups {group_id:,}/{len(group_keys):,}; "
                        f"train={writers['train'].count:,}, val={writers['val'].count:,}, "
                        f"test={writers['test'].count:,}, control={len(control_writer.keys):,}",
                        flush=True,
                    )
        finally:
            for writer in writers.values():
                writer.close()
            control_writer.close()

    summary = {
        "raw_h5ad": str(raw_h5ad),
        "split_toml": str(split_toml),
        "out_root": str(out_root),
        "split_name": args.split_name,
        "control_pert": args.control_pert,
        "min_pert_cells": args.min_pert_cells,
        "shard_size": args.shard_size,
        "padding_seed": args.padding_seed,
        "n_rows": int(n_rows),
        "n_vars": int(n_vars),
        "n_groups": len(group_keys),
        "split_group_counts": split_group_counts,
        "lmdb_counts": {
            "train": writers["train"].count,
            "val": writers["val"].count,
            "test": writers["test"].count,
            "control": len(control_writer.keys),
        },
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = out_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5ad", default=DEFAULT_RAW_H5AD)
    parser.add_argument("--split-toml", default=DEFAULT_SPLIT_TOML)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--split-name", default="split_5_state")
    parser.add_argument("--control-pert", default="PBS")
    parser.add_argument("--min-pert-cells", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--padding-seed", type=int, default=0)
    parser.add_argument("--map-size-gb", type=int, default=180)
    parser.add_argument("--write-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    parser.add_argument("--group-progress-every", type=int, default=500)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
