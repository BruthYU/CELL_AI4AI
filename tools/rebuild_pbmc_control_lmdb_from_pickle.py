#!/usr/bin/env python3
"""Rebuild PBMC control LMDBs from State_Parse_Filtered/control_dict.pkl."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import lmdb
from tqdm import tqdm


DEFAULT_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/"
    "preprocessing/arcinstitute/datasets/State_Parse_Filtered"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--map-size-gb", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify_lmdb_len(path: Path, expected_len: int) -> None:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1024)
    try:
        with env.begin() as txn:
            got = txn.get(b"__len__")
            if got is None:
                raise RuntimeError(f"{path} missing __len__")
            actual_len = int(got)
    finally:
        env.close()

    if actual_len != expected_len:
        raise RuntimeError(f"{path} __len__={actual_len}, expected {expected_len}")


def dump_dict_to_lmdb(
    data_dict: dict[tuple[str, str, str], Any],
    target_path: Path,
    *,
    map_size_bytes: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    tmp_path = target_path.with_name(f"{target_path.name}.rebuild_tmp")

    if tmp_path.exists():
        if not overwrite:
            raise FileExistsError(f"Temporary path exists: {tmp_path}")
        shutil.rmtree(tmp_path)

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target path exists; pass --overwrite to replace it: {target_path}")

    keys = list(data_dict.keys())
    tmp_path.mkdir(parents=True, exist_ok=False)
    env = lmdb.open(
        str(tmp_path),
        map_size=map_size_bytes,
        subdir=True,
        create=True,
        lock=True,
        readahead=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__len__", str(len(keys)).encode("utf-8"))
            txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))

        for start in tqdm(range(0, len(keys), batch_size), desc=f"writing {target_path.name}"):
            end = min(start + batch_size, len(keys))
            with env.begin(write=True) as txn:
                for key in keys[start:end]:
                    txn.put(
                        str(key).encode("utf-8"),
                        pickle.dumps(data_dict[key], protocol=pickle.HIGHEST_PROTOCOL),
                    )
        env.sync()
    finally:
        env.close()

    verify_lmdb_len(tmp_path, len(keys))
    if target_path.exists():
        shutil.rmtree(target_path)
    os.replace(tmp_path, target_path)
    verify_lmdb_len(target_path, len(keys))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    control_path = root / "control_dict.pkl"

    print(f"[load] {control_path}", flush=True)
    with control_path.open("rb") as handle:
        control_data = pickle.load(handle)

    map_size_bytes = int(args.map_size_gb * (1 << 30))
    summaries = []
    for split_id in args.splits:
        started = time.time()
        split_name = f"split_{split_id}"
        out_root = root / "few_shot" / split_name
        target = out_root / f"pbmc_control_{split_name}"
        dump_dict_to_lmdb(
            control_data,
            target,
            map_size_bytes=map_size_bytes,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        summary = {
            "split": split_name,
            "target": str(target),
            "control_count": len(control_data),
            "elapsed_seconds": time.time() - started,
        }
        summary_path = out_root / f"rebuild_control_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        summaries.append(summary)

    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
