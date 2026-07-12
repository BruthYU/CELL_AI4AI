#!/usr/bin/env python3
"""Rebuild PBMC train/val/test LMDBs from cached pickle lists.

This parameterizes the State_Parse_Filtered/02_train_val_test.ipynb logic for
PBMC few-shot splits. It intentionally rebuilds only perturbation LMDBs and
leaves the existing control LMDBs untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import lmdb
from tqdm import tqdm

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/"
    "preprocessing/arcinstitute/datasets/State_Parse_Filtered"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--control-pert", default="PBS")
    parser.add_argument("--map-size-gb", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split_pairs(toml_path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    with toml_path.open("rb") as handle:
        cfg = tomllib.load(handle)

    val_pairs: set[tuple[str, str]] = set()
    test_pairs: set[tuple[str, str]] = set()
    for key, entry in cfg.get("fewshot", {}).items():
        cell_name = key.split(".")[-1].replace("_", " ")
        val_pairs.update((cell_name, pert) for pert in entry.get("val", []))
        test_pairs.update((cell_name, pert) for pert in entry.get("test", []))
    return val_pairs, test_pairs


def classify_samples(
    pert_data: list[dict[str, Any]],
    *,
    control_keys: set[tuple[str, str, str]],
    val_pairs: set[tuple[str, str]],
    test_pairs: set[tuple[str, str]],
    control_pert: str,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[tuple[str, str, str]]]:
    split_lists: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped: Counter[tuple[str, str, str]] = Counter()

    for sample in tqdm(pert_data, desc="classifying samples"):
        donor, cell_type, pert = sample["cartesian_key"]
        control_key = (donor, cell_type, control_pert)
        if control_key not in control_keys:
            skipped[(donor, cell_type, pert)] += 1
            continue

        cell_pert = (cell_type, pert)
        if cell_pert in val_pairs:
            split_lists["val"].append(sample)
            split_lists["test"].append(sample)
        elif cell_pert in test_pairs:
            split_lists["test"].append(sample)
        else:
            split_lists["train"].append(sample)

    return split_lists, skipped


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


def dump_list_to_lmdb(
    data_list: list[dict[str, Any]],
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
            txn.put(b"__len__", str(len(data_list)).encode("utf-8"))

        for start in tqdm(range(0, len(data_list), batch_size), desc=f"writing {target_path.name}"):
            end = min(start + batch_size, len(data_list))
            with env.begin(write=True) as txn:
                for idx in range(start, end):
                    txn.put(
                        str(idx).encode("utf-8"),
                        pickle.dumps(data_list[idx], protocol=pickle.HIGHEST_PROTOCOL),
                    )
        env.sync()
    finally:
        env.close()

    verify_lmdb_len(tmp_path, len(data_list))
    if target_path.exists():
        shutil.rmtree(target_path)
    os.replace(tmp_path, target_path)
    verify_lmdb_len(target_path, len(data_list))


def rebuild_split(
    split_id: int,
    *,
    root: Path,
    pert_data: list[dict[str, Any]],
    control_keys: set[tuple[str, str, str]],
    control_pert: str,
    map_size_bytes: int,
    batch_size: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    split_name = f"split_{split_id}"
    toml_path = root / "few_shot" / f"{split_name}.toml"
    out_root = root / "few_shot" / split_name
    val_pairs, test_pairs = load_split_pairs(toml_path)

    started = time.time()
    split_lists, skipped = classify_samples(
        pert_data,
        control_keys=control_keys,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        control_pert=control_pert,
    )

    if not dry_run:
        for split, data_list in split_lists.items():
            target = out_root / f"pbmc_{split}_{split_name}"
            dump_list_to_lmdb(
                data_list,
                target,
                map_size_bytes=map_size_bytes,
                batch_size=batch_size,
                overwrite=overwrite,
            )

    summary = {
        "split": split_name,
        "toml_path": str(toml_path),
        "out_root": str(out_root),
        "dry_run": dry_run,
        "counts": {split: len(data_list) for split, data_list in split_lists.items()},
        "skipped_missing_control_count": int(sum(skipped.values())),
        "skipped_missing_control_unique": len(skipped),
        "skipped_missing_control_keys": [
            {"cartesian_key": list(key), "count": count}
            for key, count in sorted(skipped.items())
        ],
        "elapsed_seconds": time.time() - started,
    }

    summary_path = out_root / (
        f"rebuild_train_val_test_summary_{'dry_run' if dry_run else time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    pert_path = root / "pert_sample_list.pkl"
    control_path = root / "control_dict.pkl"
    print(f"[load] {control_path}", flush=True)
    with control_path.open("rb") as handle:
        control_data = pickle.load(handle)
    control_keys = set(control_data.keys())
    del control_data

    print(f"[load] {pert_path}", flush=True)
    with pert_path.open("rb") as handle:
        pert_data = pickle.load(handle)

    map_size_bytes = int(args.map_size_gb * (1 << 30))
    summaries = []
    for split_id in args.splits:
        summaries.append(
            rebuild_split(
                split_id,
                root=root,
                pert_data=pert_data,
                control_keys=control_keys,
                control_pert=args.control_pert,
                map_size_bytes=map_size_bytes,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        )

    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
