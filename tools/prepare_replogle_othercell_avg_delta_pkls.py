#!/usr/bin/env python3
"""Prepare target-specific Replogle other-cell avg_delta pkl files.

These pkl files match the schema consumed by main_inference_replogle.py:
``{"delta", "pert_names", "pert_to_idx", "counts", "cell_dim", ...}``.
For each target cell line, the source train LMDBs exclude that target cell line.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "jit_llm_replogle_v3_statealign_resid_vpred_set512_dit36_2gpu_acc4.yaml"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "dsets" / "replogle" / "othercell_avg_delta"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")


def matrix_mean(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def cell_matrix(obj: Any) -> Any:
    return obj["cell_matrix"] if isinstance(obj, dict) else obj


def open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, lock=False, max_readers=1024)


def control_candidates(cartesian_key: tuple[Any, ...], control_pert: str) -> tuple[tuple[Any, ...], ...]:
    if len(cartesian_key) >= 3:
        batch, cell_line, _ = cartesian_key[:3]
        return (
            (batch, cell_line, control_pert),
            (cell_line, control_pert),
            (batch, control_pert),
        )
    if len(cartesian_key) >= 2:
        key_a, key_b = cartesian_key[:2]
        return (
            (key_a, control_pert),
            (key_b, control_pert),
            (key_a, key_b, control_pert),
            (key_b, key_a, control_pert),
        )
    raise ValueError(f"Invalid cartesian_key={cartesian_key!r}")


def control_mean_cached(
    control_txn: Any,
    cartesian_key: tuple[Any, ...],
    control_pert: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    for key in control_candidates(cartesian_key, control_pert):
        cache_key = str(key)
        if cache_key in cache:
            return cache[cache_key]
        buf = control_txn.get(cache_key.encode("utf-8"))
        if buf is not None:
            mean = matrix_mean(cell_matrix(pickle.loads(buf)))
            cache[cache_key] = mean
            return mean
    raise KeyError(f"Missing control for cartesian_key={cartesian_key!r}")


def load_dataset_config(config_path: Path) -> dict[str, Any]:
    conf = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = dict(conf["dataset"])
    root = str(dataset.get("replogle_root", ""))
    if root.startswith("${") or not root:
        raise ValueError(f"dataset.replogle_root must be concrete in {config_path}")
    return dataset


def stat_delta(
    dataset_conf: dict[str, Any],
    *,
    config_path: Path,
    source_cells: list[str],
    limit: int | None,
) -> dict[str, Any]:
    control_pert = str(dataset_conf["control_pert"])
    replogle_root = Path(str(dataset_conf["replogle_root"]))
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    num_records = 0
    cell_dim: int | None = None

    for cell_line in source_cells:
        train_path = replogle_root / "few_shot" / cell_line / f"replogle_train_{cell_line}"
        control_path = replogle_root / "few_shot" / cell_line / f"replogle_control_{cell_line}"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train LMDB for {cell_line}: {train_path}")
        if not control_path.exists():
            raise FileNotFoundError(f"Missing control LMDB for {cell_line}: {control_path}")

        print(f"scan {cell_line}: {train_path}")
        train_env = open_env(train_path)
        control_env = open_env(control_path)
        control_cache: dict[str, np.ndarray] = {}
        with train_env.begin() as train_txn, control_env.begin() as control_txn:
            n_records = int(train_txn.get(b"__len__"))
            if limit is not None:
                n_records = min(n_records, int(limit))
            for idx in range(n_records):
                buf = train_txn.get(str(idx).encode("utf-8"))
                if buf is None:
                    raise KeyError(f"Missing train key {idx} in {train_path}")
                group = pickle.loads(buf)
                cartesian_key = tuple(group["cartesian_key"])
                pert_gene = str(cartesian_key[-1])
                if pert_gene == control_pert:
                    continue

                pert_matrix = cell_matrix(group)
                ctrl_mean = control_mean_cached(control_txn, cartesian_key, control_pert, control_cache)
                weight = int(pert_matrix.shape[0])
                delta = matrix_mean(pert_matrix) - ctrl_mean
                if cell_dim is None:
                    cell_dim = int(delta.shape[0])
                elif int(delta.shape[0]) != cell_dim:
                    raise ValueError(f"cell_dim mismatch for {cartesian_key!r}: {delta.shape[0]} vs {cell_dim}")

                if pert_gene not in sums:
                    sums[pert_gene] = np.zeros(cell_dim, dtype=np.float32)
                    counts[pert_gene] = 0
                sums[pert_gene] += delta.astype(np.float32, copy=False) * float(weight)
                counts[pert_gene] += weight
                num_records += 1
        train_env.close()
        control_env.close()

    if not sums or cell_dim is None:
        raise RuntimeError("No perturbation records were collected.")

    pert_names = sorted(sums)
    delta = np.stack([sums[name] / float(counts[name]) for name in pert_names]).astype(np.float32, copy=False)
    count_array = np.asarray([counts[name] for name in pert_names], dtype=np.int64)
    return {
        "delta": delta,
        "pert_names": pert_names,
        "pert_to_idx": {name: idx for idx, name in enumerate(pert_names)},
        "counts": count_array,
        "cell_dim": int(cell_dim),
        "control_pert": control_pert,
        "source": {
            "config": str(config_path),
            "replogle_root": str(replogle_root),
            "cell_lines": source_cells,
            "num_records": int(num_records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--limit", type=int, default=None, help="Optional train record limit for smoke tests.")
    args = parser.parse_args()

    base_dataset = load_dataset_config(args.config)
    cell_lines = [str(x).lower() for x in args.cell_lines]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for target_cell in cell_lines:
        source_cells = [cell for cell in cell_lines if cell != target_cell]
        dataset_conf = dict(base_dataset)
        dataset_conf["cell_lines"] = source_cells
        payload = stat_delta(
            dataset_conf,
            config_path=args.config,
            source_cells=source_cells,
            limit=args.limit,
        )
        payload["source"] = {
            **dict(payload.get("source", {})),
            "target_cell_line": target_cell,
            "source_cell_lines": source_cells,
            "mode": "D_same_gene_other_cellline",
            "target_cell_excluded": True,
        }

        out_path = args.out_dir / f"replogle_train_avg_delta_othercell_for_{target_cell}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"wrote {out_path} | target={target_cell} sources={','.join(source_cells)} "
            f"perts={len(payload['pert_names'])} records={payload['source']['num_records']}"
        )


if __name__ == "__main__":
    main()
