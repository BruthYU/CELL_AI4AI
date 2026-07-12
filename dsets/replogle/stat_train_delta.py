#!/usr/bin/env python3
"""Stat global perturbation delta means from Replogle train LMDBs.

The output pkl is intended for inference-time post-processing:

    pred = pred + payload["delta"][payload["pert_to_idx"][pert_gene]]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "jit_llm_replogle_v3_statealign_resid_vpred_set512_dit36_2gpu_acc4.yaml"
DEFAULT_OUT = Path(__file__).resolve().parent / "replogle_train_avg_delta.pkl"


def matrix_mean(matrix: Any) -> np.ndarray:
    """Return a dense float32 feature mean for dense or sparse matrices."""
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def cell_matrix(obj: Any) -> Any:
    """LMDB values may be either a matrix or a dict containing cell_matrix."""
    return obj["cell_matrix"] if isinstance(obj, dict) else obj


def open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, lock=False, max_readers=1024)


def control_candidates(cartesian_key: tuple[Any, ...], control_pert: str) -> tuple[tuple[Any, ...], ...]:
    """Try the key schemas used by current and older Replogle LMDBs."""
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
    """Cache control means because many perturbations share the same control group."""
    for key in control_candidates(cartesian_key, control_pert):
        cache_key = str(key)
        if cache_key in cache:
            return cache[cache_key]
        buf = control_txn.get(str(key).encode("utf-8"))
        if buf is not None:
            mean = matrix_mean(cell_matrix(pickle.loads(buf)))
            cache[cache_key] = mean
            return mean
    raise KeyError(f"Missing control for cartesian_key={cartesian_key!r}")


def train_control_pairs(dataset_conf: dict[str, Any]) -> list[tuple[str, Path, Path]]:
    root = dataset_conf.get("replogle_root")
    cell_lines = [str(x).lower() for x in dataset_conf.get("cell_lines", [])]
    if root and cell_lines:
        root_path = Path(root)
        return [
            (
                cell,
                root_path / "few_shot" / cell / f"replogle_train_{cell}",
                root_path / "few_shot" / cell / f"replogle_control_{cell}",
            )
            for cell in cell_lines
        ]
    return [
        (
            "single",
            Path(dataset_conf["train_lmdb_path"]),
            Path(dataset_conf["control_lmdb_path"]),
        )
    ]


def stat_delta(dataset_conf: dict[str, Any], *, config_path: Path, limit: int | None) -> dict[str, Any]:
    control_pert = str(dataset_conf["control_pert"])
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    num_records = 0
    cell_dim: int | None = None
    source_cell_lines: list[str] = []

    for label, train_path, control_path in train_control_pairs(dataset_conf):
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train LMDB for {label}: {train_path}")
        if not control_path.exists():
            raise FileNotFoundError(f"Missing control LMDB for {label}: {control_path}")

        source_cell_lines.append(label)
        train_env = open_env(train_path)
        control_env = open_env(control_path)
        control_cache: dict[str, np.ndarray] = {}
        with train_env.begin() as train_txn, control_env.begin() as control_txn:
            n = int(train_txn.get(b"__len__"))
            if limit is not None:
                n = min(n, int(limit))

            for idx in tqdm(range(n), desc=f"scan {label}", unit="record", mininterval=5):
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
            "replogle_root": dataset_conf.get("replogle_root"),
            "cell_lines": source_cell_lines,
            "num_records": int(num_records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit per train LMDB for smoke tests.")
    args = parser.parse_args()

    conf = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    payload = stat_delta(conf["dataset"], config_path=args.config, limit=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"wrote {args.out} | "
        f"perts={len(payload['pert_names'])} cell_dim={payload['cell_dim']} "
        f"records={payload['source']['num_records']}"
    )


if __name__ == "__main__":
    main()
