#!/usr/bin/env python3
"""Build a target-specific PDC center table from a train-only flow delta anchor.

The small-dataset memory-anchor artifacts store train-only deltas in pickle
files. Dataset lookup uses this fallback order:

  (plate, cell, perturbation) -> (cell, perturbation) -> perturbation

This script resolves that same lookup for every target group in a prediction
h5ad and writes an npz table consumable by apply_prior_delta_centering.py with
--center-table-mode prior_delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    if "obs" not in handle or name not in handle["obs"]:
        raise KeyError(f"{handle.filename} missing obs/{name}")
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def load_anchor(path: Path) -> tuple[dict[Any, np.ndarray], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    deltas = payload.get("deltas", payload) if isinstance(payload, dict) else payload
    if not isinstance(deltas, dict):
        raise TypeError(f"Anchor deltas must be a dict, got {type(deltas)!r}: {path}")
    out = {key: np.asarray(value, dtype=np.float32).reshape(-1) for key, value in deltas.items()}
    meta = payload if isinstance(payload, dict) else {}
    return out, {
        "anchor_path": str(path),
        "payload_mode": meta.get("mode"),
        "payload_dataset": meta.get("dataset"),
        "payload_control_pert": meta.get("control_pert"),
        "payload_n_deltas": meta.get("n_deltas"),
        "num_loaded_deltas": len(out),
    }


def key_variants(key: Any) -> list[Any]:
    variants: list[Any] = []
    try:
        hash(key)
    except TypeError:
        pass
    else:
        variants.append(key)
    variants.append(str(key))
    if isinstance(key, list):
        tkey = tuple(key)
        variants.extend([tkey, str(tkey), "|".join(map(str, tkey))])
    elif isinstance(key, tuple):
        variants.append("|".join(map(str, key)))
    return variants


def lookup_anchor(
    deltas: dict[Any, np.ndarray],
    *,
    plate: str | None,
    cell: str,
    perturbation: str,
    mode: str,
) -> tuple[np.ndarray | None, Any | None, str]:
    candidates: list[tuple[str, Any]]
    if mode == "plate_cell_pert":
        if plate is None:
            raise ValueError("plate_cell_pert mode requires --batch-col")
        candidates = [
            ("plate_cell_pert", (plate, cell, perturbation)),
            ("cell_pert", (cell, perturbation)),
            ("pert_only", perturbation),
        ]
    elif mode == "cell_pert":
        candidates = [
            ("cell_pert", (cell, perturbation)),
            ("pert_only", perturbation),
        ]
    else:
        raise ValueError(f"Unsupported key mode: {mode}")

    for label, candidate in candidates:
        for variant in key_variants(candidate):
            if variant in deltas:
                return deltas[variant], variant, label
    return None, None, "missing"


def build_table(args: argparse.Namespace) -> dict[str, Any]:
    deltas, anchor_summary = load_anchor(args.anchor_pkl)
    if not deltas:
        raise ValueError(f"No deltas loaded from {args.anchor_pkl}")
    n_vars = len(next(iter(deltas.values())))

    with h5py.File(args.pred_h5ad, "r") as handle:
        n_obs, pred_n_vars = h5ad_shape(handle)
        if int(pred_n_vars) != int(n_vars):
            raise ValueError(f"anchor n_vars={n_vars}, pred h5ad n_vars={pred_n_vars}")
        contexts = obs_column(handle, args.context_col).astype(str)
        perts = obs_column(handle, args.pert_col).astype(str)
        cells = obs_column(handle, args.cell_col).astype(str) if args.cell_col else contexts
        plates = obs_column(handle, args.batch_col).astype(str) if args.batch_col else None

    if not (len(contexts) == len(perts) == len(cells) == n_obs):
        raise ValueError("obs column lengths do not match h5ad rows")
    if plates is not None and len(plates) != n_obs:
        raise ValueError("batch column length does not match h5ad rows")

    key_mode = args.key_mode
    if key_mode == "auto":
        key_mode = "plate_cell_pert" if args.batch_col else "cell_pert"

    group_first_row: dict[tuple[str, str], int] = {}
    group_counts: Counter[tuple[str, str]] = Counter()
    for idx, (context, pert) in enumerate(zip(contexts, perts)):
        context = str(context)
        pert = str(pert)
        if pert == args.control_label:
            continue
        key = (context, pert)
        group_counts[key] += 1
        group_first_row.setdefault(key, idx)

    vectors: list[np.ndarray] = []
    table_contexts: list[str] = []
    table_perts: list[str] = []
    support_rows: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    missing_groups = 0

    for context, pert in sorted(group_first_row):
        idx = group_first_row[(context, pert)]
        plate = None if plates is None else str(plates[idx])
        cell = str(cells[idx])
        vector, matched_key, mode = lookup_anchor(
            deltas,
            plate=plate,
            cell=cell,
            perturbation=pert,
            mode=key_mode,
        )
        mode_counts[mode] += 1
        if vector is None:
            missing_groups += 1
        else:
            vectors.append(vector.astype(np.float32, copy=False))
            table_contexts.append(context)
            table_perts.append(pert)
        support_rows.append(
            {
                "context": context,
                "perturbation": pert,
                "plate": "" if plate is None else plate,
                "cell": cell,
                "mode": mode,
                "matched_key": "" if matched_key is None else repr(matched_key),
                "target_cell_count": int(group_counts[(context, pert)]),
            }
        )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    if vectors:
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
    else:
        matrix = np.zeros((0, n_vars), dtype=np.float32)
    np.savez_compressed(
        args.out_npz,
        vectors=matrix,
        contexts=np.asarray(table_contexts, dtype=object),
        pert_names=np.asarray(table_perts, dtype=object),
        source_anchor=np.asarray([str(args.anchor_pkl)], dtype=object),
    )

    if args.support_csv:
        args.support_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.support_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "context",
                    "perturbation",
                    "plate",
                    "cell",
                    "mode",
                    "matched_key",
                    "target_cell_count",
                ],
            )
            writer.writeheader()
            writer.writerows(support_rows)

    target_counts = np.asarray(list(group_counts.values()), dtype=np.int64)
    summary = {
        "pred_h5ad": str(args.pred_h5ad),
        "out_npz": str(args.out_npz),
        "context_col": args.context_col,
        "cell_col": args.cell_col,
        "batch_col": args.batch_col,
        "pert_col": args.pert_col,
        "control_label": args.control_label,
        "key_mode": key_mode,
        "num_rows": int(n_obs),
        "num_vars": int(n_vars),
        "num_target_groups": len(group_counts),
        "num_resolved_vectors": len(vectors),
        "missing_group_count": int(missing_groups),
        "mode_counts": dict(mode_counts),
        "target_cell_count_min": int(target_counts.min()) if target_counts.size else 0,
        "target_cell_count_median": float(np.median(target_counts)) if target_counts.size else 0.0,
        "target_cell_count_max": int(target_counts.max()) if target_counts.size else 0,
        "anchor_summary": anchor_summary,
    }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-pkl", type=Path, required=True)
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--context-col", required=True)
    parser.add_argument("--cell-col", default=None)
    parser.add_argument("--batch-col", default=None)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--key-mode", choices=("auto", "plate_cell_pert", "cell_pert"), default="auto")
    parser.add_argument("--support-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    summary = build_table(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
