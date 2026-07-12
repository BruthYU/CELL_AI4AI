#!/usr/bin/env python3
"""Check that Replogle prediction H5AD files keep cell replication.

This is a lightweight h5py-only guard for full-cell evaluation artifacts.  It
fails when any non-control perturbation has fewer than the requested number of
rows, catching compact one-row-per-target files before Cell-Eval/DE metrics are
run on them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(bytes).decode("utf-8")
    return str(value)


def _obs_values(file: h5py.File, column: str) -> list[str]:
    if "obs" not in file or column not in file["obs"]:
        raise KeyError(f"Missing obs/{column}")
    obj = file["obs"][column]
    if isinstance(obj, h5py.Group):
        categories = [_decode(value) for value in obj["categories"][:]]
        return [categories[int(code)] for code in obj["codes"][:]]
    return [_decode(value) for value in obj[:]]


def _check_file(
    path: Path,
    *,
    pert_col: str,
    control_pert: str,
    min_cells_per_target: int,
) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        perts = _obs_values(file, pert_col)
    counts = Counter(pert for pert in perts if pert != control_pert)
    if not counts:
        raise ValueError(f"{path}: no non-control perturbation rows found")
    values = np.asarray(list(counts.values()), dtype=np.int64)
    bad = [
        {"perturbation": pert, "n_cells": int(count)}
        for pert, count in sorted(counts.items())
        if count < min_cells_per_target
    ]
    return {
        "path": str(path),
        "num_targets": int(len(counts)),
        "total_rows": int(len(perts)),
        "min_cells_per_target": int(values.min()),
        "median_cells_per_target": float(np.median(values)),
        "max_cells_per_target": int(values.max()),
        "num_bad_targets": int(len(bad)),
        "bad_targets": bad[:50],
    }


def _parse_cell_lines(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--cell-lines is empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cell-lines", default="rpe1,hepg2,jurkat,k562")
    parser.add_argument("--pattern", default="replogle_pred_{cell_line}.h5ad")
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--min-cells-per-target", type=int, default=2)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    rows = []
    for cell_line in _parse_cell_lines(args.cell_lines):
        path = workspace / args.pattern.format(cell_line=cell_line, cell=cell_line)
        if not path.exists():
            raise FileNotFoundError(path)
        row = _check_file(
            path,
            pert_col=args.pert_col,
            control_pert=args.control_pert,
            min_cells_per_target=int(args.min_cells_per_target),
        )
        row["cell_line"] = cell_line
        rows.append(row)

    payload = {
        "workspace": str(workspace),
        "min_cells_per_target_required": int(args.min_cells_per_target),
        "passed": all(row["num_bad_targets"] == 0 for row in rows),
        "cell_lines": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
