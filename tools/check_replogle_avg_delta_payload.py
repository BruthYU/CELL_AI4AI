#!/usr/bin/env python3
"""Validate avg_delta payloads consumed by main_inference_replogle.py."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-cell-line", default="")
    parser.add_argument("--expected-dim", type=int, default=2000)
    parser.add_argument("--require-source", action="store_true")
    args = parser.parse_args()

    with args.path.open("rb") as handle:
        payload = pickle.load(handle)

    failures: list[str] = []
    if not isinstance(payload, dict):
        failures.append(f"payload is {type(payload).__name__}, expected dict")

    delta = payload.get("delta") if isinstance(payload, dict) else None
    pert_to_idx = payload.get("pert_to_idx") if isinstance(payload, dict) else None
    pert_names = payload.get("pert_names") if isinstance(payload, dict) else None
    source = payload.get("source", {}) if isinstance(payload, dict) else {}

    shape = getattr(delta, "shape", None)
    if shape is None or len(shape) != 2:
        failures.append(f"delta has invalid shape: {shape}")
    elif int(shape[1]) != int(args.expected_dim):
        failures.append(f"delta dim mismatch: {shape[1]} vs expected {args.expected_dim}")

    if not isinstance(pert_to_idx, dict) or not pert_to_idx:
        failures.append("pert_to_idx is missing or empty")
    elif shape is not None and len(pert_to_idx) != int(shape[0]):
        failures.append(f"pert_to_idx length {len(pert_to_idx)} != delta rows {shape[0]}")

    if pert_names is None:
        failures.append("pert_names is missing")
    elif shape is not None and len(pert_names) != int(shape[0]):
        failures.append(f"pert_names length {len(pert_names)} != delta rows {shape[0]}")

    if args.require_source and not isinstance(source, dict):
        failures.append("source metadata is missing")

    target_cell_line = source.get("target_cell_line") if isinstance(source, dict) else None
    if args.expected_cell_line and target_cell_line != args.expected_cell_line:
        failures.append(
            f"source.target_cell_line={target_cell_line!r} != expected {args.expected_cell_line!r}"
        )

    uses_real = source.get("uses_real_target_expression") if isinstance(source, dict) else None
    if uses_real is not False:
        failures.append(f"source.uses_real_target_expression={uses_real!r}, expected False")

    memory_pred_path = source.get("memory_pred_path") if isinstance(source, dict) else None
    if args.require_source and memory_pred_path and not Path(memory_pred_path).exists():
        failures.append(f"source.memory_pred_path does not exist: {memory_pred_path}")

    print(
        f"avg_delta payload: path={args.path} shape={shape} "
        f"target_cell_line={target_cell_line} memory_branch={source.get('memory_branch') if isinstance(source, dict) else None}"
    )

    if failures:
        print("FAILED avg_delta payload check:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASSED avg_delta payload check")


if __name__ == "__main__":
    main()
