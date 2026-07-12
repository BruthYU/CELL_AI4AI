#!/usr/bin/env python3
"""Build a train-only Tahoe prior-delta table for PDC.

The output ``.npz`` is consumed by ``tools/apply_prior_delta_centering.py`` with
``--center-source center_table --center-table-mode prior_delta``.

For each target group ``(c, g)`` in the raw Tahoe prediction h5ad this script
stores a vector ``D_train[c,g]`` built only from the supplied train plate h5ads.
The default branch is the Tahoe wording of the PDC same-perturbation prior:

    average over train contexts c' != c of
        mean(train_pert[c',g]) - mean(train_ctrl[c'])

``real_h5ad`` is intentionally not an input.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp


DEFAULT_CONTROL = "[('DMSO_TF', 0.0, 'uM')]"
DEFAULT_STRATEGIES = "same_pert_other_context,global,zero"


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if isinstance(x, h5py.Group) and "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def obs_codes_and_categories(handle: h5py.File, name: str) -> tuple[np.ndarray, np.ndarray]:
    if "obs" not in handle or name not in handle["obs"]:
        raise KeyError(f"{handle.filename} missing obs/{name}")
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        return obj["codes"][:].astype(np.int64, copy=False), obj["categories"].asstr()[:].astype(str)
    values = obj.asstr()[:] if hasattr(obj, "asstr") else obj[:].astype(str)
    categories, codes = np.unique(values.astype(str), return_inverse=True)
    return codes.astype(np.int64, copy=False), categories.astype(str)


def read_csr_rows(x: h5py.Group, start: int, end: int, n_vars: int) -> sp.csr_matrix:
    indptr = x["indptr"][start : end + 1].astype(np.int64, copy=False)
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    local_indptr = indptr - data_start
    return sp.csr_matrix(
        (
            x["data"][data_start:data_end],
            x["indices"][data_start:data_end],
            local_indptr,
        ),
        shape=(end - start, n_vars),
    )


def read_matrix_rows(x: h5py.Group | h5py.Dataset, start: int, end: int, n_vars: int) -> sp.csr_matrix:
    if isinstance(x, h5py.Group):
        if not {"data", "indices", "indptr"} <= set(x.keys()):
            raise ValueError(f"{x.name}: only CSR h5ad X groups are supported")
        return read_csr_rows(x, start, end, n_vars)
    return sp.csr_matrix(np.asarray(x[start:end], dtype=np.float32))


def target_groups_from_pred(
    pred_h5ad: Path,
    *,
    context_col: str,
    pert_col: str,
    control_label: str,
) -> tuple[list[tuple[str, str]], set[str], int, int]:
    with h5py.File(pred_h5ad, "r") as handle:
        n_obs, n_vars = h5ad_shape(handle)
        context_codes, context_cats = obs_codes_and_categories(handle, context_col)
        pert_codes, pert_cats = obs_codes_and_categories(handle, pert_col)
        if len(context_codes) != n_obs or len(pert_codes) != n_obs:
            raise ValueError(f"{pred_h5ad}: obs row count does not match X")
        groups = {
            (str(context_cats[int(c)]), str(pert_cats[int(p)]))
            for c, p in zip(context_codes, pert_codes)
            if str(pert_cats[int(p)]) != control_label
        }
    target_perts = {pert for _, pert in groups}
    return sorted(groups), target_perts, n_obs, n_vars


def train_files_from_dir(train_dir: Path, regex: str) -> list[Path]:
    pattern = re.compile(regex)
    files = sorted(path for path in train_dir.glob("*.h5ad") if pattern.match(path.name))
    if not files:
        raise FileNotFoundError(f"No train h5ad files in {train_dir} matching {regex!r}")
    return files


def add_sum(
    sums: dict[tuple[str, str], np.ndarray],
    counts: dict[tuple[str, str], int],
    key: tuple[str, str],
    value: np.ndarray,
    count: int,
) -> None:
    if count <= 0:
        return
    if key in sums:
        sums[key] += value
        counts[key] += int(count)
    else:
        sums[key] = value.astype(np.float64, copy=True)
        counts[key] = int(count)


def scan_train_h5ads(
    train_files: list[Path],
    *,
    target_perts: set[str],
    control_label: str,
    plate_col: str,
    cell_line_col: str,
    pert_col: str,
    context_template: str,
    chunk_rows: int,
    target_perts_only: bool,
    n_vars_expected: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], dict[str, Any]]:
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    report: dict[str, Any] = {
        "train_files": [],
        "target_perts_only": bool(target_perts_only),
        "context_template": context_template,
    }

    for train_file in train_files:
        print(f"[tahoe-pdc-center] scanning {train_file}", flush=True)
        with h5py.File(train_file, "r") as handle:
            n_obs, n_vars = h5ad_shape(handle)
            if n_vars != n_vars_expected:
                raise ValueError(f"{train_file}: n_vars={n_vars}, expected {n_vars_expected}")
            plate_codes, plate_cats = obs_codes_and_categories(handle, plate_col)
            cell_codes, cell_cats = obs_codes_and_categories(handle, cell_line_col)
            pert_codes, pert_cats = obs_codes_and_categories(handle, pert_col)
            if not (len(plate_codes) == len(cell_codes) == len(pert_codes) == n_obs):
                raise ValueError(f"{train_file}: obs row counts do not match X")
            pert_keep_codes = {
                idx
                for idx, value in enumerate(pert_cats)
                if str(value) == control_label
                or (not target_perts_only and str(value) != control_label)
                or str(value) in target_perts
            }
            if not pert_keep_codes:
                raise ValueError(f"{train_file}: no kept perturbation categories")

            x = handle["X"]
            file_counts = Counter()
            file_rows_kept = 0
            for start in range(0, n_obs, chunk_rows):
                end = min(start + chunk_rows, n_obs)
                pert_chunk = pert_codes[start:end]
                keep = np.isin(pert_chunk, np.fromiter(pert_keep_codes, dtype=np.int64))
                if not np.any(keep):
                    continue
                matrix = read_matrix_rows(x, start, end, n_vars)
                plate_chunk = plate_codes[start:end]
                cell_chunk = cell_codes[start:end]
                packed = (
                    plate_chunk.astype(np.int64) * (len(cell_cats) * len(pert_cats))
                    + cell_chunk.astype(np.int64) * len(pert_cats)
                    + pert_chunk.astype(np.int64)
                )
                for code in np.unique(packed[keep]):
                    mask = packed == int(code)
                    count = int(np.count_nonzero(mask))
                    if count <= 0:
                        continue
                    rem = int(code)
                    plate_idx = rem // (len(cell_cats) * len(pert_cats))
                    rem = rem % (len(cell_cats) * len(pert_cats))
                    cell_idx = rem // len(pert_cats)
                    pert_idx = rem % len(pert_cats)
                    pert = str(pert_cats[pert_idx])
                    if pert != control_label and target_perts_only and pert not in target_perts:
                        continue
                    context = context_template.format(
                        plate=str(plate_cats[plate_idx]),
                        cell_line=str(cell_cats[cell_idx]),
                    )
                    value = np.asarray(matrix[mask].sum(axis=0)).ravel()
                    add_sum(sums, counts, (context, pert), value, count)
                    file_counts[pert] += count
                    file_rows_kept += count

            report["train_files"].append(
                {
                    "path": str(train_file),
                    "rows": int(n_obs),
                    "kept_rows": int(file_rows_kept),
                    "num_perturbations_seen": int(len(file_counts)),
                    "has_control": bool(control_label in file_counts),
                }
            )
            print(
                "[tahoe-pdc-center] finished "
                f"{train_file.name}: rows={n_obs} kept_rows={file_rows_kept} "
                f"perturbations_seen={len(file_counts)}",
                flush=True,
            )

    return sums, counts, report


def build_combo_deltas(
    sums: dict[tuple[str, str], np.ndarray],
    counts: dict[tuple[str, str], int],
    *,
    control_label: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], dict[str, np.ndarray], dict[str, int]]:
    control_means: dict[str, np.ndarray] = {}
    control_counts: dict[str, int] = {}
    for (context, pert), value in sums.items():
        if pert != control_label:
            continue
        count = int(counts[(context, pert)])
        control_means[context] = (value / float(count)).astype(np.float32)
        control_counts[context] = count

    combo_delta: dict[tuple[str, str], np.ndarray] = {}
    combo_count: dict[tuple[str, str], int] = {}
    for (context, pert), value in sums.items():
        if pert == control_label or context not in control_means:
            continue
        count = int(counts[(context, pert)])
        pert_mean = (value / float(count)).astype(np.float32)
        combo_delta[(context, pert)] = (pert_mean - control_means[context]).astype(np.float32)
        combo_count[(context, pert)] = count
    return combo_delta, combo_count, control_means, control_counts


def average_deltas(
    candidates: list[tuple[str, str]],
    combo_delta: dict[tuple[str, str], np.ndarray],
    combo_count: dict[tuple[str, str], int],
    *,
    average_mode: str,
) -> tuple[np.ndarray | None, float, list[str]]:
    if not candidates:
        return None, 0.0, []
    contexts = [context for context, _ in candidates]
    if average_mode == "cell_weighted":
        weights = np.asarray([combo_count[key] for key in candidates], dtype=np.float64)
        vectors = np.vstack([combo_delta[key] for key in candidates]).astype(np.float64)
        return np.average(vectors, axis=0, weights=weights).astype(np.float32), float(weights.sum()), contexts
    vectors = np.vstack([combo_delta[key] for key in candidates]).astype(np.float64)
    support = float(sum(combo_count[key] for key in candidates))
    return vectors.mean(axis=0).astype(np.float32), support, contexts


def parse_strategies(value: str) -> list[str]:
    strategies = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"same_pert_other_context", "global", "zero"}
    bad = sorted(set(strategies) - allowed)
    if bad:
        raise ValueError(f"Unsupported strategies: {bad}")
    if not strategies:
        raise ValueError("At least one strategy is required")
    return strategies


def global_delta_for_target(
    combo_delta: dict[tuple[str, str], np.ndarray],
    combo_count: dict[tuple[str, str], int],
    *,
    target_context: str,
    pert: str,
) -> tuple[np.ndarray | None, float]:
    total = None
    weight = 0.0
    for key, delta in combo_delta.items():
        if key == (target_context, pert):
            continue
        count = float(combo_count[key])
        weighted = delta.astype(np.float64) * count
        total = weighted if total is None else total + weighted
        weight += count
    if total is None or weight <= 0.0:
        return None, 0.0
    return (total / weight).astype(np.float32), float(weight)


def resolve_delta(
    *,
    context: str,
    pert: str,
    combo_delta: dict[tuple[str, str], np.ndarray],
    combo_count: dict[tuple[str, str], int],
    strategies: list[str],
    average_mode: str,
    n_vars: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    for strategy in strategies:
        if strategy == "same_pert_other_context":
            candidates = sorted((c, p) for c, p in combo_delta if p == pert and c != context)
            delta, support, contexts = average_deltas(
                candidates,
                combo_delta,
                combo_count,
                average_mode=average_mode,
            )
            if delta is not None:
                return delta, {
                    "mode": "same_pert_other_context",
                    "support_context_count": len(contexts),
                    "support_cell_count": support,
                    "source_contexts": contexts,
                    "exact_target_combo_present": (context, pert) in combo_delta,
                }
        elif strategy == "global":
            delta, support = global_delta_for_target(
                combo_delta,
                combo_count,
                target_context=context,
                pert=pert,
            )
            if delta is not None:
                return delta, {
                    "mode": "global",
                    "support_context_count": 0,
                    "support_cell_count": support,
                    "source_contexts": [],
                    "exact_target_combo_present": (context, pert) in combo_delta,
                }
        elif strategy == "zero":
            return np.zeros(n_vars, dtype=np.float32), {
                "mode": "zero",
                "support_context_count": 0,
                "support_cell_count": 0.0,
                "source_contexts": [],
                "exact_target_combo_present": (context, pert) in combo_delta,
            }
    raise RuntimeError(f"No strategy resolved for context={context!r}, pert={pert!r}")


def write_support_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context",
        "perturbation",
        "mode",
        "support_context_count",
        "support_cell_count",
        "source_contexts",
        "exact_target_combo_present",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["source_contexts"] = ";".join(map(str, payload.get("source_contexts", [])))
            writer.writerow(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--support-csv", type=Path, required=True)
    parser.add_argument("--pred-context-col", default="celltype")
    parser.add_argument("--pred-pert-col", default="drugname_drugconc")
    parser.add_argument("--train-plate-col", default="plate")
    parser.add_argument("--train-cell-line-col", default="cell_line")
    parser.add_argument("--train-pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--context-template", default="{plate}_{cell_line}")
    parser.add_argument("--train-file-regex", default=r"^plate\d+_hvg\.h5ad$")
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--average-mode", choices=("equal_combo", "cell_weighted"), default="equal_combo")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--target-perts-only", action="store_true")
    parser.add_argument("--limit-train-files", type=int, default=None)
    parser.add_argument("--limit-target-groups", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = parse_strategies(args.strategies)
    target_groups, target_perts, pred_rows, n_vars = target_groups_from_pred(
        args.pred_h5ad,
        context_col=args.pred_context_col,
        pert_col=args.pred_pert_col,
        control_label=args.control_label,
    )
    if args.limit_target_groups is not None:
        target_groups = target_groups[: int(args.limit_target_groups)]
        target_perts = {pert for _, pert in target_groups}

    train_files = train_files_from_dir(args.train_dir, args.train_file_regex)
    if args.limit_train_files is not None:
        train_files = train_files[: int(args.limit_train_files)]

    sums, counts, scan_report = scan_train_h5ads(
        train_files,
        target_perts=target_perts,
        control_label=args.control_label,
        plate_col=args.train_plate_col,
        cell_line_col=args.train_cell_line_col,
        pert_col=args.train_pert_col,
        context_template=args.context_template,
        chunk_rows=int(args.chunk_rows),
        target_perts_only=bool(args.target_perts_only),
        n_vars_expected=n_vars,
    )
    combo_delta, combo_count, control_means, control_counts = build_combo_deltas(
        sums,
        counts,
        control_label=args.control_label,
    )
    if not combo_delta:
        raise RuntimeError("No train perturbation deltas were built")

    vectors = []
    contexts = []
    pert_names = []
    support_rows: list[dict[str, Any]] = []
    mode_counts = Counter()
    for context, pert in target_groups:
        delta, info = resolve_delta(
            context=context,
            pert=pert,
            combo_delta=combo_delta,
            combo_count=combo_count,
            strategies=strategies,
            average_mode=args.average_mode,
            n_vars=n_vars,
        )
        vectors.append(delta.astype(np.float32, copy=False))
        contexts.append(context)
        pert_names.append(pert)
        mode_counts[str(info["mode"])] += 1
        support_rows.append(
            {
                "context": context,
                "perturbation": pert,
                **info,
            }
        )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        vectors=np.vstack(vectors).astype(np.float32),
        contexts=np.asarray(contexts, dtype=object),
        pert_names=np.asarray(pert_names, dtype=object),
    )
    write_support_csv(args.support_csv, support_rows)
    manifest = {
        "method_name": "Prior Delta Centering",
        "table_type": "prior_delta",
        "dataset": "Tahoe",
        "pred_h5ad": str(args.pred_h5ad),
        "train_dir": str(args.train_dir),
        "train_files": [str(path) for path in train_files],
        "out_npz": str(args.out_npz),
        "support_csv": str(args.support_csv),
        "pred_context_col": args.pred_context_col,
        "pred_pert_col": args.pred_pert_col,
        "train_plate_col": args.train_plate_col,
        "train_cell_line_col": args.train_cell_line_col,
        "train_pert_col": args.train_pert_col,
        "control_label": args.control_label,
        "formula": "D_train[c,g] = average_{train c' != c}(mean(train_pert[c',g]) - mean(train_ctrl[c'])) with configured fallback",
        "strategies": strategies,
        "average_mode": args.average_mode,
        "context_template": args.context_template,
        "train_file_regex": args.train_file_regex,
        "target_perts_only": bool(args.target_perts_only),
        "pred_rows": int(pred_rows),
        "n_vars": int(n_vars),
        "num_target_groups": len(target_groups),
        "num_target_perturbations": len({pert for _, pert in target_groups}),
        "num_train_context_pert_pairs": len(combo_delta),
        "num_train_control_contexts": len(control_means),
        "mode_counts": dict(mode_counts),
        "control_count_min": int(min(control_counts.values())) if control_counts else 0,
        "control_count_median": float(np.median(list(control_counts.values()))) if control_counts else 0.0,
        "control_count_max": int(max(control_counts.values())) if control_counts else 0,
        "no_real_h5ad_used": True,
        "scan_report": scan_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
