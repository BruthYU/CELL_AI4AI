#!/usr/bin/env python3
"""Apply Prior Delta Centering (PDC) to full-cell h5ad predictions.

For every non-control row in ``--pred-h5ad`` this script writes:

    out[c,g,i] = pred[c,g,i] + lambda * (prior_center[c,g] - mu_model[c,g])

where ``mu_model[c,g]`` is the raw model mean for the context/perturbation
group and ``prior_center[c,g]`` is either built from train-only deltas or read
from a supplied table.  A Replogle correction pkl can also be supplied as a
precomputed shift table to exactly replay existing corrected full-cell files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp


DEFAULT_STRATEGIES = "same_pert_other_context,global,zero"
VECTOR_TABLE_MODES = ("correction_shift", "prior_delta", "prior_center")


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


def obs_or_default(
    handle: h5py.File,
    *,
    context_col: str | None,
    default_context: str | None,
    n_obs: int,
) -> np.ndarray:
    if context_col:
        return obs_column(handle, context_col).astype(str)
    if default_context is None:
        raise ValueError("Provide --context-col or --default-context")
    return np.full(n_obs, str(default_context), dtype=object)


def read_rows(x: h5py.Group | h5py.Dataset, start: int, end: int, n_vars: int) -> np.ndarray:
    if isinstance(x, h5py.Group):
        if not {"data", "indices", "indptr"} <= set(x.keys()):
            raise ValueError("Only CSR h5ad X groups are supported")
        indptr = x["indptr"][start : end + 1].astype(np.int64, copy=False)
        data_start = int(indptr[0])
        data_end = int(indptr[-1])
        local_indptr = indptr - data_start
        matrix = sp.csr_matrix(
            (
                x["data"][data_start:data_end],
                x["indices"][data_start:data_end],
                local_indptr,
            ),
            shape=(end - start, n_vars),
        )
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(x[start:end], dtype=np.float32)


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


def create_csr_output(dst: h5py.File, shape: tuple[int, int]) -> tuple[h5py.Dataset, h5py.Dataset, np.ndarray]:
    x = dst.create_group("X")
    x.attrs["encoding-type"] = "csr_matrix"
    x.attrs["encoding-version"] = "0.1.0"
    x.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    data_ds = x.create_dataset(
        "data",
        shape=(0,),
        maxshape=(None,),
        chunks=(1_000_000,),
        dtype=np.float32,
    )
    indices_ds = x.create_dataset(
        "indices",
        shape=(0,),
        maxshape=(None,),
        chunks=(1_000_000,),
        dtype=np.int32,
    )
    indptr = np.empty(shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    return data_ds, indices_ds, indptr


def create_dense_output(dst: h5py.File, shape: tuple[int, int], chunk_rows: int) -> h5py.Dataset:
    rows_per_chunk = max(1, min(int(chunk_rows), 512))
    x = dst.create_dataset(
        "X",
        shape=shape,
        chunks=(rows_per_chunk, shape[1]),
        dtype=np.float32,
    )
    x.attrs["encoding-type"] = "array"
    x.attrs["encoding-version"] = "0.2.0"
    return x


def grouped_means(
    path: Path,
    *,
    context_col: str | None,
    default_context: str | None,
    pert_col: str,
    chunk_rows: int,
    clip_min: float | None,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], int]:
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    with h5py.File(path, "r") as handle:
        n_obs, n_vars = h5ad_shape(handle)
        contexts = obs_or_default(
            handle,
            context_col=context_col,
            default_context=default_context,
            n_obs=n_obs,
        ).astype(str)
        perts = obs_column(handle, pert_col).astype(str)
        if len(contexts) != n_obs or len(perts) != n_obs:
            raise ValueError(f"{path}: obs columns do not match X rows")
        x = handle["X"]
        for start in range(0, n_obs, chunk_rows):
            end = min(start + chunk_rows, n_obs)
            block = read_rows(x, start, end, n_vars).astype(np.float64, copy=False)
            if clip_min is not None:
                np.maximum(block, float(clip_min), out=block)
            context_chunk = contexts[start:end]
            pert_chunk = perts[start:end]
            for context, pert in sorted(set(zip(context_chunk, pert_chunk))):
                mask = (context_chunk == context) & (pert_chunk == pert)
                key = (str(context), str(pert))
                value = block[mask].sum(axis=0)
                if key in sums:
                    sums[key] += value
                    counts[key] += int(np.count_nonzero(mask))
                else:
                    sums[key] = value.astype(np.float64, copy=True)
                    counts[key] = int(np.count_nonzero(mask))
    means = {key: (value / float(counts[key])).astype(np.float32) for key, value in sums.items()}
    return means, counts, n_vars


def split_control_means(
    means: dict[tuple[str, str], np.ndarray],
    counts: dict[tuple[str, str], int],
    *,
    control_label: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    control_means = {context: mean for (context, pert), mean in means.items() if pert == control_label}
    control_counts = {
        context: counts[(context, pert)]
        for (context, pert) in counts
        if pert == control_label
    }
    return control_means, control_counts


def build_delta_table(
    *,
    source_h5ad: Path,
    source_control_h5ad: Path | None,
    context_col: str | None,
    default_context: str | None,
    pert_col: str,
    control_label: str,
    chunk_rows: int,
    clip_min: float | None,
) -> dict[str, Any]:
    source_means, source_counts, n_vars = grouped_means(
        source_h5ad,
        context_col=context_col,
        default_context=default_context,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )
    if source_control_h5ad is None:
        control_means, control_counts = split_control_means(
            source_means,
            source_counts,
            control_label=control_label,
        )
    else:
        control_group_means, control_group_counts, control_n_vars = grouped_means(
            source_control_h5ad,
            context_col=context_col,
            default_context=default_context,
            pert_col=pert_col,
            chunk_rows=chunk_rows,
            clip_min=clip_min,
        )
        if control_n_vars != n_vars:
            raise ValueError(f"{source_control_h5ad} vars={control_n_vars} but source vars={n_vars}")
        control_means, control_counts = split_control_means(
            control_group_means,
            control_group_counts,
            control_label=control_label,
        )

    combo_delta: dict[tuple[str, str], np.ndarray] = {}
    combo_count: dict[tuple[str, str], int] = {}
    global_sum = np.zeros(n_vars, dtype=np.float64)
    global_weight = 0.0
    missing_control_contexts: set[str] = set()

    for (context, pert), mean in source_means.items():
        if pert == control_label:
            continue
        control_mean = control_means.get(context)
        if control_mean is None:
            missing_control_contexts.add(context)
            continue
        count = int(source_counts[(context, pert)])
        delta = (mean - control_mean).astype(np.float32)
        combo_delta[(context, pert)] = delta
        combo_count[(context, pert)] = count
        global_sum += delta.astype(np.float64) * float(count)
        global_weight += float(count)

    global_delta = None
    if global_weight > 0.0:
        global_delta = (global_sum / global_weight).astype(np.float32)
    return {
        "n_vars": n_vars,
        "combo_delta": combo_delta,
        "combo_count": combo_count,
        "control_means": control_means,
        "control_counts": control_counts,
        "global_delta": global_delta,
        "global_weight": global_weight,
        "missing_control_contexts": sorted(missing_control_contexts),
    }


def average_deltas(
    candidates: list[tuple[str, str]],
    *,
    combo_delta: dict[tuple[str, str], np.ndarray],
    combo_count: dict[tuple[str, str], int],
    average_mode: str,
) -> tuple[np.ndarray | None, float]:
    if not candidates:
        return None, 0.0
    deltas = [combo_delta[key] for key in candidates]
    if average_mode == "cell_weighted":
        weights = np.asarray([combo_count[key] for key in candidates], dtype=np.float64)
        return np.average(np.vstack(deltas), axis=0, weights=weights).astype(np.float32), float(weights.sum())
    return np.mean(np.vstack(deltas), axis=0).astype(np.float32), float(sum(combo_count[key] for key in candidates))


def resolve_prior_delta(
    *,
    context: str,
    pert: str,
    strategies: list[str],
    combo_delta: dict[tuple[str, str], np.ndarray],
    combo_count: dict[tuple[str, str], int],
    global_delta: np.ndarray | None,
    global_weight: float,
    n_vars: int,
    average_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    for strategy in strategies:
        if strategy == "same_pert_other_context":
            candidates = sorted((c, p) for c, p in combo_delta if p == pert and c != context)
            delta, support = average_deltas(
                candidates,
                combo_delta=combo_delta,
                combo_count=combo_count,
                average_mode=average_mode,
            )
            if delta is not None:
                return delta, {
                    "mode": "same_pert_other_context",
                    "support_context_count": len({c for c, _ in candidates}),
                    "support_cell_count": support,
                    "source_contexts": sorted({c for c, _ in candidates}),
                }
        elif strategy == "same_pert_all_contexts":
            candidates = sorted((c, p) for c, p in combo_delta if p == pert)
            delta, support = average_deltas(
                candidates,
                combo_delta=combo_delta,
                combo_count=combo_count,
                average_mode=average_mode,
            )
            if delta is not None:
                return delta, {
                    "mode": "same_pert_all_contexts",
                    "support_context_count": len({c for c, _ in candidates}),
                    "support_cell_count": support,
                    "source_contexts": sorted({c for c, _ in candidates}),
                }
        elif strategy == "same_context_same_pert":
            key = (context, pert)
            if key in combo_delta:
                return combo_delta[key], {
                    "mode": "same_context_same_pert",
                    "support_context_count": 1,
                    "support_cell_count": float(combo_count[key]),
                    "source_contexts": [context],
                }
        elif strategy == "same_context_other_pert":
            candidates = sorted((c, p) for c, p in combo_delta if c == context and p != pert)
            delta, support = average_deltas(
                candidates,
                combo_delta=combo_delta,
                combo_count=combo_count,
                average_mode=average_mode,
            )
            if delta is not None:
                return delta, {
                    "mode": "same_context_other_pert",
                    "support_context_count": 1,
                    "support_cell_count": support,
                    "source_contexts": [context],
                }
        elif strategy == "global":
            if global_delta is not None:
                return global_delta, {
                    "mode": "global",
                    "support_context_count": 0,
                    "support_cell_count": float(global_weight),
                    "source_contexts": [],
                }
        elif strategy == "zero":
            return np.zeros(n_vars, dtype=np.float32), {
                "mode": "zero",
                "support_context_count": 0,
                "support_cell_count": 0.0,
                "source_contexts": [],
            }
        else:
            raise ValueError(f"Unknown prior delta strategy: {strategy}")

    return np.zeros(n_vars, dtype=np.float32), {
        "mode": "zero",
        "support_context_count": 0,
        "support_cell_count": 0.0,
        "source_contexts": [],
    }


def parse_strategies(text: str) -> list[str]:
    strategies = [item.strip() for item in text.split(",") if item.strip()]
    if not strategies:
        raise ValueError("At least one prior delta strategy is required")
    allowed = {
        "same_pert_other_context",
        "same_pert_all_contexts",
        "same_context_same_pert",
        "same_context_other_pert",
        "global",
        "zero",
    }
    bad = sorted(set(strategies) - allowed)
    if bad:
        raise ValueError(f"Unknown strategies: {bad}")
    return strategies


def load_vector_table(path: Path, *, n_vars: int) -> tuple[dict[tuple[str | None, str], np.ndarray], dict[str, Any]]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        pert_names = [str(x) for x in payload["pert_names"]]
        contexts = None
        if "contexts" in payload:
            contexts = [None if str(x) in {"", "*", "None"} else str(x) for x in payload["contexts"]]
        if vectors.ndim != 2 or int(vectors.shape[1]) != n_vars:
            raise ValueError(f"{path} vectors shape={vectors.shape}, expected second dim {n_vars}")
        table = {}
        for idx, pert in enumerate(pert_names):
            context = None if contexts is None else contexts[idx]
            table[(context, pert)] = vectors[idx].astype(np.float32, copy=False)
        return table, {"format": "npz", "num_vectors": len(table)}

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and {"delta", "pert_to_idx"} <= set(payload.keys()):
        vectors = np.asarray(payload["delta"], dtype=np.float32)
        if vectors.ndim != 2 or int(vectors.shape[1]) != n_vars:
            raise ValueError(f"{path} delta shape={vectors.shape}, expected second dim {n_vars}")
        table = {
            (None, str(pert)): vectors[int(idx)].astype(np.float32, copy=False)
            for pert, idx in dict(payload["pert_to_idx"]).items()
        }
        return table, {
            "format": "replogle_correction_pkl",
            "num_vectors": len(table),
            "source": payload.get("source", {}),
        }

    if isinstance(payload, dict):
        table: dict[tuple[str | None, str], np.ndarray] = {}
        for key, value in payload.items():
            if isinstance(key, tuple) and len(key) == 2:
                context, pert = key
                table[(str(context), str(pert))] = np.asarray(value, dtype=np.float32)
            else:
                table[(None, str(key))] = np.asarray(value, dtype=np.float32)
        for key, vector in table.items():
            if vector.shape != (n_vars,):
                raise ValueError(f"{path} vector for {key} has shape={vector.shape}, expected {(n_vars,)}")
        return table, {"format": "dict_pickle", "num_vectors": len(table)}

    raise ValueError(f"Unsupported center table payload in {path}")


def resolve_vector_table(
    table: dict[tuple[str | None, str], np.ndarray],
    *,
    context: str,
    pert: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    for key in ((context, pert), (None, pert), ("*", pert)):
        if key in table:
            return table[key], {
                "mode": "vector_table",
                "source_contexts": [] if key[0] is None else [str(key[0])],
                "support_context_count": 0 if key[0] is None else 1,
                "support_cell_count": 0.0,
            }
    return None, {
        "mode": "missing",
        "source_contexts": [],
        "support_context_count": 0,
        "support_cell_count": 0.0,
    }


def target_groups_from_obs(
    pred_h5ad: Path,
    *,
    context_col: str | None,
    default_context: str | None,
    pert_col: str,
    control_label: str,
) -> tuple[list[tuple[str, str]], int, int]:
    with h5py.File(pred_h5ad, "r") as handle:
        n_obs, n_vars = h5ad_shape(handle)
        contexts = obs_or_default(
            handle,
            context_col=context_col,
            default_context=default_context,
            n_obs=n_obs,
        ).astype(str)
        perts = obs_column(handle, pert_col).astype(str)
    groups = sorted(
        {
            (str(context), str(pert))
            for context, pert in zip(contexts, perts)
            if str(pert) != control_label
        }
    )
    return groups, n_obs, n_vars


def build_shift_table(
    *,
    args: argparse.Namespace,
    model_means: dict[tuple[str, str], np.ndarray],
    model_control_means: dict[str, np.ndarray],
    n_vars: int,
    target_groups: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    shifts: dict[tuple[str, str], np.ndarray] = {}
    support: dict[tuple[str, str], dict[str, Any]] = {}
    source_summary: dict[str, Any] = {}
    center_source = normalize_center_source(args.center_source)

    if center_source in {"train_delta", "memory_h5ad"}:
        source_h5ad = args.train_h5ad if center_source == "train_delta" else args.memory_h5ad
        source_control_h5ad = args.train_control_h5ad if center_source == "train_delta" else args.memory_control_h5ad
        if source_h5ad is None:
            raise ValueError(f"--{center_source.replace('_', '-')} requires a source h5ad")
        delta_table = build_delta_table(
            source_h5ad=source_h5ad,
            source_control_h5ad=source_control_h5ad,
            context_col=args.source_context_col if args.source_context_col else args.context_col,
            default_context=args.source_default_context if args.source_default_context else args.default_context,
            pert_col=args.source_pert_col if args.source_pert_col else args.pert_col,
            control_label=args.control_label,
            chunk_rows=int(args.chunk_rows),
            clip_min=args.center_clip_min,
        )
        if int(delta_table["n_vars"]) != n_vars:
            raise ValueError(f"source vars={delta_table['n_vars']} but model vars={n_vars}")
        strategies = parse_strategies(args.prior_delta_strategies)
        for context, pert in target_groups:
            delta, info = resolve_prior_delta(
                context=context,
                pert=pert,
                strategies=strategies,
                combo_delta=delta_table["combo_delta"],
                combo_count=delta_table["combo_count"],
                global_delta=delta_table["global_delta"],
                global_weight=float(delta_table["global_weight"]),
                n_vars=n_vars,
                average_mode=args.average_mode,
            )
            control_mean = model_control_means.get(context)
            if control_mean is None:
                raise KeyError(f"Missing model/test control mean for context={context!r}")
            model_mean = model_means[(context, pert)]
            shifts[(context, pert)] = (control_mean + delta - model_mean).astype(np.float32)
            support[(context, pert)] = info
        source_summary = {
            "source_h5ad": str(source_h5ad),
            "source_control_h5ad": None if source_control_h5ad is None else str(source_control_h5ad),
            "prior_delta_strategies": strategies,
            "average_mode": args.average_mode,
            "num_source_context_pert_pairs": len(delta_table["combo_delta"]),
            "source_global_cell_count": float(delta_table["global_weight"]),
            "source_missing_control_contexts": delta_table["missing_control_contexts"],
            "source_control_contexts": sorted(delta_table["control_means"]),
        }
        return shifts, support, source_summary

    if args.center_table is None:
        raise ValueError(f"{center_source} requires --center-table")
    vector_table, table_summary = load_vector_table(args.center_table, n_vars=n_vars)
    table_mode = args.center_table_mode
    if center_source == "correction_pkl":
        table_mode = "correction_shift"

    for context, pert in target_groups:
        vector, info = resolve_vector_table(vector_table, context=context, pert=pert)
        if vector is None:
            if args.missing_policy == "error":
                raise KeyError(f"Missing center vector for context={context!r}, pert={pert!r}")
            shifts[(context, pert)] = np.zeros(n_vars, dtype=np.float32)
            support[(context, pert)] = info | {"missing_policy": args.missing_policy}
            continue
        if table_mode == "correction_shift":
            shift = vector
        elif table_mode == "prior_delta":
            control_mean = model_control_means.get(context)
            if control_mean is None:
                raise KeyError(f"Missing model/test control mean for context={context!r}")
            shift = control_mean + vector - model_means[(context, pert)]
        elif table_mode == "prior_center":
            shift = vector - model_means[(context, pert)]
        else:
            raise ValueError(f"Unsupported center table mode: {table_mode}")
        shifts[(context, pert)] = np.asarray(shift, dtype=np.float32)
        support[(context, pert)] = info | {"center_table_mode": table_mode}

    source_summary = {
        "center_table": str(args.center_table),
        "center_table_mode": table_mode,
        "table_summary": table_summary,
    }
    return shifts, support, source_summary


def normalize_center_source(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    aliases = {
        "table": "center_table",
        "external_center": "center_table",
        "external": "center_table",
        "correction": "correction_pkl",
        "correction_table": "correction_pkl",
        "memory": "memory_h5ad",
        "train": "train_delta",
    }
    return aliases.get(normalized, normalized)


def write_pdc_h5ad(
    *,
    pred_h5ad: Path,
    out_h5ad: Path,
    context_col: str | None,
    default_context: str | None,
    pert_col: str,
    control_label: str,
    shifts: dict[tuple[str, str], np.ndarray],
    lambda_weight: float,
    chunk_rows: int,
    clip_min: float | None,
    missing_policy: str,
    output_format: str,
) -> dict[str, Any]:
    if out_h5ad.exists() or out_h5ad.is_symlink():
        out_h5ad.unlink()
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)

    mode_row_counts: dict[str, int] = {}
    target_counts: Counter[tuple[str, str]] = Counter()
    control_rows = 0
    missing_rows = 0
    applied_rows = 0
    with h5py.File(pred_h5ad, "r") as src, h5py.File(out_h5ad, "w") as dst:
        n_obs, n_vars = h5ad_shape(src)
        contexts = obs_or_default(
            src,
            context_col=context_col,
            default_context=default_context,
            n_obs=n_obs,
        ).astype(str)
        perts = obs_column(src, pert_col).astype(str)
        copy_non_x_groups(src, dst)
        if output_format == "csr":
            data_ds, indices_ds, out_indptr = create_csr_output(dst, (n_obs, n_vars))
            dense_ds = None
        elif output_format == "dense":
            data_ds = indices_ds = None
            out_indptr = None
            dense_ds = create_dense_output(dst, (n_obs, n_vars), chunk_rows)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        x = src["X"]
        nnz_so_far = 0

        for start in range(0, n_obs, chunk_rows):
            end = min(start + chunk_rows, n_obs)
            block = read_rows(x, start, end, n_vars).astype(np.float32, copy=False)
            out = block.copy()
            context_chunk = contexts[start:end]
            pert_chunk = perts[start:end]
            for context, pert in sorted(set(zip(context_chunk, pert_chunk))):
                mask = (context_chunk == context) & (pert_chunk == pert)
                n_rows = int(np.count_nonzero(mask))
                if pert == control_label:
                    control_rows += n_rows
                    continue
                key = (str(context), str(pert))
                shift = shifts.get(key)
                if shift is None:
                    missing_rows += n_rows
                    if missing_policy == "error":
                        raise KeyError(f"Missing PDC shift for context={context!r}, pert={pert!r}")
                    if missing_policy in {"keep", "zero"}:
                        continue
                    raise ValueError(f"Unsupported missing policy: {missing_policy}")
                out[mask] += float(lambda_weight) * shift.reshape(1, -1)
                applied_rows += n_rows
                target_counts[key] += n_rows
                mode_row_counts["applied"] = mode_row_counts.get("applied", 0) + n_rows

            if clip_min is not None:
                np.maximum(out, float(clip_min), out=out)

            if output_format == "dense":
                dense_ds[start:end, :] = out.astype(np.float32, copy=False)
                nnz_so_far += int(np.count_nonzero(out))
            else:
                csr = sp.csr_matrix(out)
                append_1d(data_ds, csr.data.astype(np.float32, copy=False))
                append_1d(indices_ds, csr.indices.astype(np.int32, copy=False))
                out_indptr[start + 1 : end + 1] = csr.indptr[1:] + nnz_so_far
                nnz_so_far += int(csr.nnz)

        if output_format == "csr":
            indptr_dtype = np.int32 if nnz_so_far <= np.iinfo(np.int32).max else np.int64
            dst["X"].create_dataset(
                "indptr",
                data=out_indptr.astype(indptr_dtype, copy=False),
                chunks=(min(n_obs + 1, 65536),),
            )

    target_values = np.asarray(list(target_counts.values()), dtype=np.int64)
    return {
        "num_rows_in": int(n_obs),
        "num_rows_out": int(n_obs),
        "num_vars": int(n_vars),
        "applied_rows": int(applied_rows),
        "control_rows": int(control_rows),
        "missing_rows": int(missing_rows),
        "nnz": int(nnz_so_far),
        "mode_row_counts": mode_row_counts,
        "target_cell_count_min": int(target_values.min()) if target_values.size else 0,
        "target_cell_count_median": float(np.median(target_values)) if target_values.size else 0.0,
        "target_cell_count_max": int(target_values.max()) if target_values.size else 0,
        "num_target_groups": int(len(target_counts)),
    }


def write_support_csv(
    path: Path,
    support_by_group: dict[tuple[str, str], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "context",
                "perturbation",
                "mode",
                "support_context_count",
                "support_cell_count",
                "source_contexts",
                "center_table_mode",
                "missing_policy",
            ],
        )
        writer.writeheader()
        for context, pert in sorted(support_by_group):
            info = support_by_group[(context, pert)]
            writer.writerow(
                {
                    "context": context,
                    "perturbation": pert,
                    "mode": info.get("mode", ""),
                    "support_context_count": info.get("support_context_count", ""),
                    "support_cell_count": info.get("support_cell_count", ""),
                    "source_contexts": ";".join(map(str, info.get("source_contexts", []))),
                    "center_table_mode": info.get("center_table_mode", ""),
                    "missing_policy": info.get("missing_policy", ""),
                }
            )


def symlink_real(real_h5ad: Path | None, out_real_h5ad: Path | None, out_h5ad: Path) -> str | None:
    if real_h5ad is None:
        return None
    out_real = out_real_h5ad or out_h5ad.with_name(out_h5ad.name.replace("_pred_", "_real_"))
    if out_real.exists() or out_real.is_symlink():
        out_real.unlink()
    out_real.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(real_h5ad.resolve(), out_real)
    return str(out_real)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def score_direct_delta_pcc(
    *,
    pred_h5ad: Path,
    real_h5ad: Path,
    context_col: str | None,
    default_context: str | None,
    pert_col: str,
    control_label: str,
    chunk_rows: int,
    clip_min: float | None,
) -> dict[str, Any]:
    pred_means, pred_counts, _ = grouped_means(
        pred_h5ad,
        context_col=context_col,
        default_context=default_context,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )
    real_means, real_counts, _ = grouped_means(
        real_h5ad,
        context_col=context_col,
        default_context=default_context,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )
    pred_control, _ = split_control_means(pred_means, pred_counts, control_label=control_label)
    real_control, _ = split_control_means(real_means, real_counts, control_label=control_label)
    rows = []
    for context, pert in sorted((set(pred_means) & set(real_means))):
        if pert == control_label:
            continue
        if context not in pred_control or context not in real_control:
            continue
        value = pearson(
            pred_means[(context, pert)] - pred_control[context],
            real_means[(context, pert)] - real_control[context],
        )
        rows.append(
            {
                "context": context,
                "perturbation": pert,
                "pearson_delta": value,
                "pred_count": int(pred_counts[(context, pert)]),
                "real_count": int(real_counts[(context, pert)]),
            }
        )
    values = np.asarray([row["pearson_delta"] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "num_scores": int(values.size),
        "num_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "rows": rows,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--out-h5ad", type=Path, required=True)
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--default-context", default=None)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-label", "--control-pert", dest="control_label", required=True)
    parser.add_argument("--lambda", dest="lambda_weight", type=float, required=True)
    parser.add_argument(
        "--center-source",
        required=True,
        help="train_delta, memory_h5ad, center_table/table/external_center, or correction_pkl",
    )
    parser.add_argument("--train-h5ad", type=Path, default=None)
    parser.add_argument("--train-control-h5ad", type=Path, default=None)
    parser.add_argument("--memory-h5ad", type=Path, default=None)
    parser.add_argument("--memory-control-h5ad", type=Path, default=None)
    parser.add_argument("--center-table", type=Path, default=None)
    parser.add_argument("--center-table-mode", choices=VECTOR_TABLE_MODES, default="correction_shift")
    parser.add_argument("--source-context-col", default=None)
    parser.add_argument("--source-default-context", default=None)
    parser.add_argument("--source-pert-col", default=None)
    parser.add_argument("--control-h5ad", type=Path, default=None)
    parser.add_argument("--real-h5ad", type=Path, default=None)
    parser.add_argument("--out-real-h5ad", type=Path, default=None)
    parser.add_argument("--prior-delta-strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--average-mode", choices=("equal_combo", "cell_weighted"), default="equal_combo")
    parser.add_argument("--missing-policy", choices=("error", "keep", "zero"), default="error")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--output-format", choices=("csr", "dense"), default="csr")
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--center-clip-min", type=float, default=None)
    parser.add_argument("--score-clip-min", type=float, default=0.0)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--sanity-json", type=Path, default=None)
    parser.add_argument("--support-csv", type=Path, default=None)
    parser.add_argument("--score-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    center_source = normalize_center_source(args.center_source)
    if center_source not in {"train_delta", "memory_h5ad", "center_table", "correction_pkl"}:
        raise ValueError(f"Unsupported center source: {args.center_source}")

    if not (0.0 <= float(args.lambda_weight) <= 1.0):
        raise ValueError("--lambda must be between 0 and 1 for PDC center interpolation")

    model_means, model_counts, model_n_vars = grouped_means(
        args.pred_h5ad,
        context_col=args.context_col,
        default_context=args.default_context,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=None,
    )
    control_source = args.control_h5ad or args.pred_h5ad
    control_means_all, control_counts_all, control_n_vars = grouped_means(
        control_source,
        context_col=args.context_col,
        default_context=args.default_context,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.center_clip_min,
    )
    if int(control_n_vars) != int(model_n_vars):
        raise ValueError(f"control vars={control_n_vars} but model vars={model_n_vars}")
    model_control_means, model_control_counts = split_control_means(
        control_means_all,
        control_counts_all,
        control_label=args.control_label,
    )
    target_groups, n_obs, n_vars = target_groups_from_obs(
        args.pred_h5ad,
        context_col=args.context_col,
        default_context=args.default_context,
        pert_col=args.pert_col,
        control_label=args.control_label,
    )
    if int(n_vars) != int(model_n_vars):
        raise ValueError(f"target scan vars={n_vars} but model means vars={model_n_vars}")

    shifts, support_by_group, source_summary = build_shift_table(
        args=args,
        model_means=model_means,
        model_control_means=model_control_means,
        n_vars=n_vars,
        target_groups=target_groups,
    )

    output_summary = write_pdc_h5ad(
        pred_h5ad=args.pred_h5ad,
        out_h5ad=args.out_h5ad,
        context_col=args.context_col,
        default_context=args.default_context,
        pert_col=args.pert_col,
        control_label=args.control_label,
        shifts=shifts,
        lambda_weight=float(args.lambda_weight),
        chunk_rows=int(args.chunk_rows),
        clip_min=args.clip_min,
        missing_policy=args.missing_policy,
        output_format=args.output_format,
    )
    out_real_h5ad = symlink_real(args.real_h5ad, args.out_real_h5ad, args.out_h5ad)

    mode_group_counts: dict[str, int] = {}
    for info in support_by_group.values():
        mode = str(info.get("mode", "unknown"))
        mode_group_counts[mode] = mode_group_counts.get(mode, 0) + 1

    sanity = {
        "pred_h5ad": str(args.pred_h5ad),
        "out_h5ad": str(args.out_h5ad),
        "context_col": args.context_col,
        "default_context": args.default_context,
        "pert_col": args.pert_col,
        "control_label": args.control_label,
        "num_rows_in": int(n_obs),
        "num_rows_out": int(output_summary["num_rows_out"]),
        "num_vars": int(n_vars),
        "num_contexts": len({context for context, _ in target_groups}),
        "num_perturbations": len({pert for _, pert in target_groups}),
        "num_target_groups": len(target_groups),
        "target_cell_count_min": output_summary["target_cell_count_min"],
        "target_cell_count_median": output_summary["target_cell_count_median"],
        "target_cell_count_max": output_summary["target_cell_count_max"],
        "control_rows": output_summary["control_rows"],
        "applied_rows": output_summary["applied_rows"],
        "missing_rows": output_summary["missing_rows"],
        "mode_group_counts": mode_group_counts,
    }

    support_path = args.support_csv or args.out_h5ad.with_suffix(".support.csv")
    write_support_csv(support_path, support_by_group)

    score = None
    if args.real_h5ad is not None and args.score_json is not None:
        score = score_direct_delta_pcc(
            pred_h5ad=args.out_h5ad,
            real_h5ad=args.real_h5ad,
            context_col=args.context_col,
            default_context=args.default_context,
            pert_col=args.pert_col,
            control_label=args.control_label,
            chunk_rows=int(args.chunk_rows),
            clip_min=args.score_clip_min,
        )
        write_json(args.score_json, score)

    manifest = {
        "method_name": "Prior Delta Centering",
        "method_variant": f"PDC-{float(args.lambda_weight):g}",
        "lambda": float(args.lambda_weight),
        "formula": "out[c,g,i] = pred[c,g,i] + lambda * (prior_center[c,g] - mu_model[c,g])",
        "center_source": center_source,
        "center_source_summary": source_summary,
        "pred_h5ad": str(args.pred_h5ad),
        "out_h5ad": str(args.out_h5ad),
        "real_h5ad_eval_only": None if args.real_h5ad is None else str(args.real_h5ad),
        "out_real_h5ad": out_real_h5ad,
        "context_col": args.context_col,
        "default_context": args.default_context,
        "pert_col": args.pert_col,
        "control_label": args.control_label,
        "train_or_memory_source": {
            "train_h5ad": None if args.train_h5ad is None else str(args.train_h5ad),
            "train_control_h5ad": None if args.train_control_h5ad is None else str(args.train_control_h5ad),
            "memory_h5ad": None if args.memory_h5ad is None else str(args.memory_h5ad),
            "memory_control_h5ad": None if args.memory_control_h5ad is None else str(args.memory_control_h5ad),
            "center_table": None if args.center_table is None else str(args.center_table),
        },
        "fallback_policy": args.prior_delta_strategies,
        "missing_policy": args.missing_policy,
        "clip_min": args.clip_min,
        "center_clip_min": args.center_clip_min,
        "score_clip_min": args.score_clip_min,
        "output_format": args.output_format,
        "num_rows_in": int(n_obs),
        "num_rows_out": int(output_summary["num_rows_out"]),
        "num_contexts": sanity["num_contexts"],
        "num_perturbations": sanity["num_perturbations"],
        "target_cell_count_min": sanity["target_cell_count_min"],
        "target_cell_count_median": sanity["target_cell_count_median"],
        "target_cell_count_max": sanity["target_cell_count_max"],
        "support_counts_by_branch": mode_group_counts,
        "missing_center_count": int(sum(1 for info in support_by_group.values() if info.get("mode") == "missing")),
        "support_csv": str(support_path),
        "sanity_json": None if args.sanity_json is None else str(args.sanity_json),
        "score_json": None if args.score_json is None else str(args.score_json),
        "strict_train_only_note": "real_h5ad is used only for scoring after output is written; it is not used to construct PDC centers.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_summary": output_summary,
        "score_summary": None if score is None else {k: v for k, v in score.items() if k != "rows"},
    }

    manifest_path = args.manifest_json or args.out_h5ad.with_suffix(".manifest.json")
    sanity_path = args.sanity_json or args.out_h5ad.with_suffix(".sanity.json")
    write_json(manifest_path, manifest)
    write_json(sanity_path, sanity)
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
