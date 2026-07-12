#!/usr/bin/env python3
"""Build generic h5ad predictions with model delta plus train delta.

For every non-control row in ``--model-pred-h5ad`` this script writes:

    out = control_test + model_weight * (model_pred - control_test)
        + prior_weight * train_delta

``control_test`` is the mean control expression for the same context.  By
default it is read from control rows inside ``--model-pred-h5ad``; optionally it
can come from ``--test-control-h5ad``.  ``train_delta`` is built only from
``--train-h5ad`` and optional ``--train-control-h5ad``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp


DEFAULT_STRATEGIES = "same_pert_other_context,global,zero"


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle["obs"]:
        raise KeyError(f"{handle.filename} missing obs/{name}")
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


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


def grouped_means(
    path: Path,
    *,
    context_col: str,
    pert_col: str,
    chunk_rows: int,
    clip_min: float | None,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], int]:
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    with h5py.File(path, "r") as handle:
        contexts = obs_column(handle, context_col).astype(str)
        perts = obs_column(handle, pert_col).astype(str)
        n_obs, n_vars = h5ad_shape(handle)
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


def build_train_delta(
    *,
    train_h5ad: Path,
    train_control_h5ad: Path | None,
    context_col: str,
    pert_col: str,
    control_pert: str,
    chunk_rows: int,
    clip_min: float | None,
) -> dict[str, Any]:
    train_means, train_counts, n_vars = grouped_means(
        train_h5ad,
        context_col=context_col,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )
    if train_control_h5ad is None:
        control_means = {context: mean for (context, pert), mean in train_means.items() if pert == control_pert}
        control_counts = {
            context: train_counts[(context, pert)]
            for (context, pert) in train_counts
            if pert == control_pert
        }
    else:
        control_group_means, control_group_counts, control_n_vars = grouped_means(
            train_control_h5ad,
            context_col=context_col,
            pert_col=pert_col,
            chunk_rows=chunk_rows,
            clip_min=clip_min,
        )
        if control_n_vars != n_vars:
            raise ValueError(f"{train_control_h5ad} vars={control_n_vars} but train vars={n_vars}")
        control_means = {
            context: mean
            for (context, pert), mean in control_group_means.items()
            if pert == control_pert
        }
        control_counts = {
            context: control_group_counts[(context, pert)]
            for (context, pert) in control_group_counts
            if pert == control_pert
        }

    combo_delta: dict[tuple[str, str], np.ndarray] = {}
    combo_count: dict[tuple[str, str], int] = {}
    global_sum = np.zeros(n_vars, dtype=np.float64)
    global_weight = 0.0
    missing_control_contexts: set[str] = set()

    for (context, pert), mean in train_means.items():
        if pert == control_pert:
            continue
        control_mean = control_means.get(context)
        if control_mean is None:
            missing_control_contexts.add(context)
            continue
        count = int(train_counts[(context, pert)])
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


def resolve_train_delta(
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
            raise ValueError(f"Unknown train delta strategy: {strategy}")

    return np.zeros(n_vars, dtype=np.float32), {
        "mode": "zero",
        "support_context_count": 0,
        "support_cell_count": 0.0,
        "source_contexts": [],
    }


def load_control_means(
    *,
    model_pred_h5ad: Path,
    test_control_h5ad: Path | None,
    context_col: str,
    pert_col: str,
    control_pert: str,
    chunk_rows: int,
    clip_min: float | None,
) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    source = model_pred_h5ad if test_control_h5ad is None else test_control_h5ad
    means, counts, n_vars = grouped_means(
        source,
        context_col=context_col,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )
    control_means = {context: mean for (context, pert), mean in means.items() if pert == control_pert}
    control_counts = {
        context: counts[(context, pert)]
        for (context, pert) in counts
        if pert == control_pert
    }
    return control_means, control_counts, n_vars


def write_blended_h5ad(
    *,
    model_pred_h5ad: Path,
    out_pred_h5ad: Path,
    context_col: str,
    pert_col: str,
    control_pert: str,
    control_means: dict[str, np.ndarray],
    train_delta_by_group: dict[tuple[str, str], np.ndarray],
    support_by_group: dict[tuple[str, str], dict[str, Any]],
    model_weight: float,
    prior_weight: float,
    chunk_rows: int,
    clip_min: float | None,
) -> dict[str, Any]:
    if out_pred_h5ad.exists() or out_pred_h5ad.is_symlink():
        out_pred_h5ad.unlink()
    out_pred_h5ad.parent.mkdir(parents=True, exist_ok=True)

    mode_row_counts: dict[str, int] = {}
    target_row_count = 0
    control_row_count = 0
    with h5py.File(model_pred_h5ad, "r") as src, h5py.File(out_pred_h5ad, "w") as dst:
        contexts = obs_column(src, context_col).astype(str)
        perts = obs_column(src, pert_col).astype(str)
        n_obs, n_vars = h5ad_shape(src)
        copy_non_x_groups(src, dst)
        data_ds, indices_ds, out_indptr = create_csr_output(dst, (n_obs, n_vars))
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
                if pert == control_pert:
                    control_row_count += n_rows
                    continue
                control_mean = control_means.get(str(context))
                if control_mean is None:
                    raise KeyError(f"Missing test control mean for context={context!r}")
                delta_train = train_delta_by_group[(str(context), str(pert))]
                out[mask] = (
                    control_mean.reshape(1, -1)
                    + float(model_weight) * (block[mask] - control_mean.reshape(1, -1))
                    + float(prior_weight) * delta_train.reshape(1, -1)
                )
                mode = str(support_by_group[(str(context), str(pert))]["mode"])
                mode_row_counts[mode] = mode_row_counts.get(mode, 0) + n_rows
                target_row_count += n_rows

            if clip_min is not None:
                np.maximum(out, float(clip_min), out=out)

            csr = sp.csr_matrix(out)
            append_1d(data_ds, csr.data.astype(np.float32, copy=False))
            append_1d(indices_ds, csr.indices.astype(np.int32, copy=False))
            out_indptr[start + 1 : end + 1] = csr.indptr[1:] + nnz_so_far
            nnz_so_far += int(csr.nnz)

        indptr_dtype = np.int32 if nnz_so_far <= np.iinfo(np.int32).max else np.int64
        dst["X"].create_dataset(
            "indptr",
            data=out_indptr.astype(indptr_dtype, copy=False),
            chunks=(min(n_obs + 1, 65536),),
        )

    return {
        "rows": int(n_obs),
        "vars": int(n_vars),
        "target_rows": int(target_row_count),
        "control_rows": int(control_row_count),
        "mode_row_counts": mode_row_counts,
        "nnz": int(nnz_so_far),
    }


def parse_strategies(text: str) -> list[str]:
    strategies = [item.strip() for item in text.split(",") if item.strip()]
    if not strategies:
        raise ValueError("At least one train delta strategy is required")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pred-h5ad", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--out-pred-h5ad", type=Path, required=True)
    parser.add_argument("--model-real-h5ad", type=Path, default=None)
    parser.add_argument("--out-real-h5ad", type=Path, default=None)
    parser.add_argument("--test-control-h5ad", type=Path, default=None)
    parser.add_argument("--train-control-h5ad", type=Path, default=None)
    parser.add_argument("--context-col", required=True)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-pert", required=True)
    parser.add_argument("--model-weight", type=float, default=0.1)
    parser.add_argument("--prior-weight", type=float, default=0.9)
    parser.add_argument("--train-delta-strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--average-mode", choices=("equal_combo", "cell_weighted"), default="equal_combo")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--support-csv", type=Path, default=None)
    args = parser.parse_args()

    clip_min = None if args.no_clip else float(args.clip_min)
    strategies = parse_strategies(args.train_delta_strategies)
    train = build_train_delta(
        train_h5ad=args.train_h5ad,
        train_control_h5ad=args.train_control_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        chunk_rows=int(args.chunk_rows),
        clip_min=clip_min,
    )
    control_means, control_counts, control_n_vars = load_control_means(
        model_pred_h5ad=args.model_pred_h5ad,
        test_control_h5ad=args.test_control_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        chunk_rows=int(args.chunk_rows),
        clip_min=clip_min,
    )
    if int(train["n_vars"]) != int(control_n_vars):
        raise ValueError(f"train vars={train['n_vars']} but control/model vars={control_n_vars}")

    with h5py.File(args.model_pred_h5ad, "r") as handle:
        contexts = obs_column(handle, args.context_col).astype(str)
        perts = obs_column(handle, args.pert_col).astype(str)
        model_rows, model_vars = h5ad_shape(handle)
    if int(model_vars) != int(train["n_vars"]):
        raise ValueError(f"model vars={model_vars} but train vars={train['n_vars']}")

    target_groups = sorted(
        {
            (str(context), str(pert))
            for context, pert in zip(contexts, perts)
            if str(pert) != args.control_pert
        }
    )
    train_delta_by_group: dict[tuple[str, str], np.ndarray] = {}
    support_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    for context, pert in target_groups:
        delta, info = resolve_train_delta(
            context=context,
            pert=pert,
            strategies=strategies,
            combo_delta=train["combo_delta"],
            combo_count=train["combo_count"],
            global_delta=train["global_delta"],
            global_weight=float(train["global_weight"]),
            n_vars=int(train["n_vars"]),
            average_mode=args.average_mode,
        )
        train_delta_by_group[(context, pert)] = delta
        support_by_group[(context, pert)] = info

    output_summary = write_blended_h5ad(
        model_pred_h5ad=args.model_pred_h5ad,
        out_pred_h5ad=args.out_pred_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        control_means=control_means,
        train_delta_by_group=train_delta_by_group,
        support_by_group=support_by_group,
        model_weight=float(args.model_weight),
        prior_weight=float(args.prior_weight),
        chunk_rows=int(args.chunk_rows),
        clip_min=clip_min,
    )

    if args.model_real_h5ad is not None:
        out_real = args.out_real_h5ad or args.out_pred_h5ad.with_name(
            args.out_pred_h5ad.name.replace("_pred_", "_real_")
        )
        if out_real.exists() or out_real.is_symlink():
            out_real.unlink()
        out_real.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(args.model_real_h5ad.resolve(), out_real)
    else:
        out_real = None

    support_path = args.support_csv or args.out_pred_h5ad.with_suffix(".support.csv")
    support_path.parent.mkdir(parents=True, exist_ok=True)
    with support_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "context",
                "perturbation",
                "mode",
                "support_context_count",
                "support_cell_count",
                "source_contexts",
            ],
        )
        writer.writeheader()
        for context, pert in target_groups:
            info = support_by_group[(context, pert)]
            writer.writerow(
                {
                    "context": context,
                    "perturbation": pert,
                    "mode": info["mode"],
                    "support_context_count": info["support_context_count"],
                    "support_cell_count": info["support_cell_count"],
                    "source_contexts": ";".join(info["source_contexts"]),
                }
            )

    mode_group_counts: dict[str, int] = {}
    for info in support_by_group.values():
        mode = str(info["mode"])
        mode_group_counts[mode] = mode_group_counts.get(mode, 0) + 1

    manifest = {
        "method": "model_plus_train_delta_h5ad",
        "formula": "out = control_test + model_weight * (model_pred - control_test) + prior_weight * train_delta",
        "model_pred_h5ad": str(args.model_pred_h5ad),
        "model_real_h5ad": None if args.model_real_h5ad is None else str(args.model_real_h5ad),
        "out_pred_h5ad": str(args.out_pred_h5ad),
        "out_real_h5ad": None if out_real is None else str(out_real),
        "train_h5ad": str(args.train_h5ad),
        "train_control_h5ad": None if args.train_control_h5ad is None else str(args.train_control_h5ad),
        "test_control_h5ad": None if args.test_control_h5ad is None else str(args.test_control_h5ad),
        "context_col": args.context_col,
        "pert_col": args.pert_col,
        "control_pert": args.control_pert,
        "model_weight": float(args.model_weight),
        "prior_weight": float(args.prior_weight),
        "train_delta_strategies": strategies,
        "average_mode": args.average_mode,
        "clip_min": clip_min,
        "strict_train_only_requires": "train_h5ad/train_control_h5ad must contain only allowed train rows; this script never reads model_real_h5ad to build train_delta.",
        "exact_context_pert_warning": (
            "same_context_same_pert uses an exact context+pert train delta. Use it only when the benchmark split allows "
            "that support, such as held-out donor splits; do not use it for combo/generalization splits."
        ),
        "train_num_context_pert_pairs": len(train["combo_delta"]),
        "train_global_cell_count": float(train["global_weight"]),
        "train_missing_control_contexts": train["missing_control_contexts"],
        "test_control_contexts": sorted(control_means),
        "test_control_counts": control_counts,
        "num_target_groups": len(target_groups),
        "mode_group_counts": mode_group_counts,
        "output_summary": output_summary,
        "support_csv": str(support_path),
        "model_rows": int(model_rows),
        "vars": int(model_vars),
    }
    manifest_path = args.manifest_json or args.out_pred_h5ad.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
