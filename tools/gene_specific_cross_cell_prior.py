#!/usr/bin/env python3
"""Build and score D_same_gene_other_cellline on generic perturbation data.

Canonical method name:

    D_same_gene_other_cellline

For each target perturbation p in context c, this script predicts the response
delta using train-only deltas for the same perturbation in other contexts:

    delta_train(p, c_source) = mean_train(p, c_source) - mean_control(c_source)
    D_same_gene_other_cellline(p, c) = mean_source(delta_train(p, c_source))

The direct delta Pearson score is computed per target across expression
features, then summarized per context and across contexts. Missing same-gene
other-context support defaults to a zero delta, which produces NaN Pearson for
the target and contributes zero under the nan-as-zero summary.

The script is deliberately data-format light:

* ``csv`` mode reads row-level train/eval CSV files with expression columns.
* ``h5ad`` mode reads a single AnnData/H5AD with either a split column or
  explicit train/eval pair CSV files. It uses h5py directly and does not require
  importing anndata for scoring.
* ``make-example`` writes a deterministic toy CSV dataset for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CONTROL_PERT = "non-targeting"


@dataclass
class GroupStats:
    means: dict[tuple[str, str], np.ndarray]
    counts: dict[tuple[str, str], int]
    gene_names: list[str]


@dataclass
class PriorResult:
    delta: np.ndarray
    mode: str
    support_context_count: int
    support_cell_count: int
    source_contexts: list[str]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(bytes).decode("utf-8")
    return str(value)


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "H5AD mode requires h5py. Install h5py or use the CSV mode with "
            "pre-extracted row-level expression data."
        ) from exc
    return h5py


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Pearson shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _nan_as_zero_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).mean())


def _summarize_target_pearsons(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    nan_count = int(arr.size - finite_count)
    return {
        "num_targets": int(arr.size),
        "finite_pearson_targets": finite_count,
        "nan_pearson_targets": nan_count,
        "finite_pearson_fraction": float(finite_count / arr.size) if arr.size else 0.0,
        "pearson_delta_finite_only": _finite_mean(values),
        "pearson_delta_nan_as_zero": _nan_as_zero_mean(values),
        "_finite_pearson_sum": float(arr[finite].sum()) if finite_count else 0.0,
        "_nan_as_zero_pearson_sum": float(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).sum())
        if arr.size
        else 0.0,
    }


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _read_gene_cols(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    cols = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return cols or None


def _infer_gene_cols(df: pd.DataFrame, metadata_cols: set[str]) -> list[str]:
    gene_cols = [
        col
        for col in df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not gene_cols:
        raise ValueError(
            "Could not infer expression columns. Pass --gene-cols or --gene-cols-file."
        )
    return [str(col) for col in gene_cols]


def _parse_gene_cols(text: str | None, file_path: str | None) -> list[str] | None:
    from_text = _split_csv_arg(text)
    from_file = _read_gene_cols(Path(file_path)) if file_path else None
    if from_text and from_file:
        raise ValueError("Use only one of --gene-cols or --gene-cols-file")
    return from_text or from_file


def _load_csv_stats(
    path: Path,
    *,
    context_col: str,
    pert_col: str,
    gene_cols: list[str] | None,
    metadata_cols: set[str],
    split_col: str | None = None,
    split_values: set[str] | None = None,
    pairs: set[tuple[str, str]] | None = None,
    include_controls: bool = True,
    control_pert: str = DEFAULT_CONTROL_PERT,
) -> GroupStats:
    df = pd.read_csv(path)
    if context_col not in df.columns or pert_col not in df.columns:
        raise KeyError(f"{path}: missing {context_col!r} or {pert_col!r}")
    if gene_cols is None:
        gene_cols = _infer_gene_cols(df, metadata_cols | {context_col, pert_col})
    missing = [col for col in gene_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing expression columns: {missing[:10]}")

    if split_col and split_values is not None:
        if split_col not in df.columns:
            raise KeyError(f"{path}: missing split column {split_col!r}")
        split_mask = df[split_col].astype(str).isin(split_values)
    else:
        split_mask = pd.Series(True, index=df.index)

    if pairs is not None:
        pair_values = list(zip(df[context_col].astype(str), df[pert_col].astype(str), strict=True))
        pair_mask = pd.Series([pair in pairs for pair in pair_values], index=df.index)
    else:
        pair_mask = pd.Series(True, index=df.index)

    pert_is_control = df[pert_col].astype(str) == str(control_pert)
    control_mask = pd.Series(bool(include_controls), index=df.index)
    if split_col and split_values is not None:
        control_mask &= split_mask
    keep = (split_mask & pair_mask & ~pert_is_control) | (control_mask & pert_is_control)
    work = df.loc[keep, [context_col, pert_col, *gene_cols]].copy()
    if work.empty:
        raise ValueError(f"{path}: no rows left after filters")
    work[context_col] = work[context_col].astype(str)
    work[pert_col] = work[pert_col].astype(str)

    means: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    grouped = work.groupby([context_col, pert_col], sort=True, observed=True)
    for key, group in grouped:
        context, pert = str(key[0]), str(key[1])
        values = group[gene_cols].to_numpy(dtype=np.float64, copy=False)
        means[(context, pert)] = values.mean(axis=0).astype(np.float32)
        counts[(context, pert)] = int(values.shape[0])
    return GroupStats(means=means, counts=counts, gene_names=gene_cols)


def _obs_values(file: Any, column: str, h5py_module: Any) -> list[str]:
    obj = file["obs"][column]
    if isinstance(obj, h5py_module.Group):
        codes = obj["codes"][:].astype(np.int64, copy=False)
        categories = [_decode(value) for value in obj["categories"][:]]
        return [categories[int(code)] for code in codes]
    return [_decode(value) for value in obj[:]]


def _x_shape(file: Any, h5py_module: Any) -> tuple[int, int]:
    x = file["X"]
    if isinstance(x, h5py_module.Group):
        return tuple(int(v) for v in x.attrs["shape"])  # type: ignore[return-value]
    return int(x.shape[0]), int(x.shape[1])


def _h5ad_var_names(file: Any, h5py_module: Any) -> list[str]:
    var = file["var"]
    for key in ("_index", "gene_ids", "gene_symbols"):
        if key in var and not isinstance(var[key], h5py_module.Group):
            return [_decode(value) for value in var[key][:]]
    return [f"gene_{idx}" for idx in range(_x_shape(file, h5py_module)[1])]


def _load_pairs_csv(path: str | None, context_col: str, pert_col: str) -> set[tuple[str, str]] | None:
    if not path:
        return None
    df = pd.read_csv(path)
    if context_col not in df.columns or pert_col not in df.columns:
        raise KeyError(f"{path}: missing {context_col!r} or {pert_col!r}")
    return set(zip(df[context_col].astype(str), df[pert_col].astype(str), strict=True))


def _load_h5ad_stats(
    path: Path,
    *,
    context_col: str,
    pert_col: str,
    split_col: str | None,
    split_values: set[str] | None,
    pairs: set[tuple[str, str]] | None,
    include_controls: bool,
    control_pert: str,
    gene_names_subset: list[str] | None,
    progress_every: int,
) -> GroupStats:
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    h5py_module = _require_h5py()

    with h5py_module.File(path, "r") as f:
        n_obs, n_vars = _x_shape(f, h5py_module)
        all_gene_names = _h5ad_var_names(f, h5py_module)
        if len(all_gene_names) != n_vars:
            all_gene_names = [f"gene_{idx}" for idx in range(n_vars)]

        if gene_names_subset is None:
            gene_indices = np.arange(n_vars, dtype=np.int64)
            gene_names = all_gene_names
        else:
            lookup = {name: idx for idx, name in enumerate(all_gene_names)}
            missing = [name for name in gene_names_subset if name not in lookup]
            if missing:
                raise KeyError(f"{path}: missing genes in var: {missing[:10]}")
            gene_indices = np.asarray([lookup[name] for name in gene_names_subset], dtype=np.int64)
            gene_names = gene_names_subset

        contexts = _obs_values(f, context_col, h5py_module)
        perts = _obs_values(f, pert_col, h5py_module)
        if len(contexts) != n_obs or len(perts) != n_obs:
            raise ValueError(f"{path}: obs length mismatch")
        splits = _obs_values(f, split_col, h5py_module) if split_col else None
        if splits is not None and len(splits) != n_obs:
            raise ValueError(f"{path}: split obs length mismatch")

        include = np.zeros(n_obs, dtype=bool)
        for idx, (context, pert) in enumerate(zip(contexts, perts, strict=True)):
            if pert == control_pert:
                if not include_controls:
                    include[idx] = False
                elif split_col is not None and split_values is not None and splits is not None:
                    include[idx] = str(splits[idx]) in split_values
                else:
                    include[idx] = True
            elif pairs is not None:
                include[idx] = (context, pert) in pairs
            elif split_col is not None and split_values is not None and splits is not None:
                include[idx] = str(splits[idx]) in split_values
            else:
                include[idx] = True

        x = f["X"]
        if isinstance(x, h5py_module.Group) and x.attrs.get("encoding-type") == "csr_matrix":
            data = x["data"]
            indices = x["indices"]
            indptr = x["indptr"][:]
            subset_lookup = {int(col_idx): out_idx for out_idx, col_idx in enumerate(gene_indices)}
            for row_idx in range(n_obs):
                if not include[row_idx]:
                    continue
                key = (contexts[row_idx], perts[row_idx])
                if key not in sums:
                    sums[key] = np.zeros(len(gene_indices), dtype=np.float64)
                    counts[key] = 0
                start = int(indptr[row_idx])
                end = int(indptr[row_idx + 1])
                if end > start:
                    row_indices = indices[start:end]
                    row_data = data[start:end]
                    for col, value in zip(row_indices, row_data, strict=True):
                        out_idx = subset_lookup.get(int(col))
                        if out_idx is not None:
                            sums[key][out_idx] += float(value)
                counts[key] += 1
                if progress_every > 0 and row_idx and row_idx % progress_every == 0:
                    print(f"[h5ad] {path.name}: {row_idx:,}/{n_obs:,}", file=sys.stderr, flush=True)
        else:
            chunk_size = 8192
            for start in range(0, n_obs, chunk_size):
                end = min(start + chunk_size, n_obs)
                block = np.asarray(x[start:end], dtype=np.float64)
                block = block[:, gene_indices]
                for local_idx, row in enumerate(block):
                    row_idx = start + local_idx
                    if not include[row_idx]:
                        continue
                    key = (contexts[row_idx], perts[row_idx])
                    if key not in sums:
                        sums[key] = np.zeros(len(gene_indices), dtype=np.float64)
                        counts[key] = 0
                    sums[key] += row
                    counts[key] += 1
                if progress_every > 0 and end % progress_every == 0:
                    print(f"[h5ad] {path.name}: {end:,}/{n_obs:,}", file=sys.stderr, flush=True)

    means = {
        key: (value / max(counts[key], 1)).astype(np.float32)
        for key, value in sums.items()
    }
    if not means:
        raise ValueError(f"{path}: no groups left after filters")
    return GroupStats(means=means, counts=counts, gene_names=gene_names)


def _build_train_delta_memory(
    train: GroupStats,
    *,
    control_pert: str,
    require_control: bool,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], dict[str, np.ndarray]]:
    control_means: dict[str, np.ndarray] = {}
    for (context, pert), mean in train.means.items():
        if pert == control_pert:
            control_means[context] = mean

    deltas: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    for key, mean in train.means.items():
        context, pert = key
        if pert == control_pert:
            continue
        control = control_means.get(context)
        if control is None:
            if require_control:
                raise KeyError(f"Missing train control for context={context!r}")
            continue
        deltas[key] = (mean - control).astype(np.float32)
        counts[key] = int(train.counts[key])
    return deltas, counts, control_means


def _resolve_prior(
    *,
    target_context: str,
    pert: str,
    train_deltas: dict[tuple[str, str], np.ndarray],
    train_counts: dict[tuple[str, str], int],
    dim: int,
    average_mode: str,
    missing_action: str,
) -> PriorResult:
    source_keys = [
        key
        for key in sorted(train_deltas)
        if key[1] == pert and key[0] != target_context
    ]
    if not source_keys:
        if missing_action == "error":
            raise KeyError(
                f"Missing same-perturbation other-context support for {(target_context, pert)!r}"
            )
        return PriorResult(
            delta=np.zeros(dim, dtype=np.float32),
            mode="zero_delta_missing",
            support_context_count=0,
            support_cell_count=0,
            source_contexts=[],
        )

    if average_mode == "equal_context":
        delta = np.mean(np.vstack([train_deltas[key] for key in source_keys]), axis=0)
    elif average_mode == "cell_weighted":
        weights = np.asarray([train_counts[key] for key in source_keys], dtype=np.float64)
        matrix = np.vstack([train_deltas[key] for key in source_keys]).astype(np.float64, copy=False)
        delta = (matrix * weights[:, None]).sum(axis=0) / weights.sum()
    else:
        raise ValueError(f"Unknown average mode: {average_mode}")

    return PriorResult(
        delta=delta.astype(np.float32),
        mode="D_same_gene_other_cellline",
        support_context_count=len(source_keys),
        support_cell_count=int(sum(train_counts[key] for key in source_keys)),
        source_contexts=[key[0] for key in source_keys],
    )


def _score_prior(
    *,
    train_stats: GroupStats,
    eval_stats: GroupStats,
    control_pert: str,
    average_mode: str,
    missing_action: str,
    real_control_source: str,
    require_train_control: bool,
    write_predicted: bool,
    out_dir: Path,
    score_name: str,
) -> dict[str, Any]:
    if train_stats.gene_names != eval_stats.gene_names:
        raise ValueError("Train and eval feature names/order differ")

    train_deltas, train_counts, train_controls = _build_train_delta_memory(
        train_stats,
        control_pert=control_pert,
        require_control=require_train_control,
    )
    eval_controls = {
        context: mean
        for (context, pert), mean in eval_stats.means.items()
        if pert == control_pert
    }
    dim = len(train_stats.gene_names)
    target_rows: list[dict[str, Any]] = []
    pred_delta_rows: list[dict[str, Any]] = []
    pred_mean_rows: list[dict[str, Any]] = []

    for (context, pert), eval_mean in sorted(eval_stats.means.items()):
        if pert == control_pert:
            continue
        if real_control_source == "train":
            real_control = train_controls.get(context)
        elif real_control_source == "train_fallback":
            real_control = eval_controls.get(context)
            if real_control is None:
                real_control = train_controls.get(context)
        else:
            real_control = eval_controls.get(context)
        if real_control is None:
            raise KeyError(
                f"Missing eval control for context={context!r}; "
                "use --real-control-source train_fallback only if appropriate."
            )

        prior = _resolve_prior(
            target_context=context,
            pert=pert,
            train_deltas=train_deltas,
            train_counts=train_counts,
            dim=dim,
            average_mode=average_mode,
            missing_action=missing_action,
        )
        real_delta = (eval_mean - real_control).astype(np.float32)
        pearson = _pearson(prior.delta, real_delta)
        target_row = {
            "context": context,
            "perturbation": pert,
            "pearson_delta": pearson,
            "mode": prior.mode,
            "support_context_count": prior.support_context_count,
            "support_cell_count": prior.support_cell_count,
            "source_contexts": ",".join(prior.source_contexts),
            "eval_n_cells": int(eval_stats.counts[(context, pert)]),
        }
        target_rows.append(target_row)

        if write_predicted:
            pred_meta = {
                "context": context,
                "perturbation": pert,
                "mode": prior.mode,
                "support_context_count": prior.support_context_count,
            }
            pred_delta_rows.append(
                {
                    **pred_meta,
                    **{gene: float(value) for gene, value in zip(train_stats.gene_names, prior.delta, strict=True)},
                }
            )
            pred_mean = real_control + prior.delta
            pred_mean_rows.append(
                {
                    **pred_meta,
                    **{gene: float(value) for gene, value in zip(train_stats.gene_names, pred_mean, strict=True)},
                }
            )

    if not target_rows:
        raise ValueError("No eval perturbation targets found")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(target_rows).to_csv(out_dir / "per_target_delta_pearson.csv", index=False)
    if write_predicted:
        pd.DataFrame(pred_delta_rows).to_csv(out_dir / "predicted_deltas.csv.gz", index=False)
        pd.DataFrame(pred_mean_rows).to_csv(out_dir / "predicted_means.csv.gz", index=False)

    context_summaries: list[dict[str, Any]] = []
    for context in sorted({str(row["context"]) for row in target_rows}):
        values = [float(row["pearson_delta"]) for row in target_rows if row["context"] == context]
        row = _summarize_target_pearsons(values)
        row["context"] = context
        row["pearson_delta"] = row["pearson_delta_finite_only"]
        context_summaries.append(row)

    finite_total = int(sum(int(row["finite_pearson_targets"]) for row in context_summaries))
    target_total = int(sum(int(row["num_targets"]) for row in context_summaries))
    finite_sum = float(sum(float(row["_finite_pearson_sum"]) for row in context_summaries))
    nan_as_zero_sum = float(sum(float(row["_nan_as_zero_pearson_sum"]) for row in context_summaries))
    mean_finite = _finite_mean([float(row["pearson_delta_finite_only"]) for row in context_summaries])
    mean_nan_as_zero = _finite_mean([float(row["pearson_delta_nan_as_zero"]) for row in context_summaries])

    payload = {
        "name": score_name,
        "canonical_branch": "D_same_gene_other_cellline",
        "formula": "pred_delta(p,c*) = mean_{c_source != c*}(mean_train(p,c_source)-mean_control(c_source))",
        "average_mode": average_mode,
        "missing_action": missing_action,
        "real_control_source": real_control_source,
        "control_pert": control_pert,
        "num_features": dim,
        "num_train_delta_combos": len(train_deltas),
        "num_contexts": len(context_summaries),
        "num_targets": target_total,
        "finite_pearson_targets": finite_total,
        "nan_pearson_targets": int(target_total - finite_total),
        "finite_pearson_fraction": float(finite_total / target_total) if target_total else 0.0,
        "mean_pearson_delta": mean_finite,
        "mean_pearson_delta_finite_only": mean_finite,
        "mean_pearson_delta_nan_as_zero": mean_nan_as_zero,
        "target_weighted_pearson_delta_finite_only": float(finite_sum / finite_total)
        if finite_total
        else float("nan"),
        "target_weighted_pearson_delta_nan_as_zero": float(nan_as_zero_sum / target_total)
        if target_total
        else float("nan"),
        # Backward-compatible alias used by older Replogle reports.
        "gene_weighted_pearson_delta_finite_only": float(finite_sum / finite_total)
        if finite_total
        else float("nan"),
        "gene_weighted_pearson_delta_nan_as_zero": float(nan_as_zero_sum / target_total)
        if target_total
        else float("nan"),
        "contexts": [_strip_internal(row) for row in context_summaries],
        "outputs": {
            "per_target_delta_pearson_csv": str(out_dir / "per_target_delta_pearson.csv"),
            "predicted_deltas_csv_gz": str(out_dir / "predicted_deltas.csv.gz") if write_predicted else None,
            "predicted_means_csv_gz": str(out_dir / "predicted_means.csv.gz") if write_predicted else None,
        },
    }
    (out_dir / "direct_delta_pearson.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return payload


def _common_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control-pert", default=DEFAULT_CONTROL_PERT)
    parser.add_argument("--average-mode", choices=("equal_context", "cell_weighted"), default="equal_context")
    parser.add_argument("--missing-action", choices=("zero", "error"), default="zero")
    parser.add_argument(
        "--real-control-source",
        choices=("eval", "train", "train_fallback"),
        default="eval",
        help="Control used for real_delta and predicted mean anchor.",
    )
    parser.add_argument("--allow-missing-train-control", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-name", default="D_same_gene_other_cellline")
    parser.add_argument("--write-predicted", action="store_true")
    parser.add_argument("--expected-mean-nan-as-zero", type=float, default=None)
    parser.add_argument("--expected-tol", type=float, default=1e-6)


def _check_expected(payload: dict[str, Any], expected: float | None, tol: float) -> None:
    if expected is None:
        return
    actual = float(payload["mean_pearson_delta_nan_as_zero"])
    if not math.isfinite(actual) or abs(actual - float(expected)) > float(tol):
        raise SystemExit(
            f"mean_pearson_delta_nan_as_zero={actual:.12g} does not match "
            f"expected={expected:.12g} within tol={tol:.3g}"
        )


def run_csv(args: argparse.Namespace) -> None:
    if args.split_col and (not args.train_splits or not args.eval_splits):
        raise ValueError("CSV split mode requires both --train-splits and --eval-splits")

    gene_cols = _parse_gene_cols(args.gene_cols, args.gene_cols_file)
    metadata_cols = {args.context_col, args.pert_col}
    if args.split_col:
        metadata_cols.add(args.split_col)

    train_stats = _load_csv_stats(
        Path(args.train_csv),
        context_col=args.context_col,
        pert_col=args.pert_col,
        gene_cols=gene_cols,
        metadata_cols=metadata_cols,
        split_col=args.split_col,
        split_values=set(_split_csv_arg(args.train_splits) or []),
        include_controls=True,
        control_pert=args.control_pert,
    )
    eval_stats = _load_csv_stats(
        Path(args.eval_csv),
        context_col=args.context_col,
        pert_col=args.pert_col,
        gene_cols=train_stats.gene_names,
        metadata_cols=metadata_cols,
        split_col=args.split_col,
        split_values=set(_split_csv_arg(args.eval_splits) or []),
        include_controls=True,
        control_pert=args.control_pert,
    )
    payload = _score_prior(
        train_stats=train_stats,
        eval_stats=eval_stats,
        control_pert=args.control_pert,
        average_mode=args.average_mode,
        missing_action=args.missing_action,
        real_control_source=args.real_control_source,
        require_train_control=not args.allow_missing_train_control,
        write_predicted=args.write_predicted,
        out_dir=Path(args.out_dir),
        score_name=args.score_name,
    )
    _check_expected(payload, args.expected_mean_nan_as_zero, args.expected_tol)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def run_h5ad(args: argparse.Namespace) -> None:
    if bool(args.split_col) == bool(args.train_pairs_csv or args.eval_pairs_csv):
        raise ValueError("H5AD mode needs either --split-col or pair CSV files, not both/neither")
    if bool(args.train_pairs_csv) != bool(args.eval_pairs_csv):
        raise ValueError("Use both --train-pairs-csv and --eval-pairs-csv")
    if args.split_col and (not args.train_splits or not args.eval_splits):
        raise ValueError("H5AD split mode requires both --train-splits and --eval-splits")

    gene_names = _read_gene_cols(Path(args.gene_cols_file)) if args.gene_cols_file else _split_csv_arg(args.gene_cols)
    train_pairs = _load_pairs_csv(args.train_pairs_csv, args.context_col, args.pert_col)
    eval_pairs = _load_pairs_csv(args.eval_pairs_csv, args.context_col, args.pert_col)
    train_stats = _load_h5ad_stats(
        Path(args.input_h5ad),
        context_col=args.context_col,
        pert_col=args.pert_col,
        split_col=args.split_col,
        split_values=set(_split_csv_arg(args.train_splits) or []),
        pairs=train_pairs,
        include_controls=True,
        control_pert=args.control_pert,
        gene_names_subset=gene_names,
        progress_every=int(args.progress_every),
    )
    eval_stats = _load_h5ad_stats(
        Path(args.input_h5ad),
        context_col=args.context_col,
        pert_col=args.pert_col,
        split_col=args.split_col,
        split_values=set(_split_csv_arg(args.eval_splits) or []),
        pairs=eval_pairs,
        include_controls=True,
        control_pert=args.control_pert,
        gene_names_subset=train_stats.gene_names,
        progress_every=int(args.progress_every),
    )
    payload = _score_prior(
        train_stats=train_stats,
        eval_stats=eval_stats,
        control_pert=args.control_pert,
        average_mode=args.average_mode,
        missing_action=args.missing_action,
        real_control_source=args.real_control_source,
        require_train_control=not args.allow_missing_train_control,
        write_predicted=args.write_predicted,
        out_dir=Path(args.out_dir),
        score_name=args.score_name,
    )
    _check_expected(payload, args.expected_mean_nan_as_zero, args.expected_tol)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def make_example(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    gene_cols = [f"g{i}" for i in range(8)]
    contexts = ["A", "B", "C"]
    perts = ["P1", "P2", "P3"]
    base = {
        context: rng.normal(loc=idx * 0.25, scale=0.2, size=len(gene_cols))
        for idx, context in enumerate(contexts)
    }
    pert_delta = {
        pert: rng.normal(loc=0.0, scale=1.0, size=len(gene_cols))
        for pert in perts
    }
    rows_train: list[dict[str, Any]] = []
    rows_eval: list[dict[str, Any]] = []

    def add_rows(rows: list[dict[str, Any]], context: str, pert: str, mean: np.ndarray, n: int) -> None:
        for _ in range(n):
            values = mean + rng.normal(scale=0.03, size=len(gene_cols))
            rows.append(
                {
                    "celltype": context,
                    "gene": pert,
                    **{gene: float(value) for gene, value in zip(gene_cols, values, strict=True)},
                }
            )

    for context in contexts:
        add_rows(rows_train, context, DEFAULT_CONTROL_PERT, base[context], 16)
        add_rows(rows_eval, context, DEFAULT_CONTROL_PERT, base[context], 12)

    # Hold out one context per perturbation. The remaining contexts provide
    # same-perturbation other-context support.
    held_out = {"P1": "A", "P2": "B", "P3": "C"}
    for pert in perts:
        for context in contexts:
            mean = base[context] + pert_delta[pert]
            if held_out[pert] == context:
                add_rows(rows_eval, context, pert, mean, 10)
            else:
                add_rows(rows_train, context, pert, mean, 10)

    train_path = out_dir / "train.csv"
    eval_path = out_dir / "eval.csv"
    pd.DataFrame(rows_train).to_csv(train_path, index=False)
    pd.DataFrame(rows_eval).to_csv(eval_path, index=False)
    manifest = {
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
        "context_col": "celltype",
        "pert_col": "gene",
        "control_pert": DEFAULT_CONTROL_PERT,
        "gene_cols": gene_cols,
        "expected_behavior": "mean_pearson_delta_nan_as_zero should be close to 1.0",
        "command": (
            f"python tools/gene_specific_cross_cell_prior.py csv --train-csv {train_path} "
            f"--eval-csv {eval_path} --out-dir {out_dir / 'score'} --gene-cols {','.join(gene_cols)}"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    csv_parser = subparsers.add_parser("csv", help="Score row-level train/eval CSV files")
    csv_parser.add_argument("--train-csv", required=True)
    csv_parser.add_argument("--eval-csv", required=True)
    csv_parser.add_argument("--gene-cols", default=None)
    csv_parser.add_argument("--gene-cols-file", default=None)
    csv_parser.add_argument("--split-col", default=None)
    csv_parser.add_argument("--train-splits", default=None)
    csv_parser.add_argument("--eval-splits", default=None)
    _common_score_args(csv_parser)
    csv_parser.set_defaults(func=run_csv)

    h5ad_parser = subparsers.add_parser("h5ad", help="Score a single H5AD file")
    h5ad_parser.add_argument("--input-h5ad", required=True)
    h5ad_parser.add_argument("--split-col", default=None)
    h5ad_parser.add_argument("--train-splits", default=None)
    h5ad_parser.add_argument("--eval-splits", default=None)
    h5ad_parser.add_argument("--train-pairs-csv", default=None)
    h5ad_parser.add_argument("--eval-pairs-csv", default=None)
    h5ad_parser.add_argument("--gene-cols", default=None)
    h5ad_parser.add_argument("--gene-cols-file", default=None)
    h5ad_parser.add_argument("--progress-every", type=int, default=200_000)
    _common_score_args(h5ad_parser)
    h5ad_parser.set_defaults(func=run_h5ad)

    example_parser = subparsers.add_parser("make-example", help="Write a deterministic toy CSV dataset")
    example_parser.add_argument("--out-dir", required=True)
    example_parser.add_argument("--seed", type=int, default=7)
    example_parser.set_defaults(func=make_example)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
