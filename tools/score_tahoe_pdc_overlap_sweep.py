#!/usr/bin/env python3
"""Score Tahoe-100M PDC lambda sweep with on-the-fly mean-level overlap metrics.

This script does not materialize one h5ad per lambda.  It loads raw prediction
and real group means once, applies

    mu_pdc = mu_model + lambda * (pred_control + D_train - mu_model)

at group-mean level, and writes delta PCC, top-|delta| overlap, top-|mean|
overlap, and RMSE summaries for each lambda.

The overlap here is a fast center-level diagnostic.  It is not the exact
cell-eval DE overlap, which depends on DE ranking and FDR filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from apply_prior_delta_centering import grouped_means, pearson, split_control_means


BASE = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang")
WORKSPACE = BASE / "master/nemo_cellflow/benchmark/workspace"
RAW_DIR = WORKSPACE / "20260615_055828_JIT_FP"
PDC_DIR = WORKSPACE / "20260615_055828_JIT_FP_pdc_trainonly_20260624"
DEFAULT_PRED = RAW_DIR / "tahoe100m_pred.h5ad"
DEFAULT_REAL = RAW_DIR / "tahoe100m_real.h5ad"
DEFAULT_CENTER_TABLE = PDC_DIR / "tahoe_pdc_trainonly_prior_delta.npz"
DEFAULT_SUPPORT_CSV = PDC_DIR / "tahoe_pdc_trainonly_prior_delta_support.csv"
DEFAULT_OUT_DIR = PDC_DIR / "overlap_sweep_20260625"
DEFAULT_CONTROL = "[('DMSO_TF', 0.0, 'uM')]"
DEFAULT_LAMBDA_GRID = ",".join(
    ["0"]
    + [f"{value / 100:.2f}" for value in range(5, 100, 5)]
    + ["1"]
)
TOP_KS = (50, 100, 200, 500)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def parse_lambda_grid(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {value}")
        values.append(round(value, 10))
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(values))


def parse_weights(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(":", 1)
        k = int(key)
        if k not in TOP_KS:
            raise ValueError(f"selection weight k={k} is not in supported TOP_KS={TOP_KS}")
        weights[k] = float(value)
    if not weights:
        raise ValueError("At least one selection weight is required")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Selection weights must sum to a positive value")
    return {key: value / total for key, value in weights.items()}


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def overlap_at_k(real_vec: np.ndarray, pred_vec: np.ndarray, k: int) -> float:
    kk = min(int(k), int(real_vec.size), int(pred_vec.size))
    if kk <= 0:
        return float("nan")
    real_top = np.argpartition(np.abs(real_vec), -kk)[-kk:]
    pred_top = np.argpartition(np.abs(pred_vec), -kk)[-kk:]
    return len(set(real_top.tolist()).intersection(pred_top.tolist())) / float(kk)


def cache_key(path: Path, *, context_col: str, pert_col: str, clip_min: float | None) -> str:
    stat = path.stat()
    payload = json.dumps(
        {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "context_col": context_col,
            "pert_col": pert_col,
            "clip_min": clip_min,
            "cache_version": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_or_compute_grouped_means(
    path: Path,
    *,
    context_col: str,
    pert_col: str,
    chunk_rows: int,
    clip_min: float | None,
    cache_dir: Path | None,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], int, Path | None, str]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"grouped_means_{cache_key(path, context_col=context_col, pert_col=pert_col, clip_min=clip_min)}.npz"
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=True)
            contexts = [str(value) for value in payload["contexts"]]
            perts = [str(value) for value in payload["perts"]]
            counts_arr = np.asarray(payload["counts"], dtype=np.int64)
            means_arr = np.asarray(payload["means"], dtype=np.float32)
            n_vars = int(payload["n_vars"])
            means = {
                (context, pert): means_arr[idx].astype(np.float32, copy=False)
                for idx, (context, pert) in enumerate(zip(contexts, perts))
            }
            counts = {
                (context, pert): int(counts_arr[idx])
                for idx, (context, pert) in enumerate(zip(contexts, perts))
            }
            return means, counts, n_vars, cache_path, "cache_hit"

    print(f"[grouped-means] scanning {path}", flush=True)
    means, counts, n_vars = grouped_means(
        path,
        context_col=context_col,
        default_context=None,
        pert_col=pert_col,
        chunk_rows=chunk_rows,
        clip_min=clip_min,
    )

    if cache_path is not None:
        keys = sorted(means)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                contexts=np.asarray([key[0] for key in keys], dtype=object),
                perts=np.asarray([key[1] for key in keys], dtype=object),
                counts=np.asarray([counts[key] for key in keys], dtype=np.int64),
                means=np.vstack([means[key] for key in keys]).astype(np.float32),
                n_vars=np.asarray([int(n_vars)], dtype=np.int64),
            )
        os.replace(tmp_path, cache_path)
        print(f"[grouped-means] cached {cache_path}", flush=True)
        return means, counts, n_vars, cache_path, "cache_write"

    return means, counts, n_vars, None, "no_cache"


def load_center_table(path: Path, n_vars: int) -> dict[tuple[str, str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    contexts = [str(value) for value in payload["contexts"]]
    perts = [str(value) for value in payload["pert_names"]]
    if vectors.ndim != 2 or int(vectors.shape[1]) != int(n_vars):
        raise ValueError(f"{path}: vectors shape={vectors.shape}, expected second dim {n_vars}")
    return {
        (context, pert): vectors[idx].astype(np.float32, copy=False)
        for idx, (context, pert) in enumerate(zip(contexts, perts))
    }


def load_support(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["context"], row["perturbation"]): row
        for row in rows
        if "context" in row and "perturbation" in row
    }


def score_one_lambda(
    *,
    lambda_weight: float,
    scored_keys: list[tuple[str, str]],
    pred_means: dict[tuple[str, str], np.ndarray],
    pred_counts: dict[tuple[str, str], int],
    real_means: dict[tuple[str, str], np.ndarray],
    real_counts: dict[tuple[str, str], int],
    pred_control: dict[str, np.ndarray],
    real_control: dict[str, np.ndarray],
    center_table: dict[tuple[str, str], np.ndarray],
    missing_policy: str,
    pdc_center_clip_min: float | None,
    support: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context, pert in scored_keys:
        key = (context, pert)
        mu_model = pred_means[key]
        prior_delta = center_table.get(key)
        center_status = "found"
        if prior_delta is None:
            if missing_policy == "error":
                raise KeyError(f"Missing center for context={context!r}, perturbation={pert!r}")
            prior_delta = np.zeros_like(mu_model)
            center_status = "missing_keep_raw"
        mu_pdc = mu_model + float(lambda_weight) * (pred_control[context] + prior_delta - mu_model)
        if pdc_center_clip_min is not None:
            mu_pdc = np.maximum(mu_pdc, float(pdc_center_clip_min))
        real_mean = real_means[key]
        pred_delta = mu_pdc - pred_control[context]
        real_delta = real_mean - real_control[context]
        mean_diff = mu_pdc - real_mean
        delta_diff = pred_delta - real_delta
        row: dict[str, Any] = {
            "lambda": float(lambda_weight),
            "context": context,
            "perturbation": pert,
            "delta_pearson": pearson(pred_delta, real_delta),
            "rmse_mean": float(np.sqrt(np.mean(mean_diff * mean_diff))),
            "rmse_delta": float(np.sqrt(np.mean(delta_diff * delta_diff))),
            "mae_mean": float(np.mean(np.abs(mean_diff))),
            "mae_delta": float(np.mean(np.abs(delta_diff))),
            "pred_count": int(pred_counts[key]),
            "real_count": int(real_counts[key]),
            "center_status": center_status,
        }
        for k in TOP_KS:
            row[f"delta_overlap_{k}"] = overlap_at_k(real_delta, pred_delta, k)
            row[f"pdf_overlap_{k}"] = overlap_at_k(real_mean, mu_pdc, k)
        if key in support:
            info = support[key]
            row["center_mode"] = info.get("mode", "")
            row["support_context_count"] = info.get("support_context_count", "")
            row["support_cell_count"] = info.get("support_cell_count", "")
            row["exact_target_combo_present"] = info.get("exact_target_combo_present", "")
        rows.append(row)
    return rows


def summarize_lambda_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "lambda": rows[0]["lambda"] if rows else float("nan"),
        "n_groups": len(rows),
        "mean_delta_pcc": finite_mean([float(row["delta_pearson"]) for row in rows]),
        "median_delta_pcc": finite_median([float(row["delta_pearson"]) for row in rows]),
        "rmse_mean_mean": finite_mean([float(row["rmse_mean"]) for row in rows]),
        "rmse_delta_mean": finite_mean([float(row["rmse_delta"]) for row in rows]),
        "mae_mean_mean": finite_mean([float(row["mae_mean"]) for row in rows]),
        "mae_delta_mean": finite_mean([float(row["mae_delta"]) for row in rows]),
    }
    for prefix in ("delta_overlap", "pdf_overlap"):
        for k in TOP_KS:
            values = [float(row[f"{prefix}_{k}"]) for row in rows]
            summary[f"{prefix}_{k}_mean"] = finite_mean(values)
            summary[f"{prefix}_{k}_median"] = finite_median(values)
    summary["delta_overlap_score"] = (
        0.5 * summary["delta_overlap_100_mean"]
        + 0.3 * summary["delta_overlap_200_mean"]
        + 0.2 * summary["delta_overlap_500_mean"]
    )
    summary["pdf_overlap_score"] = (
        0.5 * summary["pdf_overlap_100_mean"]
        + 0.3 * summary["pdf_overlap_200_mean"]
        + 0.2 * summary["pdf_overlap_500_mean"]
    )
    return summary


def objective(summary: dict[str, Any], *, prefix: str, weights: dict[int, float]) -> float:
    score = 0.0
    for k, weight in weights.items():
        value = float(summary.get(f"{prefix}_{k}_mean", float("nan")))
        if not math.isfinite(value):
            return float("-inf")
        score += float(weight) * value
    return score


def select_lambdas(
    summaries: list[dict[str, Any]],
    *,
    pcc_floor: float,
    overlap_prefix: str,
    weights: dict[int, float],
) -> dict[str, Any]:
    best_by_pcc = max(summaries, key=lambda row: float(row["mean_delta_pcc"]))
    best_by_overlap = max(
        summaries,
        key=lambda row: (
            objective(row, prefix=overlap_prefix, weights=weights),
            float(row["mean_delta_pcc"]),
            -float(row["lambda"]),
        ),
    )
    best_pcc = float(best_by_pcc["mean_delta_pcc"])
    candidates = [
        row
        for row in summaries
        if float(row["mean_delta_pcc"]) >= best_pcc - float(pcc_floor)
    ]
    if not candidates:
        candidates = [best_by_pcc]
    selected = max(
        candidates,
        key=lambda row: (
            objective(row, prefix=overlap_prefix, weights=weights),
            float(row["mean_delta_pcc"]),
            -float(row["lambda"]),
        ),
    )
    return {
        "selection_rule": "maximize overlap score among lambdas with mean_delta_pcc >= best_delta_pcc - pcc_floor",
        "pcc_floor": float(pcc_floor),
        "overlap_prefix": overlap_prefix,
        "selection_weights": {str(k): v for k, v in weights.items()},
        "best_by_delta_pcc": {
            "lambda": float(best_by_pcc["lambda"]),
            "mean_delta_pcc": float(best_by_pcc["mean_delta_pcc"]),
            "overlap_score": objective(best_by_pcc, prefix=overlap_prefix, weights=weights),
        },
        "best_by_overlap": {
            "lambda": float(best_by_overlap["lambda"]),
            "mean_delta_pcc": float(best_by_overlap["mean_delta_pcc"]),
            "overlap_score": objective(best_by_overlap, prefix=overlap_prefix, weights=weights),
        },
        "selected": {
            "lambda": float(selected["lambda"]),
            "mean_delta_pcc": float(selected["mean_delta_pcc"]),
            "overlap_score": objective(selected, prefix=overlap_prefix, weights=weights),
        },
        "num_candidates_passing_pcc_floor": len(candidates),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--real-h5ad", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--center-table", type=Path, default=DEFAULT_CENTER_TABLE)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--lambda-grid", default=DEFAULT_LAMBDA_GRID)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--pred-clip-min", type=float, default=0.0)
    parser.add_argument("--real-clip-min", type=float, default=0.0)
    parser.add_argument(
        "--pdc-center-clip-min",
        type=float,
        default=None,
        help="Optional center-level clipping after applying PDC. Leave unset to match existing mean-level sweep.",
    )
    parser.add_argument("--missing-policy", choices=("error", "keep"), default="error")
    parser.add_argument("--pcc-floor", type=float, default=0.005)
    parser.add_argument("--selection-overlap-prefix", choices=("delta_overlap", "pdf_overlap"), default="delta_overlap")
    parser.add_argument("--selection-topk-weights", default="100:0.5,200:0.3,500:0.2")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit-groups", type=int, default=None, help="Debug only: score the first N sorted groups")
    parser.add_argument("--write-per-group", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    weights = parse_weights(args.selection_topk_weights)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else out_dir / "cache"

    pred_means, pred_counts, n_vars, pred_cache_path, pred_cache_status = load_or_compute_grouped_means(
        args.pred_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.pred_clip_min,
        cache_dir=cache_dir,
    )
    real_means, real_counts, real_n_vars, real_cache_path, real_cache_status = load_or_compute_grouped_means(
        args.real_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.real_clip_min,
        cache_dir=cache_dir,
    )
    if int(real_n_vars) != int(n_vars):
        raise ValueError(f"real vars={real_n_vars}, pred vars={n_vars}")

    pred_control, _ = split_control_means(pred_means, pred_counts, control_label=args.control_label)
    real_control, _ = split_control_means(real_means, real_counts, control_label=args.control_label)
    center_table = load_center_table(args.center_table, n_vars)
    support = load_support(args.support_csv)

    scored_keys = sorted(
        key
        for key in (set(pred_means) & set(real_means))
        if key[1] != args.control_label
        and key[0] in pred_control
        and key[0] in real_control
        and (args.missing_policy == "keep" or key in center_table)
    )
    if args.limit_groups is not None:
        scored_keys = scored_keys[: int(args.limit_groups)]
    if not scored_keys:
        raise RuntimeError("No scorable target groups")

    print(
        f"[score] groups={len(scored_keys)} lambdas={lambda_grid} "
        f"pred_cache={pred_cache_status} real_cache={real_cache_status}",
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for lambda_weight in lambda_grid:
        rows = score_one_lambda(
            lambda_weight=lambda_weight,
            scored_keys=scored_keys,
            pred_means=pred_means,
            pred_counts=pred_counts,
            real_means=real_means,
            real_counts=real_counts,
            pred_control=pred_control,
            real_control=real_control,
            center_table=center_table,
            missing_policy=args.missing_policy,
            pdc_center_clip_min=args.pdc_center_clip_min,
            support=support,
        )
        summary = summarize_lambda_rows(rows)
        summaries.append(summary)
        if args.write_per_group:
            all_rows.extend(rows)
        print(
            "[score] lambda={lam:g} mean_delta_pcc={pcc:.6f} "
            "delta_overlap100={ov100:.6f} delta_overlap500={ov500:.6f}".format(
                lam=float(lambda_weight),
                pcc=float(summary["mean_delta_pcc"]),
                ov100=float(summary["delta_overlap_100_mean"]),
                ov500=float(summary["delta_overlap_500_mean"]),
            ),
            flush=True,
        )

    summary_fields = [
        "lambda",
        "n_groups",
        "mean_delta_pcc",
        "median_delta_pcc",
        "delta_overlap_score",
        "pdf_overlap_score",
        "rmse_mean_mean",
        "rmse_delta_mean",
        "mae_mean_mean",
        "mae_delta_mean",
    ]
    for prefix in ("delta_overlap", "pdf_overlap"):
        for k in TOP_KS:
            summary_fields.extend([f"{prefix}_{k}_mean", f"{prefix}_{k}_median"])
    summary_csv = out_dir / "tahoe100m_pdc_overlap_lambda_sweep_summary.csv"
    write_csv(summary_csv, summaries, summary_fields)

    per_group_csv = None
    if args.write_per_group:
        group_fields = [
            "lambda",
            "context",
            "perturbation",
            "delta_pearson",
            "rmse_mean",
            "rmse_delta",
            "mae_mean",
            "mae_delta",
        ]
        for k in TOP_KS:
            group_fields.extend([f"delta_overlap_{k}", f"pdf_overlap_{k}"])
        group_fields.extend(
            [
                "pred_count",
                "real_count",
                "center_status",
                "center_mode",
                "support_context_count",
                "support_cell_count",
                "exact_target_combo_present",
            ]
        )
        per_group_csv = out_dir / "tahoe100m_pdc_overlap_lambda_sweep_per_group.csv"
        write_csv(per_group_csv, all_rows, group_fields)

    selection = select_lambdas(
        summaries,
        pcc_floor=float(args.pcc_floor),
        overlap_prefix=args.selection_overlap_prefix,
        weights=weights,
    )
    selection_json = out_dir / "tahoe100m_pdc_overlap_lambda_selection.json"
    payload = {
        "metric_scope": "mean-level diagnostic, not exact cell-eval DE overlap",
        "formula": "mu_pdc = mu_model + lambda * (pred_control + D_train - mu_model)",
        "pred_h5ad": str(args.pred_h5ad),
        "real_h5ad_eval_only": str(args.real_h5ad),
        "center_table": str(args.center_table),
        "support_csv": str(args.support_csv) if args.support_csv else None,
        "context_col": args.context_col,
        "pert_col": args.pert_col,
        "control_label": args.control_label,
        "lambda_grid": lambda_grid,
        "pred_clip_min": args.pred_clip_min,
        "real_clip_min": args.real_clip_min,
        "pdc_center_clip_min": args.pdc_center_clip_min,
        "num_scored_groups": len(scored_keys),
        "num_center_vectors": len(center_table),
        "summary_csv": str(summary_csv),
        "per_group_csv": str(per_group_csv) if per_group_csv else None,
        "selection": selection,
        "cache": {
            "pred_cache_status": pred_cache_status,
            "pred_cache_path": str(pred_cache_path) if pred_cache_path else None,
            "real_cache_status": real_cache_status,
            "real_cache_path": str(real_cache_path) if real_cache_path else None,
        },
        "summaries": summaries,
        "note": (
            "Use selected lambda as a fast screen. For final Tahoe overlap, materialize that lambda "
            "and run benchmark/evaluate_tahoe.py because cell-eval overlap uses DE rank/FDR."
        ),
    }
    selection_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[wrote] {summary_csv}", flush=True)
    if per_group_csv is not None:
        print(f"[wrote] {per_group_csv}", flush=True)
    print(f"[wrote] {selection_json}", flush=True)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
