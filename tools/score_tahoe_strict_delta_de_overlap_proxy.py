#!/usr/bin/env python3
"""Diagnostic strict-delta grid for Tahoe-100M cell-eval overlap_at_N proxy.

Scores

    pred_center = pred_control + model_weight * raw_model_delta
                  + prior_weight * D_train

against existing real DE top-N genes.  This is a diagnostic test sweep, not a
paper-ready selection protocol, because the held-out real DE files are used for
scoring the grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from apply_prior_delta_centering import split_control_means
from score_tahoe_pdc_de_overlap_proxy import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REAL_DE_DIR,
    load_real_top_genes,
    safe_abs_log2_fold_change,
    var_names_from_h5ad,
    write_csv,
)
from score_tahoe_pdc_overlap_sweep import (
    DEFAULT_CENTER_TABLE,
    DEFAULT_CONTROL,
    DEFAULT_PRED,
    finite_mean,
    finite_median,
    load_center_table,
    load_or_compute_grouped_means,
)


DEFAULT_OUT_DIR = (
    Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang")
    / "master/nemo_cellflow/benchmark/workspace/20260615_055828_JIT_FP_pdc_trainonly_20260624"
    / "strict_delta_de_overlap_proxy_20260625"
)


def parse_grid(text: str) -> list[float]:
    return sorted({round(float(part.strip()), 10) for part in text.split(",") if part.strip()})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def score_weights(
    *,
    model_weight: float,
    prior_weight: float,
    keys: list[tuple[str, str]],
    pred_means: dict[tuple[str, str], np.ndarray],
    pred_counts: dict[tuple[str, str], int],
    pred_control: dict[str, np.ndarray],
    center_table: dict[tuple[str, str], np.ndarray],
    real_top: dict[tuple[str, str], np.ndarray],
    eps: float,
    clip_min: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context, pert in keys:
        key = (context, pert)
        real_genes = real_top[key]
        n = int(real_genes.size)
        if n <= 0:
            continue
        raw_delta = pred_means[key] - pred_control[context]
        pred_center = (
            pred_control[context]
            + float(model_weight) * raw_delta
            + float(prior_weight) * center_table[key]
        )
        if clip_min is not None:
            pred_center = np.maximum(pred_center, float(clip_min))
        pred_abs_lfc = safe_abs_log2_fold_change(pred_center, pred_control[context], eps)
        kk = min(n, int(pred_abs_lfc.size))
        pred_top = np.argpartition(pred_abs_lfc, -kk)[-kk:]
        pred_top = pred_top[np.argsort(pred_abs_lfc[pred_top])[::-1]]
        overlap = len(set(real_genes[:kk].tolist()).intersection(pred_top[:kk].tolist())) / float(kk)
        rows.append(
            {
                "model_weight": float(model_weight),
                "prior_weight": float(prior_weight),
                "context": context,
                "perturbation": pert,
                "proxy_overlap_at_N": overlap,
                "real_nsig": n,
                "pred_count": int(pred_counts[key]),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = [float(row["proxy_overlap_at_N"]) for row in rows]
    nsig = [float(row["real_nsig"]) for row in rows]
    weighted_num = sum(float(row["proxy_overlap_at_N"]) * float(row["real_nsig"]) for row in rows)
    weighted_den = sum(float(row["real_nsig"]) for row in rows)
    return {
        "n_groups": len(rows),
        "proxy_overlap_at_N_mean": finite_mean(overlaps),
        "proxy_overlap_at_N_median": finite_median(overlaps),
        "proxy_overlap_at_N_real_nsig_weighted": weighted_num / weighted_den if weighted_den else float("nan"),
        "real_nsig_mean": finite_mean(nsig),
        "real_nsig_median": finite_median(nsig),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--center-table", type=Path, default=DEFAULT_CENTER_TABLE)
    parser.add_argument("--real-de-dir", type=Path, default=DEFAULT_REAL_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--model-grid", default="0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--prior-grid", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--write-per-group", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_grid = parse_grid(args.model_grid)
    prior_grid = parse_grid(args.prior_grid)
    genes = var_names_from_h5ad(args.pred_h5ad)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    real_top = load_real_top_genes(
        real_de_dir=args.real_de_dir,
        fdr_threshold=float(args.fdr_threshold),
        gene_to_idx=gene_to_idx,
    )
    pred_means, pred_counts, n_vars, pred_cache, pred_cache_status = load_or_compute_grouped_means(
        args.pred_h5ad,
        context_col=args.context_col,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.clip_min,
        cache_dir=args.cache_dir,
    )
    pred_control, _ = split_control_means(pred_means, pred_counts, control_label=args.control_label)
    center_table = load_center_table(args.center_table, n_vars)
    keys = sorted(
        key
        for key in (set(pred_means) & set(center_table) & set(real_top))
        if key[1] != args.control_label and key[0] in pred_control
    )
    if args.limit_groups is not None:
        keys = keys[: int(args.limit_groups)]
    if not keys:
        raise RuntimeError("No scorable groups")

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for model_weight in model_grid:
        for prior_weight in prior_grid:
            rows = score_weights(
                model_weight=model_weight,
                prior_weight=prior_weight,
                keys=keys,
                pred_means=pred_means,
                pred_counts=pred_counts,
                pred_control=pred_control,
                center_table=center_table,
                real_top=real_top,
                eps=float(args.eps),
                clip_min=args.clip_min,
            )
            summary = summarize_rows(rows)
            summary["model_weight"] = float(model_weight)
            summary["prior_weight"] = float(prior_weight)
            summaries.append(summary)
            if args.write_per_group:
                all_rows.extend(rows)
            print(
                "[strict-proxy] model={m:g} prior={p:g} overlapN={s:.6f}".format(
                    m=float(model_weight),
                    p=float(prior_weight),
                    s=float(summary["proxy_overlap_at_N_mean"]),
                ),
                flush=True,
            )

    best = max(summaries, key=lambda row: float(row["proxy_overlap_at_N_mean"]))
    fields = [
        "model_weight",
        "prior_weight",
        "n_groups",
        "proxy_overlap_at_N_mean",
        "proxy_overlap_at_N_median",
        "proxy_overlap_at_N_real_nsig_weighted",
        "real_nsig_mean",
        "real_nsig_median",
    ]
    summary_csv = args.out_dir / "tahoe100m_strict_delta_de_overlap_proxy_summary.csv"
    write_csv(summary_csv, summaries, fields)
    per_group_csv = None
    if args.write_per_group:
        per_group_csv = args.out_dir / "tahoe100m_strict_delta_de_overlap_proxy_per_group.csv"
        write_csv(
            per_group_csv,
            all_rows,
            [
                "model_weight",
                "prior_weight",
                "context",
                "perturbation",
                "proxy_overlap_at_N",
                "real_nsig",
                "pred_count",
            ],
        )
    selection_json = args.out_dir / "tahoe100m_strict_delta_de_overlap_proxy_selection.json"
    payload = {
        "metric_scope": "diagnostic strict-delta proxy for cell-eval overlap_at_N",
        "formula": "pred_center = pred_control + model_weight * raw_model_delta + prior_weight * D_train",
        "pred_h5ad": str(args.pred_h5ad),
        "real_de_dir_eval_only": str(args.real_de_dir),
        "center_table": str(args.center_table),
        "model_grid": model_grid,
        "prior_grid": prior_grid,
        "fdr_threshold": float(args.fdr_threshold),
        "num_scored_groups": len(keys),
        "pred_cache_status": pred_cache_status,
        "pred_cache_path": str(pred_cache) if pred_cache else None,
        "summary_csv": str(summary_csv),
        "per_group_csv": str(per_group_csv) if per_group_csv else None,
        "best_by_proxy_overlap_at_N": best,
        "summaries": summaries,
    }
    selection_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wrote] {summary_csv}")
    if per_group_csv:
        print(f"[wrote] {per_group_csv}")
    print(f"[wrote] {selection_json}")
    print(json.dumps({"best_by_proxy_overlap_at_N": best}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
