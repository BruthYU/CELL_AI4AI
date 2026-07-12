#!/usr/bin/env python3
"""Approximate Tahoe-100M cell-eval overlap_at_N for PDC lambda sweeps.

This is a diagnostic proxy for cell-eval DE overlap:

* Real top-N genes are read from existing ``*_real_de.csv`` files, filtered by
  FDR and sorted by absolute log2 fold change, matching cell-eval.
* Predicted rankings are computed on the fly from PDC group means as absolute
  log2 fold change versus the predicted control mean.

It avoids materializing/evaluating every lambda.  Final claims still require
running ``benchmark/evaluate_tahoe.py`` on a materialized h5ad because cell-eval
uses full-cell DE statistics and predicted FDR filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from apply_prior_delta_centering import split_control_means
from score_tahoe_pdc_overlap_sweep import (
    BASE,
    DEFAULT_CENTER_TABLE,
    DEFAULT_CONTROL,
    DEFAULT_LAMBDA_GRID,
    DEFAULT_PRED,
    DEFAULT_REAL,
    DEFAULT_SUPPORT_CSV,
    PDC_DIR,
    finite_mean,
    finite_median,
    fmt,
    load_center_table,
    load_or_compute_grouped_means,
    parse_lambda_grid,
)


DEFAULT_REAL_DE_DIR = PDC_DIR / "lambda02/results_calibrate"
DEFAULT_OUT_DIR = PDC_DIR / "de_overlap_proxy_20260625"
DEFAULT_CACHE_DIR = PDC_DIR / "overlap_sweep_20260625/cache"


def var_names_from_h5ad(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_name = str(var.attrs.get("_index", "_index"))
        if index_name in var:
            obj = var[index_name]
        elif "_index" in var:
            obj = var["_index"]
        else:
            raise KeyError(f"{path}: cannot locate var index")
        values = obj.asstr()[:] if hasattr(obj, "asstr") else obj[:].astype(str)
    return [str(value) for value in values]


def safe_abs_log2_fold_change(target: np.ndarray, reference: np.ndarray, eps: float) -> np.ndarray:
    target_safe = np.maximum(np.asarray(target, dtype=np.float64), float(eps))
    reference_safe = np.maximum(np.asarray(reference, dtype=np.float64), float(eps))
    return np.abs(np.log2(target_safe / reference_safe))


def load_real_top_genes(
    *,
    real_de_dir: Path,
    fdr_threshold: float,
    gene_to_idx: dict[str, int],
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    real_de_files = sorted(real_de_dir.glob("*_real_de.csv"))
    if not real_de_files:
        raise FileNotFoundError(f"No *_real_de.csv files found under {real_de_dir}")
    for path in real_de_files:
        context = path.name[: -len("_real_de.csv")]
        by_pert: dict[str, list[tuple[int, float]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    fdr = float(row["fdr"])
                    fold_change = float(row["fold_change"])
                except (KeyError, ValueError):
                    continue
                if not math.isfinite(fdr) or fdr >= float(fdr_threshold):
                    continue
                gene_idx = gene_to_idx.get(str(row.get("feature", "")))
                if gene_idx is None:
                    continue
                abs_lfc = abs(math.log(fold_change, 2)) if fold_change > 0.0 else 0.0
                by_pert.setdefault(str(row["target"]), []).append((gene_idx, abs_lfc))
        for pert, values in by_pert.items():
            values.sort(key=lambda item: item[1], reverse=True)
            out[(context, pert)] = np.asarray([gene_idx for gene_idx, _ in values], dtype=np.int64)
    return out


def score_lambda(
    *,
    lambda_weight: float,
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
        mu_model = pred_means[key]
        mu_pdc = mu_model + float(lambda_weight) * (pred_control[context] + center_table[key] - mu_model)
        if clip_min is not None:
            mu_pdc = np.maximum(mu_pdc, float(clip_min))
        pred_abs_lfc = safe_abs_log2_fold_change(mu_pdc, pred_control[context], eps)
        kk = min(n, int(pred_abs_lfc.size))
        pred_top = np.argpartition(pred_abs_lfc, -kk)[-kk:]
        pred_top = pred_top[np.argsort(pred_abs_lfc[pred_top])[::-1]]
        overlap = len(set(real_genes[:kk].tolist()).intersection(pred_top[:kk].tolist())) / float(kk)
        rows.append(
            {
                "lambda": float(lambda_weight),
                "context": context,
                "perturbation": pert,
                "proxy_overlap_at_N": overlap,
                "real_nsig": n,
                "pred_count": int(pred_counts[key]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = [float(row["proxy_overlap_at_N"]) for row in rows]
    nsig = [float(row["real_nsig"]) for row in rows]
    weighted_num = sum(float(row["proxy_overlap_at_N"]) * float(row["real_nsig"]) for row in rows)
    weighted_den = sum(float(row["real_nsig"]) for row in rows)
    return {
        "lambda": rows[0]["lambda"] if rows else float("nan"),
        "n_groups": len(rows),
        "proxy_overlap_at_N_mean": finite_mean(overlaps),
        "proxy_overlap_at_N_median": finite_median(overlaps),
        "proxy_overlap_at_N_real_nsig_weighted": weighted_num / weighted_den if weighted_den else float("nan"),
        "real_nsig_mean": finite_mean(nsig),
        "real_nsig_median": finite_median(nsig),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--real-h5ad", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--center-table", type=Path, default=DEFAULT_CENTER_TABLE)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--real-de-dir", type=Path, default=DEFAULT_REAL_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--lambda-grid", default=DEFAULT_LAMBDA_GRID)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--write-per-group", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    cache_dir = args.cache_dir

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
        cache_dir=cache_dir,
    )
    if int(n_vars) != len(genes):
        raise ValueError(f"n_vars={n_vars}, var_names={len(genes)}")
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
    for lambda_weight in lambda_grid:
        rows = score_lambda(
            lambda_weight=lambda_weight,
            keys=keys,
            pred_means=pred_means,
            pred_counts=pred_counts,
            pred_control=pred_control,
            center_table=center_table,
            real_top=real_top,
            eps=float(args.eps),
            clip_min=args.clip_min,
        )
        summary = summarize(rows)
        summaries.append(summary)
        if args.write_per_group:
            all_rows.extend(rows)
        print(
            "[proxy] lambda={lam:g} overlapN={score:.6f} weighted={weighted:.6f}".format(
                lam=float(lambda_weight),
                score=float(summary["proxy_overlap_at_N_mean"]),
                weighted=float(summary["proxy_overlap_at_N_real_nsig_weighted"]),
            ),
            flush=True,
        )

    best = max(summaries, key=lambda row: float(row["proxy_overlap_at_N_mean"]))
    summary_csv = args.out_dir / "tahoe100m_pdc_de_overlap_proxy_summary.csv"
    write_csv(
        summary_csv,
        summaries,
        [
            "lambda",
            "n_groups",
            "proxy_overlap_at_N_mean",
            "proxy_overlap_at_N_median",
            "proxy_overlap_at_N_real_nsig_weighted",
            "real_nsig_mean",
            "real_nsig_median",
        ],
    )
    per_group_csv = None
    if args.write_per_group:
        per_group_csv = args.out_dir / "tahoe100m_pdc_de_overlap_proxy_per_group.csv"
        write_csv(
            per_group_csv,
            all_rows,
            ["lambda", "context", "perturbation", "proxy_overlap_at_N", "real_nsig", "pred_count"],
        )
    selection_json = args.out_dir / "tahoe100m_pdc_de_overlap_proxy_selection.json"
    payload = {
        "metric_scope": "diagnostic proxy for cell-eval overlap_at_N; final values require materialized h5ad cell-eval",
        "pred_h5ad": str(args.pred_h5ad),
        "real_de_dir_eval_only": str(args.real_de_dir),
        "center_table": str(args.center_table),
        "fdr_threshold": float(args.fdr_threshold),
        "lambda_grid": lambda_grid,
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
