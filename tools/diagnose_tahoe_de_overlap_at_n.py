#!/usr/bin/env python3
"""Diagnose Tahoe cell-eval ``overlap_at_N`` bottlenecks from DE CSV outputs.

The official metric is low when real significant genes do not appear in the
predicted top-N ranking.  This script separates that ranking failure from
significance recall:

* ``manual_overlap_at_N`` reproduces the cell-eval top-N overlap from
  ``*_real_de.csv`` and ``*_pred_de.csv``.
* ``sig_recall_upper_bound`` is the fraction of real significant genes that
  are also predicted significant.  This is the best overlap attainable by a
  perfect re-ranking of the existing predicted significant set.
* ``rank_frac_le_{m}N`` reports how many real significant genes land within
  predicted top ``m * N``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_DE_DIR = (
    Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang")
    / "master/nemo_cellflow/benchmark/workspace/20260615_055828_JIT_FP_pdc_trainonly_20260624"
    / "lambda02/results_calibrate"
)
DEFAULT_OUT_DIR = (
    Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang")
    / "master/nemo_cellflow/benchmark/workspace/20260615_055828_JIT_FP_pdc_trainonly_20260624"
    / "de_overlap_at_n_diagnostics_20260625"
)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def finite_mean(values: list[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else float("nan")


def finite_median(values: list[float]) -> float:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    return statistics.median(xs) if xs else float("nan")


def weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = float(row[value_key])
        weight = float(row[weight_key])
        if math.isfinite(value) and math.isfinite(weight) and weight > 0.0:
            numerator += value * weight
            denominator += weight
    return numerator / denominator if denominator else float("nan")


def abs_log2_fold_change(raw_value: str) -> float:
    try:
        fold_change = float(raw_value)
    except ValueError:
        return 0.0
    if not math.isfinite(fold_change) or fold_change <= 0.0:
        return 0.0
    value = math.log(fold_change, 2)
    return abs(value) if math.isfinite(value) else 0.0


def load_de_by_target(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    by_target: dict[str, list[tuple[str, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                fdr = float(row["fdr"])
            except (KeyError, ValueError):
                continue
            by_target.setdefault(str(row["target"]), []).append(
                (str(row["feature"]), fdr, abs_log2_fold_change(str(row["fold_change"])))
            )
    return by_target


def load_official_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["perturbation"]): row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def compute_rows(de_dir: Path, fdr_threshold: float, rank_multipliers: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real_paths = sorted(de_dir.glob("*_real_de.csv"))
    if not real_paths:
        raise FileNotFoundError(f"No *_real_de.csv files found under {de_dir}")

    for real_path in real_paths:
        context = real_path.name[: -len("_real_de.csv")]
        pred_path = de_dir / f"{context}_pred_de.csv"
        results_path = de_dir / f"{context}_results.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predicted DE file for {context}: {pred_path}")

        real_by_target = load_de_by_target(real_path)
        pred_by_target = load_de_by_target(pred_path)
        official = load_official_results(results_path)

        for target, real_records in real_by_target.items():
            pred_records = pred_by_target.get(target, [])
            real_sig = [
                (gene, abs_lfc)
                for gene, fdr, abs_lfc in real_records
                if math.isfinite(fdr) and fdr < fdr_threshold
            ]
            pred_sig = [
                (gene, abs_lfc)
                for gene, fdr, abs_lfc in pred_records
                if math.isfinite(fdr) and fdr < fdr_threshold
            ]
            if not real_sig:
                continue

            real_sig_sorted = sorted(real_sig, key=lambda item: item[1], reverse=True)
            pred_sig_sorted = sorted(pred_sig, key=lambda item: item[1], reverse=True)
            n = len(real_sig_sorted)
            real_sig_set = {gene for gene, _ in real_sig_sorted}
            pred_sig_set = {gene for gene, _ in pred_sig_sorted}
            pred_rank = {gene: idx + 1 for idx, (gene, _) in enumerate(pred_sig_sorted)}
            pred_top_n_set = {gene for gene, _ in pred_sig_sorted[: min(n, len(pred_sig_sorted))]}
            ranks = [pred_rank[gene] for gene in real_sig_set if gene in pred_rank]

            row: dict[str, Any] = {
                "context": context,
                "perturbation": target,
                "real_nsig": n,
                "pred_nsig": len(pred_sig_sorted),
                "official_overlap_at_N": float(official.get(target, {}).get("overlap_at_N", "nan")),
                "manual_overlap_at_N": len(real_sig_set & pred_top_n_set) / float(n),
                "sig_recall_upper_bound": len(real_sig_set & pred_sig_set) / float(n),
                "random_topN_expected_overlap": (
                    len(real_sig_set & pred_sig_set) / float(len(pred_sig_sorted))
                    if pred_sig_sorted
                    else 0.0
                ),
                "pred_rank_median_for_real_sig": statistics.median(ranks) if ranks else float("nan"),
                "pred_rank_p90_for_real_sig": (
                    statistics.quantiles(ranks, n=10)[8]
                    if len(ranks) >= 10
                    else (max(ranks) if ranks else float("nan"))
                ),
            }
            for multiplier in rank_multipliers:
                threshold = multiplier * n
                row[f"rank_frac_le_{multiplier}N"] = (
                    sum(1 for rank in ranks if rank <= threshold) / float(n)
                )
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], rank_multipliers: list[int]) -> dict[str, Any]:
    metric_keys = [
        "official_overlap_at_N",
        "manual_overlap_at_N",
        "sig_recall_upper_bound",
        "random_topN_expected_overlap",
    ] + [f"rank_frac_le_{multiplier}N" for multiplier in rank_multipliers]

    out: dict[str, Any] = {
        "n_groups": len(rows),
        "real_nsig_mean": finite_mean([float(row["real_nsig"]) for row in rows]),
        "real_nsig_median": finite_median([float(row["real_nsig"]) for row in rows]),
        "pred_nsig_mean": finite_mean([float(row["pred_nsig"]) for row in rows]),
        "pred_nsig_median": finite_median([float(row["pred_nsig"]) for row in rows]),
    }
    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = finite_mean(values)
        out[f"{key}_median"] = finite_median(values)
        out[f"{key}_real_nsig_weighted"] = weighted_mean(rows, key, "real_nsig")
    return out


def summarize_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_context: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_context.setdefault(str(row["context"]), []).append(row)

    context_rows: list[dict[str, Any]] = []
    for context, context_group in sorted(by_context.items()):
        context_rows.append(
            {
                "context": context,
                "n_groups": len(context_group),
                "official_overlap_at_N_mean": finite_mean(
                    [float(row["official_overlap_at_N"]) for row in context_group]
                ),
                "manual_overlap_at_N_mean": finite_mean(
                    [float(row["manual_overlap_at_N"]) for row in context_group]
                ),
                "sig_recall_upper_bound_mean": finite_mean(
                    [float(row["sig_recall_upper_bound"]) for row in context_group]
                ),
                "real_nsig_mean": finite_mean([float(row["real_nsig"]) for row in context_group]),
                "pred_nsig_mean": finite_mean([float(row["pred_nsig"]) for row in context_group]),
            }
        )
    return context_rows


def parse_rank_multipliers(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--rank-multipliers must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--rank-multipliers", default="1,2,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank_multipliers = parse_rank_multipliers(args.rank_multipliers)
    rows = compute_rows(args.de_dir, float(args.fdr_threshold), rank_multipliers)
    summary = summarize(rows, rank_multipliers)
    context_rows = summarize_contexts(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_group_csv = args.out_dir / "tahoe100m_de_overlap_at_n_diagnostics_per_group.csv"
    context_csv = args.out_dir / "tahoe100m_de_overlap_at_n_diagnostics_by_context.csv"
    summary_json = args.out_dir / "tahoe100m_de_overlap_at_n_diagnostics_summary.json"

    per_group_fields = [
        "context",
        "perturbation",
        "real_nsig",
        "pred_nsig",
        "official_overlap_at_N",
        "manual_overlap_at_N",
        "sig_recall_upper_bound",
        "random_topN_expected_overlap",
        "pred_rank_median_for_real_sig",
        "pred_rank_p90_for_real_sig",
    ] + [f"rank_frac_le_{multiplier}N" for multiplier in rank_multipliers]
    write_csv(per_group_csv, rows, per_group_fields)
    write_csv(
        context_csv,
        context_rows,
        [
            "context",
            "n_groups",
            "official_overlap_at_N_mean",
            "manual_overlap_at_N_mean",
            "sig_recall_upper_bound_mean",
            "real_nsig_mean",
            "pred_nsig_mean",
        ],
    )

    payload = {
        "metric_scope": "diagnostic reproduction and bottleneck analysis for cell-eval overlap_at_N",
        "de_dir": str(args.de_dir),
        "fdr_threshold": float(args.fdr_threshold),
        "rank_multipliers": rank_multipliers,
        "summary": summary,
        "per_group_csv": str(per_group_csv),
        "context_csv": str(context_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
