#!/usr/bin/env python3
"""Build an oracle DE CSV set that targets Tahoe cell-eval overlap_at_N.

This is intentionally a leakage/oracle diagnostic.  It uses the held-out real
DE rank to rewrite predicted DE CSVs, then scores the rewritten files with the
same top-N overlap rule used by cell-eval.  It answers one narrow question:
whether reaching a requested overlap_at_N level is mechanically possible if
the true DE rank information is injected.

It must not be treated as train-only PDC or a valid paper-ready method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
    / "de_overlap_oracle_target0p8_20260625"
)
METRIC_KS = [None, 50, 100, 200, 500]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def abs_log2_fold_change(value: str) -> float:
    fold_change = parse_float(value)
    if not math.isfinite(fold_change) or fold_change <= 0.0:
        return 0.0
    out = math.log(fold_change, 2)
    return abs(out) if math.isfinite(out) else 0.0


def signed_log2_fold_change(value: str) -> float:
    fold_change = parse_float(value)
    if not math.isfinite(fold_change) or fold_change <= 0.0:
        return 0.0
    out = math.log(fold_change, 2)
    return out if math.isfinite(out) else 0.0


def load_real_rank(
    real_path: Path,
    fdr_threshold: float,
    target_overlap: float,
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, int], dict[str, int], dict[tuple[str, str], int]]:
    """Return oracle-selected genes and real-significant metadata."""
    by_target: dict[str, list[tuple[str, float, float]]] = {}
    with real_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fdr = parse_float(str(row.get("fdr", "")))
            if not math.isfinite(fdr) or fdr >= fdr_threshold:
                continue
            target = str(row["target"])
            gene = str(row["feature"])
            signed_lfc = signed_log2_fold_change(str(row["fold_change"]))
            by_target.setdefault(target, []).append((gene, abs(signed_lfc), signed_lfc))

    selected_by_target: dict[str, list[str]] = {}
    real_sig_by_target: dict[str, set[str]] = {}
    real_nsig_by_target: dict[str, int] = {}
    selected_count_by_target: dict[str, int] = {}
    selected_rank: dict[tuple[str, str], int] = {}
    for target, values in by_target.items():
        values.sort(key=lambda item: item[1], reverse=True)
        n = len(values)
        k = min(n, int(math.ceil(float(target_overlap) * n)))
        selected = [gene for gene, _, _ in values[:k]]
        selected_by_target[target] = selected
        real_sig_by_target[target] = {gene for gene, _, _ in values}
        real_nsig_by_target[target] = n
        selected_count_by_target[target] = k
        for rank, (gene, _, _) in enumerate(values[:k]):
            selected_rank[(target, gene)] = rank
    return (
        selected_by_target,
        real_sig_by_target,
        real_nsig_by_target,
        selected_count_by_target,
        selected_rank,
    )


def oracle_fold_change(rank: int, sign_hint: float, max_score: float, rank_step: float) -> float:
    score = max(float(max_score) - float(rank) * float(rank_step), 0.25)
    if sign_hint < 0.0:
        return 2.0 ** (-score)
    return 2.0**score


def load_real_signs(real_path: Path) -> dict[tuple[str, str], float]:
    signs: dict[tuple[str, str], float] = {}
    with real_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            signs[(str(row["target"]), str(row["feature"]))] = signed_log2_fold_change(
                str(row["fold_change"])
            )
    return signs


def write_oracle_pred_de(
    *,
    pred_path: Path,
    out_path: Path,
    selected_rank: dict[tuple[str, str], int],
    real_signs: dict[tuple[str, str], float],
    max_score: float,
    rank_step: float,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_seen: set[tuple[str, str]] = set()
    row_count = 0
    with pred_path.open(newline="", encoding="utf-8") as handle_in, out_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle_out:
        reader = csv.DictReader(handle_in)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {pred_path}")
        writer = csv.DictWriter(handle_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row_count += 1
            key = (str(row["target"]), str(row["feature"]))
            if key in selected_rank:
                selected_seen.add(key)
                rank = selected_rank[key]
                fold_change = oracle_fold_change(
                    rank=rank,
                    sign_hint=real_signs.get(key, 1.0),
                    max_score=max_score,
                    rank_step=rank_step,
                )
                row["fold_change"] = fmt(fold_change)
                row["percent_change"] = fmt(fold_change - 1.0)
                row["p_value"] = "0"
                row["fdr"] = "0"
            else:
                row["fold_change"] = "1"
                row["percent_change"] = "0"
                row["p_value"] = "1"
                row["fdr"] = "1"
            writer.writerow(row)
    return row_count, len(selected_seen)


def sorted_significant_genes(path: Path, fdr_threshold: float) -> dict[str, list[str]]:
    out: dict[str, list[tuple[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fdr = parse_float(str(row.get("fdr", "")))
            if not math.isfinite(fdr) or fdr >= fdr_threshold:
                continue
            out.setdefault(str(row["target"]), []).append(
                (str(row["feature"]), abs_log2_fold_change(str(row["fold_change"])))
            )
    return {
        target: [gene for gene, _ in sorted(values, key=lambda item: item[1], reverse=True)]
        for target, values in out.items()
    }


def overlap_score(real_genes: list[str], pred_genes: list[str], k: int | None, metric: str) -> float:
    if metric == "overlap":
        k_eff = len(real_genes) if k is None else min(int(k), len(real_genes))
    elif metric == "precision":
        k_eff = len(pred_genes) if k is None else min(int(k), len(pred_genes))
    else:
        raise ValueError(metric)
    if k_eff <= 0:
        return 0.0
    return len(set(real_genes[:k_eff]) & set(pred_genes[:k_eff])) / float(k_eff)


def score_context(real_path: Path, pred_path: Path, fdr_threshold: float) -> list[dict[str, Any]]:
    real_rank = sorted_significant_genes(real_path, fdr_threshold)
    pred_rank = sorted_significant_genes(pred_path, fdr_threshold)
    rows: list[dict[str, Any]] = []
    for target in sorted(real_rank):
        real_genes = real_rank[target]
        pred_genes = pred_rank.get(target, [])
        row: dict[str, Any] = {
            "perturbation": target,
            "real_nsig": len(real_genes),
            "pred_nsig": len(pred_genes),
        }
        for metric in ["overlap", "precision"]:
            for k in METRIC_KS:
                suffix = "N" if k is None else str(k)
                row[f"{metric}_at_{suffix}"] = overlap_score(real_genes, pred_genes, k, metric)
        rows.append(row)
    return rows


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "perturbation",
        "overlap_at_N",
        "overlap_at_50",
        "overlap_at_100",
        "overlap_at_200",
        "overlap_at_500",
        "precision_at_N",
        "precision_at_50",
        "precision_at_100",
        "precision_at_200",
        "precision_at_500",
        "real_nsig",
        "pred_nsig",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def write_agg_results(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        "overlap_at_N",
        "overlap_at_50",
        "overlap_at_100",
        "overlap_at_200",
        "overlap_at_500",
        "precision_at_N",
        "precision_at_50",
        "precision_at_100",
        "precision_at_200",
        "precision_at_500",
        "real_nsig",
        "pred_nsig",
    ]
    agg_rows: list[dict[str, Any]] = []
    for statistic in ["mean", "median", "min", "max"]:
        row: dict[str, Any] = {"statistic": statistic}
        for metric in metrics:
            values = [float(item[metric]) for item in rows if math.isfinite(float(item[metric]))]
            if not values:
                row[metric] = float("nan")
            elif statistic == "mean":
                row[metric] = sum(values) / len(values)
            elif statistic == "median":
                row[metric] = statistics.median(values)
            elif statistic == "min":
                row[metric] = min(values)
            elif statistic == "max":
                row[metric] = max(values)
        agg_rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["statistic", *metrics])
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in ["statistic", *metrics]})


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = [float(row["overlap_at_N"]) for row in rows]
    real_nsig = [float(row["real_nsig"]) for row in rows]
    weighted_num = sum(float(row["overlap_at_N"]) * float(row["real_nsig"]) for row in rows)
    weighted_den = sum(float(row["real_nsig"]) for row in rows)
    return {
        "n_groups": len(rows),
        "overlap_at_N_mean": sum(overlap) / len(overlap) if overlap else float("nan"),
        "overlap_at_N_median": statistics.median(overlap) if overlap else float("nan"),
        "overlap_at_N_real_nsig_weighted": weighted_num / weighted_den if weighted_den else float("nan"),
        "real_nsig_mean": sum(real_nsig) / len(real_nsig) if real_nsig else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-overlap", type=float, default=0.8)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--max-score", type=float, default=10.0)
    parser.add_argument("--rank-step", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.target_overlap) <= 1.0):
        raise ValueError("--target-overlap must be in [0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    total_rows_written = 0
    total_selected_seen = 0
    for real_path in sorted(args.de_dir.glob("*_real_de.csv")):
        context = real_path.name[: -len("_real_de.csv")]
        pred_path = args.de_dir / f"{context}_pred_de.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predicted DE CSV: {pred_path}")

        out_real = args.out_dir / f"{context}_real_de.csv"
        out_pred = args.out_dir / f"{context}_pred_de.csv"
        shutil.copyfile(real_path, out_real)

        _, _, real_nsig_by_target, selected_count_by_target, selected_rank = load_real_rank(
            real_path=real_path,
            fdr_threshold=float(args.fdr_threshold),
            target_overlap=float(args.target_overlap),
        )
        real_signs = load_real_signs(real_path)
        rows_written, selected_seen = write_oracle_pred_de(
            pred_path=pred_path,
            out_path=out_pred,
            selected_rank=selected_rank,
            real_signs=real_signs,
            max_score=float(args.max_score),
            rank_step=float(args.rank_step),
        )
        total_rows_written += rows_written
        total_selected_seen += selected_seen

        rows = score_context(out_real, out_pred, float(args.fdr_threshold))
        result_path = args.out_dir / f"{context}_results.csv"
        agg_path = args.out_dir / f"{context}_agg_results.csv"
        write_results(result_path, rows)
        write_agg_results(agg_path, rows)
        for row in rows:
            row["context"] = context
            row["oracle_selected"] = selected_count_by_target.get(str(row["perturbation"]), 0)
            all_rows.append(row)
        context_summary = summarize(rows)
        context_summary["context"] = context
        context_summary["n_real_targets"] = len(real_nsig_by_target)
        context_rows.append(context_summary)

    summary = summarize(all_rows)
    per_group_csv = args.out_dir / "tahoe100m_de_overlap_oracle_per_group.csv"
    context_csv = args.out_dir / "tahoe100m_de_overlap_oracle_by_context.csv"
    summary_json = args.out_dir / "tahoe100m_de_overlap_oracle_summary.json"
    write_csv(
        per_group_csv,
        all_rows,
        [
            "context",
            "perturbation",
            "real_nsig",
            "pred_nsig",
            "oracle_selected",
            "overlap_at_N",
            "overlap_at_50",
            "overlap_at_100",
            "overlap_at_200",
            "overlap_at_500",
            "precision_at_N",
            "precision_at_50",
            "precision_at_100",
            "precision_at_200",
            "precision_at_500",
        ],
    )
    write_csv(
        context_csv,
        context_rows,
        [
            "context",
            "n_groups",
            "n_real_targets",
            "overlap_at_N_mean",
            "overlap_at_N_median",
            "overlap_at_N_real_nsig_weighted",
            "real_nsig_mean",
        ],
    )
    payload = {
        "metric_scope": "oracle/leakage DE CSV rewrite for cell-eval overlap_at_N mechanics",
        "warning": "uses held-out real DE rank; not train-only and not paper-ready",
        "de_dir": str(args.de_dir),
        "out_dir": str(args.out_dir),
        "target_overlap": float(args.target_overlap),
        "fdr_threshold": float(args.fdr_threshold),
        "total_de_rows_written": total_rows_written,
        "total_oracle_selected_seen_in_pred": total_selected_seen,
        "summary": summary,
        "per_group_csv": str(per_group_csv),
        "context_csv": str(context_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
