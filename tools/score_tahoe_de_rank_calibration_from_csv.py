#!/usr/bin/env python3
"""Score DE-rank calibration heuristics from existing Tahoe cell-eval CSVs.

This is a lightweight proxy sweep that does not read h5ad files.  It uses
``*_pred_de.csv`` to build predicted-only gene bias statistics, then re-ranks
predicted significant genes and scores against the real top-N DE genes.

The sweep is diagnostic: it tests whether simple post-hoc ranking calibration
has enough headroom to improve ``overlap_at_N`` before investing in
materializing a new prediction h5ad and running full cell-eval.
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
    / "de_rank_calibration_proxy_20260625"
)


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


def parse_grid(text: str) -> list[float]:
    values = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("grid must not be empty")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def finite_mean(values: list[float]) -> float:
    xs = [value for value in values if math.isfinite(value)]
    return sum(xs) / len(xs) if xs else float("nan")


def load_groups(
    de_dir: Path,
    fdr_threshold: float,
) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], int], dict[tuple[str, str], list[tuple[str, float]]]]:
    real_sets: dict[tuple[str, str], set[str]] = {}
    real_ns: dict[tuple[str, str], int] = {}
    pred_groups: dict[tuple[str, str], list[tuple[str, float]]] = {}

    real_paths = sorted(de_dir.glob("*_real_de.csv"))
    if not real_paths:
        raise FileNotFoundError(f"No *_real_de.csv files found under {de_dir}")

    for real_path in real_paths:
        context = real_path.name[: -len("_real_de.csv")]
        by_target: dict[str, list[tuple[str, float]]] = {}
        with real_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fdr = parse_float(str(row.get("fdr", "")))
                if math.isfinite(fdr) and fdr < fdr_threshold:
                    by_target.setdefault(str(row["target"]), []).append(
                        (str(row["feature"]), abs_log2_fold_change(str(row["fold_change"])))
                    )
        for target, values in by_target.items():
            values.sort(key=lambda item: item[1], reverse=True)
            key = (context, target)
            real_sets[key] = {gene for gene, _ in values}
            real_ns[key] = len(values)

    for pred_path in sorted(de_dir.glob("*_pred_de.csv")):
        context = pred_path.name[: -len("_pred_de.csv")]
        by_target: dict[str, list[tuple[str, float]]] = {}
        with pred_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (context, str(row["target"]))
                if key not in real_ns:
                    continue
                fdr = parse_float(str(row.get("fdr", "")))
                if math.isfinite(fdr) and fdr < fdr_threshold:
                    by_target.setdefault(str(row["target"]), []).append(
                        (str(row["feature"]), abs_log2_fold_change(str(row["fold_change"])))
                    )
        for target, values in by_target.items():
            pred_groups[(context, target)] = values

    return real_sets, real_ns, pred_groups


def build_gene_stats(
    keys: list[tuple[str, str]],
    real_ns: dict[tuple[str, str], int],
    pred_groups: dict[tuple[str, str], list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    group_count = float(len(keys))
    gene_sum: dict[str, float] = {}
    gene_sumsq: dict[str, float] = {}
    gene_count: dict[str, int] = {}
    top500_count: dict[str, int] = {}
    top1000_count: dict[str, int] = {}
    topn_count: dict[str, int] = {}

    for key in keys:
        values = pred_groups[key]
        ranked = sorted(values, key=lambda item: item[1], reverse=True)
        n = real_ns[key]
        for gene, score in values:
            gene_sum[gene] = gene_sum.get(gene, 0.0) + score
            gene_sumsq[gene] = gene_sumsq.get(gene, 0.0) + score * score
            gene_count[gene] = gene_count.get(gene, 0) + 1
        for gene, _ in ranked[: min(500, len(ranked))]:
            top500_count[gene] = top500_count.get(gene, 0) + 1
        for gene, _ in ranked[: min(1000, len(ranked))]:
            top1000_count[gene] = top1000_count.get(gene, 0) + 1
        for gene, _ in ranked[: min(n, len(ranked))]:
            topn_count[gene] = topn_count.get(gene, 0) + 1

    stats: dict[str, dict[str, float]] = {}
    for gene, total in gene_sum.items():
        count = float(gene_count[gene])
        mean = total / count
        variance = max(0.0, gene_sumsq[gene] / count - mean * mean)
        stats[gene] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "top500_freq": top500_count.get(gene, 0) / group_count,
            "top1000_freq": top1000_count.get(gene, 0) / group_count,
            "topN_freq": topn_count.get(gene, 0) / group_count,
        }
    return stats


def calibrated_score(
    gene: str,
    score: float,
    method: str,
    param: float,
    stats: dict[str, dict[str, float]],
    freq_eps: float,
) -> float:
    gene_stats = stats[gene]
    if method == "raw":
        return score
    if method == "mean_div":
        return score / ((gene_stats["mean"] + 1e-8) ** param)
    if method == "mean_sub":
        return score - param * gene_stats["mean"]
    if method == "z":
        return (score - gene_stats["mean"]) / (gene_stats["std"] + param)
    if method == "top500_div":
        return score / ((gene_stats["top500_freq"] + freq_eps) ** param)
    if method == "top1000_div":
        return score / ((gene_stats["top1000_freq"] + freq_eps) ** param)
    if method == "topN_div":
        return score / ((gene_stats["topN_freq"] + freq_eps) ** param)
    raise ValueError(f"Unknown method: {method}")


def score_method(
    *,
    method: str,
    param: float,
    keys: list[tuple[str, str]],
    real_sets: dict[tuple[str, str], set[str]],
    real_ns: dict[tuple[str, str], int],
    pred_groups: dict[tuple[str, str], list[tuple[str, float]]],
    stats: dict[str, dict[str, float]],
    freq_eps: float,
) -> dict[str, Any]:
    overlaps: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for key in keys:
        n = real_ns[key]
        ranked = [
            (gene, calibrated_score(gene, score, method, param, stats, freq_eps))
            for gene, score in pred_groups[key]
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        pred_top = {gene for gene, _ in ranked[: min(n, len(ranked))]}
        overlap = len(real_sets[key] & pred_top) / float(n)
        overlaps.append(overlap)
        weighted_num += overlap * n
        weighted_den += n
    return {
        "method": method,
        "param": param,
        "n_groups": len(keys),
        "proxy_overlap_at_N_mean": finite_mean(overlaps),
        "proxy_overlap_at_N_median": statistics.median(overlaps) if overlaps else float("nan"),
        "proxy_overlap_at_N_real_nsig_weighted": weighted_num / weighted_den if weighted_den else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--beta-grid", default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10")
    parser.add_argument("--sub-grid", default="0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--z-grid", default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--freq-eps", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_sets, real_ns, pred_groups = load_groups(args.de_dir, float(args.fdr_threshold))
    keys = sorted(set(real_sets) & set(real_ns) & set(pred_groups))
    if not keys:
        raise RuntimeError("No scorable DE groups")
    stats = build_gene_stats(keys, real_ns, pred_groups)

    rows: list[dict[str, Any]] = []
    rows.append(
        score_method(
            method="raw",
            param=0.0,
            keys=keys,
            real_sets=real_sets,
            real_ns=real_ns,
            pred_groups=pred_groups,
            stats=stats,
            freq_eps=float(args.freq_eps),
        )
    )
    for method, grid in [
        ("mean_div", parse_grid(args.beta_grid)),
        ("mean_sub", parse_grid(args.sub_grid)),
        ("z", parse_grid(args.z_grid)),
        ("top500_div", parse_grid(args.beta_grid)),
        ("top1000_div", parse_grid(args.beta_grid)),
        ("topN_div", parse_grid(args.beta_grid)),
    ]:
        for param in grid:
            rows.append(
                score_method(
                    method=method,
                    param=param,
                    keys=keys,
                    real_sets=real_sets,
                    real_ns=real_ns,
                    pred_groups=pred_groups,
                    stats=stats,
                    freq_eps=float(args.freq_eps),
                )
            )

    best = max(rows, key=lambda row: float(row["proxy_overlap_at_N_mean"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "tahoe100m_de_rank_calibration_proxy_summary.csv"
    selection_json = args.out_dir / "tahoe100m_de_rank_calibration_proxy_selection.json"
    fields = [
        "method",
        "param",
        "n_groups",
        "proxy_overlap_at_N_mean",
        "proxy_overlap_at_N_median",
        "proxy_overlap_at_N_real_nsig_weighted",
    ]
    write_csv(summary_csv, rows, fields)
    payload = {
        "metric_scope": "diagnostic pred-only DE rank calibration proxy for cell-eval overlap_at_N",
        "de_dir": str(args.de_dir),
        "fdr_threshold": float(args.fdr_threshold),
        "freq_eps": float(args.freq_eps),
        "num_scored_groups": len(keys),
        "summary_csv": str(summary_csv),
        "best_by_proxy_overlap_at_N": best,
        "summaries": rows,
    }
    selection_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_by_proxy_overlap_at_N": best, "summary_csv": str(summary_csv)}, indent=2))


if __name__ == "__main__":
    main()
