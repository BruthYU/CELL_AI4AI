#!/usr/bin/env python3
"""Audit protected Replogle eval outputs after submit_evaluate_replogle_rjob.sh.

This script is read-only. It assumes eval was run through
benchmark/evaluate_replogle.py and checks the files that script writes under:

    benchmark/workspace/{step_dir}/{out_subfolder}/{cell_line}/agg_results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_RPE1_STEP_DIR = "20260615_105743_JIT_LLM_REPLOGLE_V3_STATEALIGN"
DEFAULT_SKIP_METRICS = ("pearson_edistance", "clustering_agreement")


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_agg(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            statistic = str(row.pop("statistic"))
            rows[statistic] = {key: parse_float(value) for key, value in row.items()}
    return rows


def latest_eval_log(step_path: Path, cell_line: str) -> Path | None:
    pattern = str(step_path / "logs" / f"evaluate_replogle_rjob_{cell_line}_*.log")
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def log_has_skip_metrics(path: Path | None, skip_metrics: tuple[str, ...]) -> bool | None:
    if path is None:
        return None
    text = path.read_text(errors="replace")
    return all(metric in text for metric in skip_metrics)


def resolve_step_dir(cell_line: str, explicit_step_dir: str, prefix: str) -> str:
    if explicit_step_dir:
        return explicit_step_dir
    if cell_line == "rpe1":
        return DEFAULT_RPE1_STEP_DIR
    return f"{prefix}{cell_line}"


def audit_cell_line(
    *,
    workspace: Path,
    cell_line: str,
    step_dir: str,
    out_subfolder: str,
    skip_metrics: tuple[str, ...],
) -> tuple[dict[str, object], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    step_path = workspace / step_dir
    agg_path = step_path / out_subfolder / cell_line / "agg_results.csv"
    if not agg_path.exists():
        return (
            {
                "cell_line": cell_line,
                "step_dir": step_dir,
                "agg_results": str(agg_path),
                "exists": False,
            },
            [f"missing agg_results for {cell_line}: {agg_path}"],
            warnings,
        )

    rows = load_agg(agg_path)
    mean = rows.get("mean", {})
    null_count = rows.get("null_count", {})
    count = rows.get("count", {})
    required = ["pearson_delta", "mse", "mae", "mse_delta", "mae_delta"]
    for metric in required:
        if metric not in mean:
            failures.append(f"{cell_line}: missing metric {metric!r} in {agg_path}")

    pearson_delta = float(mean.get("pearson_delta", float("nan")))
    if not math.isfinite(pearson_delta):
        failures.append(f"{cell_line}: pearson_delta mean is not finite in {agg_path}")

    pearson_nulls = float(null_count.get("pearson_delta", 0.0))
    if pearson_nulls > 0:
        failures.append(f"{cell_line}: pearson_delta null_count={pearson_nulls:g}")

    overlap_metrics = [metric for metric in ("overlap_at_N", "overlap_at_50", "overlap_at_100") if metric in mean]
    overlap_means = {metric: float(mean[metric]) for metric in overlap_metrics}
    if overlap_metrics and all(value == 0.0 for value in overlap_means.values()):
        warnings.append(
            f"{cell_line}: overlap metrics are all zero; h5ad may still be structurally valid, "
            "but DE ranking should be inspected"
        )

    de_metrics = {
        metric: float(mean[metric])
        for metric in ("de_spearman_sig", "de_direction_match", "de_sig_genes_recall")
        if metric in mean
    }

    log_path = latest_eval_log(step_path, cell_line)
    skip_ok = log_has_skip_metrics(log_path, skip_metrics)
    if skip_ok is False:
        failures.append(f"{cell_line}: latest eval log does not show skip metrics {skip_metrics}: {log_path}")
    elif skip_ok is None:
        warnings.append(f"{cell_line}: no eval log found under {step_path / 'logs'}")

    summary = {
        "cell_line": cell_line,
        "step_dir": step_dir,
        "agg_results": str(agg_path),
        "exists": True,
        "num_results": int(count.get("pearson_delta", 0.0)) if math.isfinite(count.get("pearson_delta", float("nan"))) else None,
        "pearson_delta": pearson_delta,
        "pearson_delta_null_count": pearson_nulls,
        "overlap_means": overlap_means,
        "de_means": de_metrics,
        "latest_eval_log": None if log_path is None else str(log_path),
        "skip_metrics_found_in_log": skip_ok,
    }
    return summary, failures, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--cell-lines", nargs="+", default=["hepg2", "jurkat", "k562"])
    parser.add_argument("--step-dir", default="")
    parser.add_argument("--step-dir-prefix", default="replogle_inference_")
    parser.add_argument("--out-subfolder", default="results_calibrate")
    parser.add_argument("--expected-mean", type=float, default=0.75)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument(
        "--skip-metrics",
        default=",".join(DEFAULT_SKIP_METRICS),
        help="Comma-separated metrics that should have been skipped by evaluate_replogle.py.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--require-positive-overlap", action="store_true")
    args = parser.parse_args()

    skip_metrics = tuple(metric.strip() for metric in args.skip_metrics.split(",") if metric.strip())
    summaries = []
    failures: list[str] = []
    warnings: list[str] = []

    for raw_cell_line in args.cell_lines:
        cell_line = str(raw_cell_line).lower()
        step_dir = resolve_step_dir(cell_line, args.step_dir, args.step_dir_prefix)
        summary, cell_failures, cell_warnings = audit_cell_line(
            workspace=args.workspace,
            cell_line=cell_line,
            step_dir=step_dir,
            out_subfolder=args.out_subfolder,
            skip_metrics=skip_metrics,
        )
        summaries.append(summary)
        failures.extend(cell_failures)
        warnings.extend(cell_warnings)

    pearsons = [float(row["pearson_delta"]) for row in summaries if row.get("exists") and math.isfinite(float(row["pearson_delta"]))]
    macro = float(sum(pearsons) / len(pearsons)) if pearsons else float("nan")
    lower = float(args.expected_mean - args.tolerance)
    upper = float(args.expected_mean + args.tolerance)
    if not math.isfinite(macro):
        failures.append("macro pearson_delta is not finite")
    elif not (lower <= macro <= upper):
        failures.append(f"macro pearson_delta={macro:.6f} outside {args.expected_mean:.3f} +/- {args.tolerance:.3f}")

    for row in summaries:
        if not row.get("exists"):
            continue
        pearson = float(row["pearson_delta"])
        if math.isfinite(pearson) and not (lower <= pearson <= upper):
            failures.append(
                f"{row['cell_line']}: pearson_delta={pearson:.6f} outside "
                f"{args.expected_mean:.3f} +/- {args.tolerance:.3f}"
            )
        if args.require_positive_overlap:
            overlap_means = row.get("overlap_means", {})
            if isinstance(overlap_means, dict) and overlap_means and all(float(value) <= 0.0 for value in overlap_means.values()):
                failures.append(f"{row['cell_line']}: overlap metrics are all non-positive")

    payload = {
        "workspace": str(args.workspace),
        "cell_lines": summaries,
        "macro_pearson_delta": macro,
        "expected_mean": float(args.expected_mean),
        "tolerance": float(args.tolerance),
        "skip_metrics": list(skip_metrics),
        "warnings": warnings,
        "failures": failures,
    }

    print("Replogle eval audit:")
    for row in summaries:
        if not row.get("exists"):
            print(f"{row['cell_line']:6s} missing agg_results")
            continue
        overlap = row.get("overlap_means", {})
        overlap_text = ",".join(f"{key}={value:.4f}" for key, value in overlap.items()) if isinstance(overlap, dict) else ""
        print(
            f"{row['cell_line']:6s} pearson_delta={float(row['pearson_delta']):.6f} "
            f"n={row['num_results']} skip_log={row['skip_metrics_found_in_log']} {overlap_text}"
        )
    print(f"macro_pearson_delta={macro:.6f}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote summary: {args.summary_json}")

    if failures:
        print("FAILED eval audit:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASSED eval audit")


if __name__ == "__main__":
    main()
