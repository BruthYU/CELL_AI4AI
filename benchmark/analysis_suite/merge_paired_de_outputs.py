#!/usr/bin/env python3
"""Merge external-baseline paired DE outputs with SCALE+PDC paired DE rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import generate_paired_de as paired
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TABLES = [
    "paired_de_top100_events.csv",
    "paired_real_de_top100.csv",
    "paired_pred_de_top100.csv",
    "paired_de_target_metrics.csv",
]
MODEL_ORDER = ["SCALE+PDC", "CPA", "BioLord", "State", "GEARS", "scLambda"]
MODEL_COLORS = {
    "SCALE+PDC": "#C25759",
    "CPA": "#599CB4",
    "BioLord": "#5B8C5A",
    "State": "#6C8EBF",
    "GEARS": "#B696B6",
    "scLambda": "#8C6D31",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--scale-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scale-label", default="SCALE+PDC")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def read_table(root: Path, name: str) -> list[dict[str, Any]]:
    return paired.iter_csv(root / "tables" / name)


def as_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def short_split_class(value: str) -> str:
    return (
        value.replace("kang_task05_lopo_patients", "Kang LOPO")
        .replace("norman_task07_heldout_combo_genes", "Norman")
        .replace("zscape_seen_cell_unseen_gene", "ZSCAPE")
    )


def write_metric_box_jitter(
    path: Path,
    target_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
) -> Path | None:
    df = pd.DataFrame(target_rows)
    if df.empty or metric not in df.columns:
        return None
    df["_metric"] = df[metric].apply(as_float)
    df = df.dropna(subset=["_metric"]).copy()
    if df.empty:
        return None
    split_classes = [x for x in ["kang_task05_lopo_patients", "norman_task07_heldout_combo_genes", "zscape_seen_cell_unseen_gene"] if x in set(df["split_class"])]
    split_classes.extend(sorted(set(df["split_class"]) - set(split_classes)))
    models = [model for model in MODEL_ORDER if model in set(df["model_label"])]
    models.extend(sorted(set(df["model_label"]) - set(models)))
    fig, ax = plt.subplots(figsize=(max(9.2, len(split_classes) * 2.4), 5.6), dpi=160)
    rng = np.random.default_rng(31)
    group_width = 0.78
    box_width = group_width / max(1, len(models)) * 0.70
    tick_positions = np.arange(len(split_classes), dtype=float)
    legend_seen = set()
    for class_idx, split_class in enumerate(split_classes):
        base = tick_positions[class_idx]
        offsets = np.linspace(-group_width / 2, group_width / 2, len(models))
        for offset, model in zip(offsets, models):
            vals = df[(df["split_class"] == split_class) & (df["model_label"] == model)]["_metric"].tolist()
            if not vals:
                continue
            x = base + float(offset)
            bp = ax.boxplot(
                [vals],
                positions=[x],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.2},
                whiskerprops={"color": "#333333", "linewidth": 0.9},
                capprops={"color": "#333333", "linewidth": 0.9},
            )
            color = MODEL_COLORS.get(model, "#777777")
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.30 if model != "SCALE+PDC" else 0.44)
            bp["boxes"][0].set_edgecolor("#333333")
            jitter = rng.uniform(-box_width * 0.38, box_width * 0.38, size=len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=13,
                color=color,
                alpha=0.58,
                edgecolors="white",
                linewidths=0.25,
                label=model if model not in legend_seen else None,
                zorder=4,
            )
            legend_seen.add(model)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([short_split_class(x) for x in split_classes], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncols=min(3, len(models)))
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    external_dir = Path(args.external_dir)
    scale_dir = Path(args.scale_dir)
    outdir = Path(args.outdir)
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    merged_tables: dict[str, list[dict[str, Any]]] = {}
    for name in TABLES:
        external_rows = [row for row in read_table(external_dir, name) if row.get("model_label") != args.scale_label]
        scale_rows = [row for row in read_table(scale_dir, name) if row.get("model_label") == args.scale_label]
        rows = external_rows + scale_rows
        merged_tables[name] = rows
        paired.write_csv(tables / name, rows)

    failure_rows = []
    for root in (external_dir, scale_dir):
        path = root / "tables" / "paired_de_failures.csv"
        if path.exists():
            failure_rows.extend(paired.iter_csv(path))
    paired.write_csv(tables / "paired_de_failures.csv", failure_rows)

    all_events = merged_tables["paired_de_top100_events.csv"]
    all_targets = merged_tables["paired_de_target_metrics.csv"]
    summaries = paired.summarize_targets(all_targets)
    model_summaries = paired.summarize_targets_by_keys(
        all_targets,
        ["dataset_id", "benchmark", "split_class", "split", "model_label", "seed"],
    )
    paired.write_csv(tables / "paired_de_metric_summary.csv", summaries)
    paired.write_csv(tables / "paired_de_model_metric_summary.csv", model_summaries)

    figure_paths: list[Path] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_events:
        by_split[row["split_class"]].append(row)
    for split_class, rows in by_split.items():
        fig = figures / f"paired_de_{split_class}_top100_real_vs_pred_log2fc.svg"
        paired.write_svg_scatter(fig, rows, f"{split_class}: paired top100 real vs predicted log2FC")
        if fig.exists():
            figure_paths.append(fig)
        rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_model[str(row.get("model_label", "model"))].append(row)
        for model_name, model_rows in sorted(rows_by_model.items()):
            model_slug = paired.slugify(model_name)
            model_fig = figures / f"paired_de_{split_class}__{model_slug}_top100_real_vs_pred_log2fc.svg"
            paired.write_svg_scatter(model_fig, model_rows, f"{split_class} {model_name}: top100 real vs predicted log2FC")
            if model_fig.exists():
                figure_paths.append(model_fig)

    metric_specs = [
        (
            "pearson_logfc_topk",
            "paired_de_metric_box_jitter_pearson_logfc_topk.png",
            "Paired DE top100 logFC Pearson by split class and model",
            "Top100 logFC Pearson",
        ),
        (
            "direction_agreement_topk",
            "paired_de_metric_box_jitter_direction_agreement_topk.png",
            "Paired DE top100 direction agreement by split class and model",
            "Direction agreement@100",
        ),
        (
            "de_overlap_at_k",
            "paired_de_metric_box_jitter_de_overlap_at_k.png",
            "Paired DE overlap@100 by split class and model",
            "DE overlap@100",
        ),
    ]
    for metric, filename, title, ylabel in metric_specs:
        fig = figures / filename
        result = write_metric_box_jitter(fig, all_targets, metric, title, ylabel)
        if result is not None and result.exists():
            figure_paths.append(result)

    config = {
        "analysis_prompt_path": ".codex/skills/perturbation-benchmark-analysis/references/full_analysis_angles.md",
        "merged_external_dir": str(external_dir),
        "merged_scale_dir": str(scale_dir),
    }
    paired.write_report(outdir, config, model_summaries, summaries, figure_paths)
    manifest = {
        "external_dir": str(external_dir),
        "scale_dir": str(scale_dir),
        "n_events": len(all_events),
        "n_targets": len(all_targets),
        "n_summaries": len(summaries),
        "n_model_summaries": len(model_summaries),
        "n_failures": len(failure_rows),
        "top_k": args.top_k,
        "n_figures": len(figure_paths),
        "scale_label": args.scale_label,
        "logfc_mean_policy": paired.LOGFC_MEAN_POLICY,
    }
    (outdir / "paired_de_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote merged paired DE outputs to {outdir}")


if __name__ == "__main__":
    main()
