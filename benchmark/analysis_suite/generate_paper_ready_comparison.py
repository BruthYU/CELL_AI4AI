#!/usr/bin/env python3
"""Generate paper-ready per-split SCALE+PDC comparisons.

This parser handles the audited paper-ready table where SCALE+PDC rows and
external-baseline rows may appear on different protocol rows for the same
split. Protocol and claim-status fields are preserved in every output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINES = ["CPA", "BioLord", "State", "GEARS", "scLambda"]
CLASS_ORDER = [
    "kang_task05_lopo_patients",
    "kang_task05_full_lopo_aggregate",
    "norman_task07_heldout_combo_genes",
    "task03_seen_drug_seen_cell_unseen_cells",
    "task03_unseen_cell_lines",
    "task03_unseen_dose_extrapolation",
    "task04_seen_single_unseen_combo",
    "zscape_seen_cell_unseen_gene",
    "other",
]
SPLIT_CLASS_LABELS = {
    "kang_task05_lopo_patients": "Kang Task05 LOPO patient splits",
    "kang_task05_full_lopo_aggregate": "Kang Task05 full LOPO aggregate",
    "norman_task07_heldout_combo_genes": "Norman Task07 held-out combo genes",
    "task03_seen_drug_seen_cell_unseen_cells": "ComboSciPlex3 Task03 seen drug/cell, unseen cells",
    "task03_unseen_cell_lines": "ComboSciPlex3 Task03 unseen cell lines",
    "task03_unseen_dose_extrapolation": "ComboSciPlex3 Task03 unseen dose extrapolation",
    "task04_seen_single_unseen_combo": "ComboSciPlex3 Task04 seen singles, unseen combo",
    "zscape_seen_cell_unseen_gene": "ZSCAPE seen cell, unseen gene",
    "other": "Other splits",
}
MODEL_COLORS = {
    "SCALE+PDC": "#C25759",
    "CPA": "#599CB4",
    "BioLord": "#5B8C5A",
    "State": "#6C8EBF",
    "GEARS": "#B696B6",
    "scLambda": "#8C6D31",
}
MODEL_ORDER = ["SCALE+PDC", *BASELINES]


def short_text(value: Any, max_len: int = 130) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-wide",
        default="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/vcbench_comparison_20260611/tables/paper_split_model_comparison_wide_20260616.csv",
    )
    parser.add_argument(
        "--kang-pdc",
        default="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/vcbench_comparison_20260611/tables/kang_baseline_pdc_results_20260625.csv",
    )
    parser.add_argument(
        "--focus-md",
        default="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/vcbench_comparison_20260611/paper_split_model_comparison_scale_pdc_focus_20260625.md",
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    figures = outdir / "figures"
    tables = outdir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def split_class(dataset: str, split_id: str) -> str:
    if dataset == "Kang Task05":
        return "kang_task05_lopo_patients" if "patient" in split_id else "kang_task05_full_lopo_aggregate"
    if dataset == "Norman Task07":
        return "norman_task07_heldout_combo_genes"
    if dataset == "ZSCAPE develop":
        return "zscape_seen_cell_unseen_gene"
    if dataset == "ComboSciPlex3 Task03":
        if "dose" in split_id:
            return "task03_unseen_dose_extrapolation"
        if "A549" in split_id or "K562" in split_id or "MCF7" in split_id or "unseen_cell_line" in split_id:
            return "task03_unseen_cell_lines"
        return "task03_seen_drug_seen_cell_unseen_cells"
    if dataset == "ComboSciPlex3 Task04":
        return "task04_seen_single_unseen_combo"
    return "other"


def external_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ext = df[df["metric_protocol"].astype(str).str.contains("VCBench external pearson_delta|Published/existing aggregate primary metric|Matched held-out-group", regex=True, na=False)]
    for _, row in ext.iterrows():
        dataset = row["dataset"]
        split_id = row["split_id"]
        for model in BASELINES:
            val = as_float(row.get(model))
            if val is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "split_label": row.get("split_label", ""),
                    "split_kind": row.get("split_kind", ""),
                    "split_class": split_class(dataset, split_id),
                    "model": model,
                    "role": "external_non_scale_baseline",
                    "value": val,
                    "metric_protocol": row.get("metric_protocol", ""),
                    "claim_status": row.get("claim_statuses", ""),
                    "paper_table": row.get("paper_table", ""),
                    "source_row_best_model": row.get("best_model", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def scale_pdc_rows(df: pd.DataFrame, kang_pdc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pdc_mask = df["metric_protocol"].astype(str).str.contains("PDC", na=False)
    for _, row in df[pdc_mask].iterrows():
        dataset = row["dataset"]
        split_id = row["split_id"]
        if dataset not in {"Kang Task05", "Norman Task07", "ZSCAPE develop", "ComboSciPlex3 Task03", "ComboSciPlex3 Task04"}:
            continue
        val = as_float(row.get("best_value"))
        if val is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "split_id": split_id,
                "split_label": row.get("split_label", ""),
                "split_kind": row.get("split_kind", ""),
                "split_class": split_class(dataset, split_id),
                "model": "SCALE+PDC",
                "role": "primary_scale_pdc",
                "value": val,
                "metric_protocol": row.get("metric_protocol", ""),
                "claim_status": row.get("claim_statuses", ""),
                "paper_table": row.get("paper_table", ""),
                "selected_lambda": row.get("selected_lambda", ""),
                "raw_value": row.get("raw_value", ""),
                "delta_to_raw": row.get("delta_to_raw", ""),
                "source_row_best_model": row.get("best_model", ""),
                "notes": row.get("notes", ""),
            }
        )
    if not kang_pdc.empty:
        for _, row in kang_pdc.iterrows():
            split_id = str(row.get("split_id", ""))
            if "aggregate" in split_id:
                mapped = "kang_full_lopo_aggregate"
                label = "full LOPO aggregate audited 56-fold"
            else:
                mapped = split_id.replace("kang_baseline_pdc_test_patient_", "kang_lopo_test_patient_")
                label = str(row.get("split_label", "")).replace("historical baseline ", "")
            rows.append(
                {
                    "dataset": "Kang Task05",
                    "split_id": mapped,
                    "split_label": label,
                    "split_kind": row.get("split_kind", "test"),
                    "split_class": split_class("Kang Task05", mapped),
                    "model": "SCALE+PDC",
                    "role": "primary_scale_pdc_audited_kang",
                    "value": as_float(row.get("metric_value")),
                    "metric_protocol": row.get("metric_protocol", ""),
                    "claim_status": row.get("claim_status", ""),
                    "paper_table": "audited_kang_pdc",
                    "selected_lambda": "",
                    "raw_value": "",
                    "delta_to_raw": "",
                    "source_row_best_model": row.get("method", ""),
                    "notes": row.get("notes", ""),
                }
            )
    out = pd.DataFrame(rows).dropna(subset=["value"])
    if out.empty:
        return out

    def priority(row: pd.Series) -> int:
        claim = str(row.get("claim_status", ""))
        if row.get("paper_table") == "audited_kang_pdc":
            return 0
        if "diagnostic_only" not in claim:
            return 1
        return 2

    out["_scale_priority"] = out.apply(priority, axis=1)
    out = out.sort_values(["dataset", "split_id", "_scale_priority", "value"], ascending=[True, True, True, False])
    out = out.drop_duplicates(["dataset", "split_id"], keep="first").drop(columns=["_scale_priority"])
    return out.reset_index(drop=True)


def best_external(external: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if external.empty:
        return pd.DataFrame()
    idx = external.groupby(keys)["value"].idxmax()
    out = external.loc[idx].copy()
    out["best_external_split_id"] = out["split_id"]
    out["best_external_split_class"] = out["split_class"]
    return out.rename(
        columns={
            "model": "best_external_model",
            "value": "best_external_value",
            "metric_protocol": "best_external_protocol",
            "claim_status": "best_external_claim_status",
        }
    )


def comparison_table(scale: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    best_direct = best_external(external, ["dataset", "split_id"])
    best_class = best_external(external, ["dataset", "split_class"])
    if scale.empty or best_direct.empty:
        return pd.DataFrame()
    scale_rows = scale.reset_index(drop=True).copy()
    scale_rows["_scale_row_id"] = np.arange(len(scale_rows))
    keep_direct = [
        "dataset",
        "split_id",
        "best_external_model",
        "best_external_value",
        "best_external_protocol",
        "best_external_claim_status",
        "best_external_split_id",
        "best_external_split_class",
    ]
    direct = scale_rows.merge(best_direct[keep_direct], on=["dataset", "split_id"], how="left")
    direct = direct.dropna(subset=["best_external_value"])
    direct["comparison_scope"] = "same_split_id"
    matched_ids = set(direct["_scale_row_id"].tolist())

    missing = scale_rows[~scale_rows["_scale_row_id"].isin(matched_ids)].copy()
    if not missing.empty and not best_class.empty:
        keep_class = [
            "dataset",
            "split_class",
            "best_external_model",
            "best_external_value",
            "best_external_protocol",
            "best_external_claim_status",
            "best_external_split_id",
            "best_external_split_class",
        ]
        fallback = missing.merge(best_class[keep_class], on=["dataset", "split_class"], how="left")
        fallback = fallback.dropna(subset=["best_external_value"])
        fallback["comparison_scope"] = "split_class_fallback"
        merged = pd.concat([direct, fallback], ignore_index=True)
    else:
        merged = direct
    merged = merged.drop(columns=["_scale_row_id"], errors="ignore")
    merged["margin_vs_best_external"] = merged["value"] - merged["best_external_value"]
    merged["beats_best_external"] = merged["margin_vs_best_external"] > 0
    return merged.sort_values(["dataset", "split_id", "role"])


def model_order(models: list[str]) -> list[str]:
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(sorted(model for model in models if model not in set(ordered)))
    return ordered


def finite_values(values: pd.Series) -> list[float]:
    parsed = [as_float(x) for x in values]
    return [float(x) for x in parsed if x is not None]


def add_jittered_points(
    ax: plt.Axes,
    x_center: float,
    values: list[float],
    color: str,
    rng: np.random.Generator,
) -> None:
    if not values:
        return
    jitter = rng.uniform(-0.09, 0.09, size=len(values)) if len(values) > 1 else np.zeros(len(values))
    ax.scatter(
        np.full(len(values), x_center) + jitter,
        values,
        s=34,
        color=color,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )


def write_box_jitter_caption(rows: pd.DataFrame, split_class_name: str) -> str:
    model_means = []
    for model, vals in rows.groupby("model"):
        values = finite_values(vals["value"])
        if values:
            model_means.append((model, float(np.mean(values)), len(values)))
    if not model_means:
        return "This grouped box plot has no finite values to compare."
    best = max(model_means, key=lambda item: item[1])
    scale_rows = [item for item in model_means if item[0].startswith("SCALE+PDC")]
    scale_text = ""
    if scale_rows:
        best_scale = max(scale_rows, key=lambda item: item[1])
        scale_text = f" SCALE+PDC mean is {best_scale[1]:.4f} across {best_scale[2]} rows."
    return (
        "Box shows the score distribution across available rows in this split class; jittered points are individual split/protocol rows. "
        f"The highest mean model here is {best[0]} ({best[1]:.4f}).{scale_text}"
    )


def plot_split_rank(rows: pd.DataFrame, dataset: str, split_class_name: str, figures: Path) -> tuple[Path, str] | None:
    sub = rows[(rows["dataset"] == dataset) & (rows["split_class"] == split_class_name)].copy()
    if sub.empty:
        return None
    models = model_order(list(dict.fromkeys(sub["model"].tolist())))
    fig, ax = plt.subplots(figsize=(max(7.2, len(models) * 0.85), 5.3), dpi=160)
    rng = np.random.default_rng(17)
    positions = np.arange(len(models), dtype=float)
    box_values: list[list[float]] = []
    for model in models:
        model_rows = sub[sub["model"] == model]
        box_values.append(finite_values(model_rows["value"]))
    bp = ax.boxplot(
        box_values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
    )
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model, "#777777"))
        patch.set_alpha(0.30 if "SCALE" not in model else 0.42)
        patch.set_edgecolor("#333333")
    for pos, model in zip(positions, models):
        model_rows = sub[sub["model"] == model]
        values = finite_values(model_rows["value"])
        add_jittered_points(ax, float(pos), values, MODEL_COLORS.get(model, "#666666"), rng)
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("deltaPCC / mean_delta_pearson")
    ax.set_title(f"{dataset}: grouped model comparison")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    out = figures / f"paper_ready_{split_class_name}_model_value_box_jitter.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out, write_box_jitter_caption(sub, split_class_name)


def margin_caption(sub: pd.DataFrame) -> str:
    values = finite_values(sub["margin_vs_best_external"])
    if not values:
        return "This margin plot has no finite comparable rows."
    n_pos = sum(1 for value in values if value > 0)
    return (
        "Box shows the distribution of SCALE+PDC minus the best non-scale baseline within each split class; "
        f"jittered points are comparable rows. Positive values favor SCALE+PDC ({n_pos}/{len(values)} rows are positive)."
    )


def plot_margin(comp: pd.DataFrame, figures: Path) -> tuple[Path, str] | None:
    sub = comp[comp["model"].isin(["SCALE+PDC"])].copy()
    if sub.empty:
        return None
    ordered_classes = [cls for cls in CLASS_ORDER if cls in set(sub["split_class"])]
    ordered_classes.extend(sorted(set(sub["split_class"]) - set(ordered_classes)))
    fig, ax = plt.subplots(figsize=(max(8, len(ordered_classes) * 1.05), 5.2), dpi=160)
    positions = np.arange(len(ordered_classes), dtype=float)
    values_by_class = [finite_values(sub[sub["split_class"] == cls]["margin_vs_best_external"]) for cls in ordered_classes]
    bp = ax.boxplot(
        values_by_class,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#C25759")
        patch.set_alpha(0.32)
        patch.set_edgecolor("#333333")
    rng = np.random.default_rng(23)
    for pos, cls in zip(positions, ordered_classes):
        class_rows = sub[sub["split_class"] == cls]
        values = finite_values(class_rows["margin_vs_best_external"])
        colors = ["#5B8C5A" if value > 0 else "#C25759" for value in values]
        jitter = rng.uniform(-0.10, 0.10, size=len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            c=colors,
            s=36,
            alpha=0.84,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([SPLIT_CLASS_LABELS.get(cls, cls) for cls in ordered_classes], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("SCALE+PDC - best non-scale baseline")
    ax.set_title("Paper-ready SCALE+PDC margin by split class")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    out = figures / "paper_ready_scale_pdc_margin_vs_best_external_box_jitter.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out, margin_caption(sub)


def write_report(
    outdir: Path,
    scale: pd.DataFrame,
    external: pd.DataFrame,
    all_rows: pd.DataFrame,
    comp: pd.DataFrame,
    figs_by_class: dict[str, tuple[Path, str]],
    margin_fig: tuple[Path, str] | None,
) -> None:
    lines = [
        "# Paper-Ready Per-Split Model Comparison",
        "",
        "This report uses `paper_split_model_comparison_wide_20260616.csv` as the audited source. It preserves protocol and claim-status labels because score-only endpoints, audited Kang PDC rows, validation-selected PDC, and diagnostic lambda sweeps are not interchangeable.",
        "",
        "When a split lacks h5ad-backed paired-DE artifacts for SCALE+PDC, this report still recovers the paper-ready SCALE+PDC score rows from the audited global comparison tables. The main sources are `paper_split_model_comparison_wide_20260616.csv`, `pdc_kang_public_endpoint_20260624.csv`, `pdc_val_selected_results_20260624.csv`, and `kang_baseline_pdc_results_20260625.csv`; paired-DE availability is handled separately and must not be confused with absence of a score result.",
        "",
        "Primary comparisons use only non-scale external baselines: CPA, BioLord, State, GEARS, and scLambda. SCALE-only, lambda sweeps, prior LLM, train delta-memory, and other SCALE-family diagnostics are intentionally excluded from the main baseline set.",
        "",
        "## Overall SCALE+PDC Rows",
        "",
        "| Dataset | Split | Model | Value | Protocol | Claim status | Source table | Raw | Delta to raw |",
        "|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for _, row in scale.sort_values(["dataset", "split_id", "role"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['split_id']} | {row['model']} | {row['value']:.4f} | {short_text(row.get('metric_protocol',''))} | {short_text(row.get('claim_status',''))} | {short_text(row.get('paper_table',''))} | {row.get('raw_value','')} | {row.get('delta_to_raw','')} |"
        )
    lines += [
        "",
        "## Split Sections",
        "",
    ]

    present_classes = [c for c in CLASS_ORDER if c in set(all_rows["split_class"])]
    present_classes.extend(sorted(set(all_rows["split_class"]) - set(present_classes)))
    for split_cls in present_classes:
        rows = all_rows[all_rows["split_class"] == split_cls].copy()
        if rows.empty:
            continue
        title = SPLIT_CLASS_LABELS.get(split_cls, split_cls)
        lines.append(f"### {title}")
        lines.append("")

        comp_rows = comp[comp["split_class"] == split_cls].copy()
        if comp_rows.empty:
            lines.append("No paired SCALE+PDC vs best non-scale baseline row could be formed from the audited table for this split class.")
            lines.append("")
        else:
            n_win = int((comp_rows["margin_vs_best_external"] > 0).sum())
            n_total = int(len(comp_rows))
            scopes = ", ".join(sorted(set(comp_rows["comparison_scope"].astype(str))))
            mean_margin = float(comp_rows["margin_vs_best_external"].mean())
            lines.append(
                f"Interpretation: SCALE+PDC beats the best non-scale baseline in {n_win}/{n_total} comparable rows; mean margin is {mean_margin:.4f}. Comparison scope: {scopes}."
            )
            lines.append("")
            lines.append("| Split | SCALE+PDC row | Value | Best non-scale | Best split | Value | Margin | Scope | Protocol | Claim status |")
            lines.append("|---|---|---:|---|---|---:|---:|---|---|---|")
            for _, row in comp_rows.sort_values(["split_id", "role"]).iterrows():
                lines.append(
                    f"| {row['split_id']} | {row['model']} | {row['value']:.4f} | {row['best_external_model']} | {row.get('best_external_split_id','')} | {row['best_external_value']:.4f} | {row['margin_vs_best_external']:.4f} | {row.get('comparison_scope','')} | {short_text(row.get('metric_protocol',''))} | {short_text(row.get('claim_status',''))} |"
                )
            lines.append("")

        scale_rows = rows[rows["model"].astype(str).str.contains("SCALE\\+PDC", regex=True)].copy()
        if not scale_rows.empty:
            lines.append("SCALE+PDC source audit:")
            lines.append("")
            lines.append("| Split | Model | Source table | Selected lambda | Raw | Delta to raw | Source row | Notes |")
            lines.append("|---|---|---|---:|---:|---:|---|---|")
            for _, row in scale_rows.sort_values(["split_id", "role", "model"]).iterrows():
                lines.append(
                    f"| {row['split_id']} | {row['model']} | {short_text(row.get('paper_table',''), 80)} | {row.get('selected_lambda','')} | {row.get('raw_value','')} | {row.get('delta_to_raw','')} | {short_text(row.get('source_row_best_model',''), 90)} | {short_text(row.get('notes',''), 180)} |"
                )
            lines.append("")

        lines.append("Detailed model rows:")
        lines.append("")
        lines.append("| Dataset | Split | Model | Role | Value | Protocol | Claim status |")
        lines.append("|---|---|---|---|---:|---|---|")
        for _, row in rows.sort_values(["split_id", "role", "model"]).iterrows():
            lines.append(
                f"| {row['dataset']} | {row['split_id']} | {row['model']} | {row['role']} | {row['value']:.4f} | {short_text(row.get('metric_protocol',''))} | {short_text(row.get('claim_status',''))} |"
            )
        lines.append("")

        fig_info = figs_by_class.get(split_cls)
        if fig_info is not None:
            fig, caption = fig_info
            rel = fig.relative_to(outdir)
            lines.append("Figure: grouped box plot with jittered/strip scatter overlay. Higher is better for these deltaPCC/mean-delta-Pearson rows.")
            lines.append("")
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
            lines.append(f"Brief interpretation: {caption}")
            lines.append("")

    if margin_fig is not None:
        lines += ["## Margin Summary", ""]
        margin_path, caption = margin_fig
        rel = margin_path.relative_to(outdir)
        lines.append(f"![{margin_path.stem}]({rel})")
        lines.append("")
        lines.append(f"Brief interpretation: {caption}")
        lines.append("")

    lines += [
        "## Output Tables",
        "",
        "- `tables/paper_ready_scale_pdc_rows.csv`: audited SCALE+PDC rows, including audited Kang PDC rows.",
        "- `tables/paper_ready_external_baseline_rows.csv`: CPA/BioLord/State/GEARS/scLambda rows.",
        "- `tables/paper_ready_all_model_rows_long.csv`: long table used for every split section.",
        "- `tables/paper_ready_scale_pdc_vs_best_external.csv`: best non-scale baseline comparison and margin.",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    figures, tables = ensure_dirs(outdir)
    wide = pd.read_csv(args.paper_wide)
    kang_pdc = pd.read_csv(args.kang_pdc) if Path(args.kang_pdc).exists() else pd.DataFrame()
    scale = scale_pdc_rows(wide, kang_pdc)
    external = external_rows(wide)
    all_rows = pd.concat([scale, external], ignore_index=True)
    comp = comparison_table(scale, external)
    scale.to_csv(tables / "paper_ready_scale_pdc_rows.csv", index=False)
    external.to_csv(tables / "paper_ready_external_baseline_rows.csv", index=False)
    all_rows.to_csv(tables / "paper_ready_all_model_rows_long.csv", index=False)
    comp.to_csv(tables / "paper_ready_scale_pdc_vs_best_external.csv", index=False)

    figs_by_class: dict[str, tuple[Path, str]] = {}
    present_classes = [c for c in CLASS_ORDER if c in set(all_rows["split_class"])]
    present_classes.extend(sorted(set(all_rows["split_class"]) - set(present_classes)))
    for split_name in present_classes:
        sub = all_rows[all_rows["split_class"] == split_name]
        if sub.empty:
            continue
        dataset = str(sub["dataset"].iloc[0])
        fig = plot_split_rank(all_rows, dataset, split_name, figures)
        if fig:
            figs_by_class[split_name] = fig
    margin_fig = plot_margin(comp, figures)
    if margin_fig:
        n_figures = len(figs_by_class) + 1
    else:
        n_figures = len(figs_by_class)
    (outdir / "run_manifest.json").write_text(
        json.dumps(
            {
                "paper_wide": args.paper_wide,
                "focus_md": args.focus_md,
                "kang_pdc": args.kang_pdc,
                "n_scale_rows": int(len(scale)),
                "n_external_rows": int(len(external)),
                "n_comparison_rows": int(len(comp)),
                "n_figures": int(n_figures),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(outdir, scale, external, all_rows, comp, figs_by_class, margin_fig)
    print(f"Wrote paper-ready comparison to {outdir}")


if __name__ == "__main__":
    main()
