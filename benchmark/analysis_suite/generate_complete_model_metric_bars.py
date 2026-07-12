#!/usr/bin/env python3
"""Generate complete per-model bar charts for the final figure book.

The charts are keyed by dataset, split_id, and metric_protocol so the report
does not mix incompatible score definitions or collapse models into a summary
average. This script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/"
    "vcbench_comparison_20260611/tables/paper_split_model_comparison_20260616.csv"
)
DEFAULT_FIGURE_BOOK = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/"
    "benchmark/workspace/final_paper_ready_figure_book_20260629"
)
REPORT_NAME = "scale_pdc_vs_external_baselines_full_report.md"

PRIMARY_BASELINES = {"CPA", "BioLord", "State", "GEARS", "scLambda"}
VCBENCH_MODELS = {
    "BioLord",
    "CPA",
    "DecoderOnly",
    "GEARS",
    "GenePert",
    "LatentAdditive",
    "LinearAdditive",
    "PRNet",
    "SAMS-VAE",
    "State",
    "scLambda",
    "original public Nemo-CellFlow",
    "train-only single-delta additive",
}
DATASET_ORDER = [
    "Kang Task05",
    "ComboSciPlex3 Task03",
    "ComboSciPlex3 Task04",
    "Norman Task07",
    "ZSCAPE develop",
    "Replogle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--figure-book", type=Path, default=DEFAULT_FIGURE_BOOK)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_FIGURE_BOOK / REPORT_NAME)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def slugify(text: str, limit: int = 150) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text or "chart")[:limit].strip("-")


def short_text(value: str | None, limit: int = 90) -> str:
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_scale(row: dict[str, str]) -> bool:
    return row.get("reported_model", "").startswith("SCALE")


def model_family(model: str) -> str:
    if model.startswith("SCALE"):
        return "SCALE/SCALE+"
    if model in PRIMARY_BASELINES:
        return "approved external baseline"
    if model in VCBENCH_MODELS:
        return "other VCBench model"
    return "diagnostic/reference"


def model_color(model: str) -> str:
    family = model_family(model)
    if family == "SCALE/SCALE+":
        return "#1f77b4"
    if family == "approved external baseline":
        return "#d95f02"
    if family == "other VCBench model":
        return "#7b61b3"
    return "#6b7280"


def group_sort_key(item: tuple[tuple[str, str, str], list[dict[str, str]]]) -> tuple[int, str, str, str]:
    key, _ = item
    dataset, split_id, metric_protocol = key
    dataset_rank = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
    return dataset_rank, dataset, split_id, metric_protocol


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("paper_table") != "main":
            continue
        if as_float(row.get("value")) is None:
            continue
        key = (row["dataset"], row["split_id"], row["metric_protocol"])
        groups[key].append(row)
    return groups


def replogle_cell_line_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != "Replogle":
            continue
        if not row.get("split_id", "").startswith("replogle_cell_line_"):
            continue
        if row.get("metric_protocol") != "Replogle direct pearson_delta":
            continue
        if as_float(row.get("value")) is None:
            continue
        key = (row["dataset"], row["split_id"], row["metric_protocol"])
        groups[key].append(row)
    return groups


def is_complete_comparison(rows: list[dict[str, str]]) -> bool:
    return any(is_scale(row) for row in rows) and any(not is_scale(row) for row in rows) and len(rows) >= 2


def has_primary_external(rows: list[dict[str, str]]) -> bool:
    return any(row.get("reported_model") in PRIMARY_BASELINES for row in rows)


def sorted_chart_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    higher = rows[0].get("higher_is_better", "TRUE").upper() == "TRUE"
    return sorted(rows, key=lambda row: as_float(row.get("value")) or 0.0, reverse=higher)


def wrap_svg_text(text: str, width: int) -> list[str]:
    parts: list[str] = []
    for piece in text.split(" "):
        if len(piece) > width:
            parts.extend(textwrap.wrap(piece, width=width, break_long_words=True))
        else:
            parts.append(piece)
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return lines[:5]


def nice_ticks(vmin: float, vmax: float, n: int = 6) -> list[float]:
    if vmin == vmax:
        pad = max(abs(vmin) * 0.1, 0.05)
        vmin, vmax = vmin - pad, vmax + pad
    span = vmax - vmin
    raw_step = span / max(n - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1:
        step = magnitude
    elif residual <= 2:
        step = 2 * magnitude
    elif residual <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def label_for_row(row: dict[str, str], duplicate_models: set[str]) -> str:
    label = row["reported_model"]
    calibration = row.get("calibration", "")
    if row["reported_model"] in duplicate_models and calibration:
        label = f"{label} ({calibration})"
    if "diagnostic_only" in row.get("claim_status", ""):
        label = f"{label} [diagnostic]"
    return label


def draw_svg_chart(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = sorted_chart_rows(rows)
    values = [as_float(row["value"]) or 0.0 for row in rows]
    raw_min = min(values + [0.0])
    raw_max = max(values + [0.0])
    pad = (raw_max - raw_min) * 0.08 if raw_max != raw_min else max(abs(raw_max) * 0.1, 0.05)
    ymin = min(0.0, raw_min - pad)
    ymax = max(0.0, raw_max + pad)
    ticks = nice_ticks(ymin, ymax)
    ymin, ymax = min(ticks), max(ticks)

    n = len(rows)
    width = max(980, 92 * n + 240)
    height = 690
    left, right = 100, 38
    top, bottom = 112, 230
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 12
    bar_w = max(28, min(64, (plot_w - gap * (n + 1)) / max(n, 1)))
    zero_y = top + (ymax - 0.0) / (ymax - ymin) * plot_h

    def y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    title = f"{rows[0]['dataset']} | {rows[0].get('split_label') or rows[0]['split_id']}"
    subtitle = f"{rows[0]['metric_protocol']} ({'; '.join(sorted(set(r['metric'] for r in rows)))})"
    model_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        model_counts[row["reported_model"]] += 1
    duplicate_models = {model for model, count in model_counts.items() if count > 1}

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{left}" y="35" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111111">{html.escape(title)}</text>')
    svg.append(f'<text x="{left}" y="63" font-family="Arial, sans-serif" font-size="13" fill="#444444">{html.escape(subtitle)}</text>')
    svg.append(f'<text x="{left}" y="86" font-family="Arial, sans-serif" font-size="12" fill="#666666">Each bar is one model/configuration row from the audited table; no cross-model mean is used.</text>')
    for tick in ticks:
        ty = y(tick)
        svg.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{width-right}" y2="{ty:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{left-10}" y="{ty+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555555">{tick:.3g}</text>')
    svg.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#777777" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333333" stroke-width="1"/>')
    svg.append(f'<text x="25" y="{top + plot_h/2}" transform="rotate(-90 25,{top + plot_h/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333333">metric value</text>')

    for idx, row in enumerate(rows):
        value = as_float(row["value"]) or 0.0
        x = left + gap + idx * (bar_w + gap)
        top_y = min(y(value), zero_y)
        bar_h = max(abs(zero_y - y(value)), 1.5)
        svg.append(f'<rect x="{x:.2f}" y="{top_y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{model_color(row["reported_model"])}" rx="2"/>')
        value_y = top_y - 7 if value >= 0 else top_y + bar_h + 14
        svg.append(f'<text x="{x + bar_w/2:.2f}" y="{value_y:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111111">{value:.3f}</text>')
        lines = wrap_svg_text(label_for_row(row, duplicate_models), 18)
        base_y = top + plot_h + 24
        svg.append(f'<text x="{x + bar_w/2:.2f}" y="{base_y:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#222222">')
        for line_idx, line in enumerate(lines):
            dy = 0 if line_idx == 0 else 12
            svg.append(f'<tspan x="{x + bar_w/2:.2f}" dy="{dy}">{html.escape(line)}</tspan>')
        svg.append("</text>")

    legend_x = left
    legend_y = height - 54
    legend = [
        ("SCALE/SCALE+", "#1f77b4"),
        ("approved external baseline", "#d95f02"),
        ("other VCBench model", "#7b61b3"),
        ("diagnostic/reference", "#6b7280"),
    ]
    for label, color in legend:
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 18}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#333333">{html.escape(label)}</text>')
        legend_x += 225
    svg.append("</svg>")
    out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def row_to_output(row: dict[str, str], figure: Path, group_type: str) -> dict[str, str]:
    out = dict(row)
    out["figure"] = str(figure)
    out["model_family"] = model_family(row["reported_model"])
    out["group_type"] = group_type
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| Rank | Model | Family | Calibration | Value | n | Claim status | Source |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(sorted_chart_rows(rows), start=1):
        value = as_float(row.get("value")) or 0.0
        lines.append(
            "| "
            f"{rank} | {row['reported_model']} | {model_family(row['reported_model'])} | "
            f"{short_text(row.get('calibration'), 60)} | {value:.4f} | "
            f"{short_text(row.get('n'), 20)} | {short_text(row.get('claim_status'), 50)} | "
            f"{short_text(row.get('source_path'), 70)} |"
        )
    return lines


def interpretation(rows: list[dict[str, str]]) -> str:
    ordered = sorted_chart_rows(rows)
    best = ordered[0]
    scale_rows = [row for row in ordered if is_scale(row)]
    best_scale = scale_rows[0] if scale_rows else None
    primary_rows = [row for row in ordered if row.get("reported_model") in PRIMARY_BASELINES]
    best_primary = primary_rows[0] if primary_rows else None
    best_text = f"Best row is {best['reported_model']} ({as_float(best.get('value')):.4f})."
    if best_scale is not None and best_primary is not None:
        margin = (as_float(best_scale.get("value")) or 0.0) - (as_float(best_primary.get("value")) or 0.0)
        return (
            f"{best_text} Best SCALE/SCALE+ row is {best_scale['reported_model']} "
            f"({as_float(best_scale.get('value')):.4f}); best approved external baseline is "
            f"{best_primary['reported_model']} ({as_float(best_primary.get('value')):.4f}); "
            f"margin is {margin:.4f}."
        )
    if best_scale is not None:
        return (
            f"{best_text} This protocol has no approved external-baseline row, so it is shown as a diagnostic comparison rather than a primary external-baseline claim."
        )
    return best_text


def old_section(text: str, start: str, end: str | None = None) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    if end is None:
        return text[start_idx:].strip()
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def write_report(
    figure_book: Path,
    source: Path,
    source_report: Path,
    plotted_groups: list[tuple[tuple[str, str, str], list[dict[str, str]], Path, str]],
    diagnostic_groups: list[tuple[tuple[str, str, str], list[dict[str, str]], Path, str]],
) -> None:
    old_text = source_report.read_text(encoding="utf-8") if source_report.exists() else ""
    xinjie_section = old_section(old_text, "## Replogle Full-Cell SCALE+PDC: Xinjie Figure Set", "## Kang Task05 LOPO patient splits")
    case_section = old_section(old_text, "## Biological Case Studies", "## Provenance")

    primary_groups = [item for item in plotted_groups if has_primary_external(item[1]) and item[0][0] != "Replogle"]
    non_primary_groups = [item for item in plotted_groups if item not in primary_groups]

    lines: list[str] = [
        "# Consolidated Paper-Ready Figure Book",
        "",
        "This is the single user-facing report. The main comparison evidence is now complete per-model bar charts: one chart and one table for each comparable dataset/split/metric protocol.",
        "",
        "## Comparison Evidence Rule",
        "",
        f"Audited source table: `{source}`.",
        "",
        "A chart is only treated as a direct comparison when all bars share the same `dataset`, `split_id`, and `metric_protocol`. Rows are per model or per configuration; no cross-model score average is used as the main result.",
        "",
        "Primary external baselines are CPA, BioLord, State, GEARS, and scLambda. Other VCBench models already present in the audited row set are also shown in the same bar chart so the comparison is complete.",
        "",
        "## Analysis Coverage",
        "",
        "| Analysis angle | Main object | Report location | Interpretation rule |",
        "|---|---|---|---|",
        "| Per-split model comparison | deltaPCC / pearson_delta rows | Complete bar-chart sections below | Compare models only within the same split and metric protocol. |",
        "| External-baseline availability | approved baseline presence | Availability table and missing-group CSV | Missing CPA/BioLord/State/GEARS/scLambda rows are reported explicitly. |",
        "| Replogle full-cell diagnostics | cell-line deltaPCC / DEG behavior | Replogle diagnostic sections | Diagnostic only unless approved external baseline artifacts exist. |",
        "| DE/logFC and biological mechanisms | paired DE, marker response, response PCA proxy | Later diagnostic sections | Supports mechanism interpretation, not a replacement for the bar-chart comparison. |",
        "",
        "## Complete Per-Model Bar-Chart Evidence",
        "",
    ]

    current_dataset = ""
    for key, rows, figure, _group_type in primary_groups:
        dataset, split_id, metric_protocol = key
        if dataset != current_dataset:
            current_dataset = dataset
            lines.append(f"### {dataset}")
            lines.append("")
        rel = figure.relative_to(figure_book)
        split_label = rows[0].get("split_label") or split_id
        lines.append(f"#### {split_label}")
        lines.append("")
        lines.append(f"Metric protocol: `{metric_protocol}`.")
        lines.append("")
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
        lines.append(f"Interpretation: {interpretation(rows)}")
        lines.append("")
        lines.extend(markdown_table(rows))
        lines.append("")

    lines.extend(
        [
            "## Diagnostic Complete Charts",
            "",
            "These charts still contain multiple model/configuration rows, but they do not have an approved external-baseline row in the same metric protocol. They are retained for transparency and should not be phrased as primary external-baseline wins.",
            "",
        ]
    )
    for key, rows, figure, _group_type in non_primary_groups:
        dataset, split_id, metric_protocol = key
        rel = figure.relative_to(figure_book)
        split_label = rows[0].get("split_label") or split_id
        lines.append(f"### {dataset}: {split_label}")
        lines.append("")
        lines.append(f"Metric protocol: `{metric_protocol}`.")
        lines.append("")
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
        lines.append(f"Interpretation: {interpretation(rows)}")
        lines.append("")
        lines.extend(markdown_table(rows))
        lines.append("")

    lines.extend(
        [
            "## Replogle Cell-Line Diagnostic Comparison",
            "",
            "Replogle currently lacks CPA/BioLord/State/GEARS/scLambda rows under the same cell-line metric protocol. The four cell-line charts below therefore compare available Replogle diagnostic/model-family rows only.",
            "",
        ]
    )
    for key, rows, figure, _group_type in diagnostic_groups:
        _dataset, split_id, metric_protocol = key
        rel = figure.relative_to(figure_book)
        split_label = rows[0].get("split_label") or split_id
        lines.append(f"### {split_label}")
        lines.append("")
        lines.append(f"Metric protocol: `{metric_protocol}`.")
        lines.append("")
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
        lines.append(f"Interpretation: {interpretation(rows)}")
        lines.append("")
        lines.extend(markdown_table(rows))
        lines.append("")

    lines.extend(
        [
            "## Output Tables",
            "",
            "- `tables/complete_model_metric_bars/complete_model_metric_bar_rows.csv`: rows used by all complete bar charts.",
            "- `tables/complete_model_metric_bars/groups_without_complete_bars.csv`: groups not plotted because they lack a SCALE/SCALE+ row, lack a non-SCALE row, or would mix incompatible protocols.",
            "- `tables/complete_model_metric_bars/external_baseline_availability.csv`: approved-baseline availability per plotted group.",
            "",
            "## Paired-DE And Transport Diagnostics",
            "",
            "The following figures are retained from the previous figure book as biological and mechanism diagnostics. They do not replace the complete per-model bar charts above.",
            "",
            "### Kang paired DE all models",
            "",
            "![paired_de_kang_task05_lopo_patients_top100_real_vs_pred_log2fc](figures/paired_de_kang_task05_lopo_patients_top100_real_vs_pred_log2fc.svg)",
            "",
            "Model-specific paired DE scatter:",
            "",
            "- [BioLord](figures/paired_de_kang_task05_lopo_patients__biolord_top100_real_vs_pred_log2fc.svg)",
            "- [CPA](figures/paired_de_kang_task05_lopo_patients__cpa_top100_real_vs_pred_log2fc.svg)",
            "- [GEARS](figures/paired_de_kang_task05_lopo_patients__gears_top100_real_vs_pred_log2fc.svg)",
            "- [State](figures/paired_de_kang_task05_lopo_patients__state_top100_real_vs_pred_log2fc.svg)",
            "- [scLambda](figures/paired_de_kang_task05_lopo_patients__sclambda_top100_real_vs_pred_log2fc.svg)",
            "- [SCALE+PDC](figures/paired_de_kang_task05_lopo_patients__scale_pdc_top100_real_vs_pred_log2fc.svg)",
            "",
            "### Kang expression-space PCA proxy",
            "",
            "![expression_pca_proxy_kang_patient101_cd4t_all_models](figures/expression_pca_proxy_kang_patient101_cd4t_all_models.svg)",
            "",
            "### ZSCAPE paired DE all models",
            "",
            "![paired_de_zscape_seen_cell_unseen_gene_top100_real_vs_pred_log2fc](figures/paired_de_zscape_seen_cell_unseen_gene_top100_real_vs_pred_log2fc.svg)",
            "",
            "Model-specific paired DE scatter:",
            "",
            "- [BioLord](figures/paired_de_zscape_seen_cell_unseen_gene__biolord_top100_real_vs_pred_log2fc.svg)",
            "- [CPA](figures/paired_de_zscape_seen_cell_unseen_gene__cpa_top100_real_vs_pred_log2fc.svg)",
            "- [State](figures/paired_de_zscape_seen_cell_unseen_gene__state_top100_real_vs_pred_log2fc.svg)",
            "- [scLambda](figures/paired_de_zscape_seen_cell_unseen_gene__sclambda_top100_real_vs_pred_log2fc.svg)",
            "- [SCALE+PDC](figures/paired_de_zscape_seen_cell_unseen_gene__scale_pdc_top100_real_vs_pred_log2fc.svg)",
            "",
        ]
    )
    if xinjie_section:
        lines.append(xinjie_section)
        lines.append("")
    if case_section:
        lines.append(case_section)
        lines.append("")
    lines.extend(
        [
            "## Provenance",
            "",
            "This consolidated report, the generated SVGs, and the row-level CSV files in `tables/complete_model_metric_bars/` are the paper-facing comparison artifacts.",
            "",
        ]
    )
    text = "\n".join(lines)
    (figure_book / REPORT_NAME).write_text(text, encoding="utf-8")
    (figure_book / "report.md").write_text(text, encoding="utf-8")


def write_outputs(source: Path, figure_book: Path, source_report: Path) -> None:
    figure_dir = figure_book / "figures" / "complete_model_metric_bars"
    table_dir = figure_book / "tables" / "complete_model_metric_bars"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    groups = group_rows(rows)
    replogle_groups = replogle_cell_line_groups(rows)

    plotted: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []
    availability: list[dict[str, str]] = []
    plotted_groups: list[tuple[tuple[str, str, str], list[dict[str, str]], Path, str]] = []
    diagnostic_groups: list[tuple[tuple[str, str, str], list[dict[str, str]], Path, str]] = []

    for key, group in sorted(groups.items(), key=group_sort_key):
        models = sorted(set(row["reported_model"] for row in group))
        if is_complete_comparison(group):
            group_type = "diagnostic_complete" if key[0] == "Replogle" or not has_primary_external(group) else "primary_complete"
            svg_path = figure_dir / f"{slugify('__'.join(key))}.svg"
            draw_svg_chart(group, svg_path)
            plotted_groups.append((key, group, svg_path, group_type))
            for row in group:
                plotted.append(row_to_output(row, svg_path, group_type))
            present = sorted(set(model for model in models if model in PRIMARY_BASELINES))
            availability.append(
                {
                    "dataset": key[0],
                    "split_id": key[1],
                    "metric_protocol": key[2],
                    "present_approved_external_baselines": "; ".join(present),
                    "missing_approved_external_baselines": "; ".join(sorted(PRIMARY_BASELINES - set(present))),
                    "has_scale_row": "yes",
                    "chart_type": group_type,
                }
            )
        else:
            has_scale = any(model.startswith("SCALE") for model in models)
            has_other = any(not model.startswith("SCALE") for model in models)
            unavailable.append(
                {
                    "dataset": key[0],
                    "split_id": key[1],
                    "metric_protocol": key[2],
                    "models": "; ".join(models),
                    "reason": "missing SCALE/SCALE+ row" if not has_scale else "missing non-SCALE row" if not has_other else "not enough rows",
                }
            )

    for key, group in sorted(replogle_groups.items(), key=group_sort_key):
        if len(group) < 2:
            continue
        svg_path = figure_dir / f"{slugify('__'.join(key))}__diagnostic.svg"
        draw_svg_chart(group, svg_path)
        diagnostic_groups.append((key, group, svg_path, "replogle_cell_line_diagnostic"))
        for row in group:
            plotted.append(row_to_output(row, svg_path, "replogle_cell_line_diagnostic"))
        availability.append(
            {
                "dataset": key[0],
                "split_id": key[1],
                "metric_protocol": key[2],
                "present_approved_external_baselines": "",
                "missing_approved_external_baselines": "; ".join(sorted(PRIMARY_BASELINES)),
                "has_scale_row": "yes" if any(is_scale(row) for row in group) else "no",
                "chart_type": "replogle_cell_line_diagnostic",
            }
        )

    write_csv(table_dir / "complete_model_metric_bar_rows.csv", plotted)
    write_csv(
        table_dir / "groups_without_complete_bars.csv",
        unavailable,
        ["dataset", "split_id", "metric_protocol", "models", "reason"],
    )
    write_csv(
        table_dir / "external_baseline_availability.csv",
        availability,
        [
            "dataset",
            "split_id",
            "metric_protocol",
            "present_approved_external_baselines",
            "missing_approved_external_baselines",
            "has_scale_row",
            "chart_type",
        ],
    )
    manifest = {
        "source": str(source),
        "figure_book": str(figure_book),
        "grouping": ["dataset", "split_id", "metric_protocol"],
        "n_source_rows": len(rows),
        "n_complete_chart_groups": len(plotted_groups),
        "n_replogle_cell_line_diagnostic_groups": len(diagnostic_groups),
        "n_plotted_rows": len(plotted),
        "n_groups_without_complete_bars": len(unavailable),
    }
    (table_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(figure_book, source, source_report, plotted_groups, diagnostic_groups)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    write_outputs(args.source, args.figure_book, args.source_report)


if __name__ == "__main__":
    main()
