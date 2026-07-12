#!/usr/bin/env python3
"""Generate reusable perturbation benchmark analyses.

The suite intentionally uses only the Python standard library. It consumes the
CSV/JSON artifacts already produced by benchmark jobs and emits tidy tables,
SVG figures, a run manifest, and a compact interpretation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


EPS = 1e-9
TOP_KS = (100, 200, 500)
CORE_METRICS = (
    "pearson_delta",
    "mse_delta",
    "mae_delta",
    "precision_at_100",
    "precision_at_200",
    "precision_at_500",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "pr_auc",
    "roc_auc",
)
DEFAULT_EXTERNAL_BASELINES = ("CPA", "BioLord", "State", "GEARS", "scLambda")
EXTERNAL_MODEL_LABELS = {
    "cpa": "CPA",
    "biolord": "BioLord",
    "state": "State",
    "gears": "GEARS",
    "sclambda": "scLambda",
}
BASELINE_DETAIL_METRICS = (
    "pearson_delta",
    "pearson_average",
    "rmse_average",
    "cosine_logfc",
    "spearman_logfc",
    "deg_precision",
    "deg_recall",
    "deg_iou",
)
MATCHED_DETAIL_METRICS = (
    "deltaPCC_mean",
    "pearson_average_mean",
    "rmse_average_mean",
    "per_gene_std_mean",
    "row_rmse_to_gene_mean",
)
MATCHED_GROUP_METRICS = (
    "deltaPCC",
    "pearson_average",
    "rmse_average",
    "per_gene_std",
    "row_rmse_to_gene",
)
PALETTE = [
    "#599CB4",
    "#C25759",
    "#B2D3A4",
    "#F6C8A8",
    "#B696B6",
    "#6C8EBF",
    "#8C6D31",
    "#5B8C5A",
]


@dataclass
class CellEvalDataset:
    config: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    de_pairs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AnalysisOutputs:
    outdir: Path
    tables: Path
    figures: Path
    report_lines: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config file.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="Skip real/pred DE gene-level analysis even when DE files exist.",
    )
    parser.add_argument(
        "--top-de-gene-limit",
        type=int,
        default=30,
        help="Number of frequent genes to show in gene-level DE heatmaps.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON. This dependency-free v1 accepts JSON configs."
        ) from exc


def ensure_outputs(outdir: Path) -> AnalysisOutputs:
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return AnalysisOutputs(outdir=outdir, tables=tables, figures=figures)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    parsed = as_float(value)
    return default if parsed is None else parsed


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(values: Iterable[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def quantile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def stdev(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def fmt(value: Any, digits: int = 4) -> str:
    parsed = as_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def metric_label(metric: str) -> str:
    labels = {
        "pearson_delta": "deltaPCC",
        "rmse_delta": "RMSE(delta)",
        "mse_delta": "MSE(delta)",
        "mae_delta": "MAE(delta)",
        "precision_at_100": "DE precision@100",
        "precision_at_200": "DE precision@200",
        "precision_at_500": "DE precision@500",
        "de_direction_match": "DE direction match",
        "mean_log2_amp_ratio": "mean log2 amplitude ratio",
        "external_pearson_delta": "external deltaPCC",
        "matched_delta_pcc": "matched deltaPCC",
        "deltaPCC": "deltaPCC",
        "deltaPCC_mean": "deltaPCC mean",
        "pearson_average": "Pearson average",
        "pearson_average_mean": "Pearson average mean",
        "rmse_average": "RMSE average",
        "rmse_average_mean": "RMSE average mean",
        "cosine_logfc": "cosine(logFC)",
        "spearman_logfc": "Spearman(logFC)",
        "deg_precision": "DEG precision",
        "deg_recall": "DEG recall",
        "deg_iou": "DEG IoU",
        "per_gene_std": "per-gene std",
        "per_gene_std_mean": "per-gene std mean",
        "row_rmse_to_gene": "row RMSE to gene mean",
        "row_rmse_to_gene_mean": "row RMSE to gene mean",
    }
    return labels.get(metric, metric)


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "dataset"


def md_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").replace("|", "\\|")


def metric_direction(metric: str) -> str:
    if metric.startswith("rmse") or metric.startswith("mse") or metric.startswith("mae"):
        return "lower"
    if "rmse" in metric or metric in {"energy_distance", "sinkhorn_divergency"}:
        return "lower"
    return "higher"


def model_label(model: str, mapping: dict[str, str] | None = None) -> str:
    key = str(model).strip()
    lower = key.lower()
    if mapping and key in mapping:
        return mapping[key]
    if mapping and lower in mapping:
        return mapping[lower]
    return EXTERNAL_MODEL_LABELS.get(lower, key)


def allowed_value(value: str, allowed: Iterable[str]) -> bool:
    allowed_list = list(allowed)
    if not allowed_list:
        return True
    for item in allowed_list:
        if item.endswith("*") and value.startswith(item[:-1]):
            return True
        if value == item:
            return True
    return False


def split_class_for(benchmark: Any = "", dataset: Any = "", group: Any = "") -> str:
    text = f"{benchmark} {dataset} {group}".lower()
    if "replogle" in text:
        return "replogle_full_cell_lines"
    if "kang" in text or "lopo" in text:
        return "kang_task05_lopo_patients"
    if "norman" in text or "task07" in text:
        return "norman_task07_heldout_combo_genes"
    if "zscape" in text:
        return "zscape_external_splits"
    if "task04" in text or "seen_single_unseen_combo" in text:
        return "task04_seen_single_unseen_combo"
    if "seen_drug_seen_cell_unseen_cells" in text:
        return "task03_seen_drug_seen_cell_unseen_cells"
    if "unseen_cell_line" in text:
        return "task03_unseen_cell_lines"
    if "unseen_dose" in text:
        return "task03_unseen_dose_extrapolation"
    return slugify(f"{benchmark}_{group}" if benchmark or group else dataset)


def benchmark_label_from_dataset(dataset: str, mapping: dict[str, str] | None = None) -> str:
    if mapping and dataset in mapping:
        return mapping[dataset]
    if dataset.startswith("kang"):
        return "Kang Task05"
    if dataset.startswith("norman"):
        return "Norman Task07"
    if dataset.startswith("zscape"):
        return "ZSCAPE develop"
    return dataset


def svg_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .axis{stroke:#333;stroke-width:1} .grid{stroke:#e7e7e7;stroke-width:1}</style>',
                body,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def nice_range(values: list[float], lower_zero: bool = False) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    if lower_zero:
        lo = min(0.0, lo)
    if lo == hi:
        pad = 1.0 if hi == 0 else abs(hi) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def scale(value: float, lo: float, hi: float, start: float, stop: float, invert: bool = False) -> float:
    if hi == lo:
        t = 0.5
    else:
        t = (value - lo) / (hi - lo)
    if invert:
        t = 1.0 - t
    return start + t * (stop - start)


def bar_chart(path: Path, rows: list[dict[str, Any]], title: str, x_key: str, y_key: str, ylabel: str) -> None:
    rows = [row for row in rows if as_float(row.get(y_key)) is not None]
    if not rows:
        return
    width = max(640, 90 * len(rows) + 160)
    height = 430
    left, right, top, bottom = 70, 30, 45, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [safe_float(row[y_key]) for row in rows]
    y0, y1 = nice_range(values, lower_zero=True)
    bar_w = plot_w / len(rows) * 0.62
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    for i in range(5):
        val = y0 + (y1 - y0) * i / 4
        y = scale(val, y0, y1, top + plot_h, top, invert=False)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{fmt(val, 2)}</text>')
    for idx, row in enumerate(rows):
        cx = left + plot_w * (idx + 0.5) / len(rows)
        value = safe_float(row[y_key])
        y = scale(value, y0, y1, top + plot_h, top)
        h = top + plot_h - y
        color = PALETTE[idx % len(PALETTE)]
        body.append(f'<rect x="{cx-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" stroke="#333" stroke-width="0.5"/>')
        label = svg_escape(row.get(x_key, ""))
        body.append(f'<text x="{cx:.1f}" y="{height-55}" text-anchor="middle" font-size="10" transform="rotate(-30 {cx:.1f},{height-55})">{label}</text>')
        body.append(f'<text x="{cx:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="10">{fmt(value, 3)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    body.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18,{top+plot_h/2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>')
    write_svg(path, width, height, "\n".join(body))


def box_chart(path: Path, groups: dict[str, list[float]], title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    labels = [label for label, vals in groups.items() if vals]
    if not labels:
        return
    width = max(640, 95 * len(labels) + 160)
    height = 430
    left, right, top, bottom = 70, 30, 45, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for label in labels for v in groups[label]]
    y0, y1 = ylim if ylim else nice_range(values, lower_zero=True)
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    for i in range(5):
        val = y0 + (y1 - y0) * i / 4
        y = scale(val, y0, y1, top + plot_h, top)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{fmt(val, 2)}</text>')
    for idx, label in enumerate(labels):
        vals = groups[label]
        q1 = quantile(vals, 0.25)
        q2 = median(vals)
        q3 = quantile(vals, 0.75)
        lo = min(vals)
        hi = max(vals)
        if None in (q1, q2, q3):
            continue
        cx = left + plot_w * (idx + 0.5) / len(labels)
        box_w = min(52, plot_w / len(labels) * 0.55)
        y_q1 = scale(q1, y0, y1, top + plot_h, top)
        y_q2 = scale(q2, y0, y1, top + plot_h, top)
        y_q3 = scale(q3, y0, y1, top + plot_h, top)
        y_lo = scale(lo, y0, y1, top + plot_h, top)
        y_hi = scale(hi, y0, y1, top + plot_h, top)
        color = PALETTE[idx % len(PALETTE)]
        body.append(f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#333"/>')
        body.append(f'<line x1="{cx-box_w/4:.1f}" y1="{y_hi:.1f}" x2="{cx+box_w/4:.1f}" y2="{y_hi:.1f}" stroke="#333"/>')
        body.append(f'<line x1="{cx-box_w/4:.1f}" y1="{y_lo:.1f}" x2="{cx+box_w/4:.1f}" y2="{y_lo:.1f}" stroke="#333"/>')
        body.append(f'<rect x="{cx-box_w/2:.1f}" y="{y_q3:.1f}" width="{box_w:.1f}" height="{max(1, y_q1-y_q3):.1f}" fill="{color}" fill-opacity="0.72" stroke="#333"/>')
        body.append(f'<line x1="{cx-box_w/2:.1f}" y1="{y_q2:.1f}" x2="{cx+box_w/2:.1f}" y2="{y_q2:.1f}" stroke="#111" stroke-width="1.4"/>')
        body.append(f'<text x="{cx:.1f}" y="{height-38}" text-anchor="middle" font-size="11">{svg_escape(label)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    body.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18,{top+plot_h/2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>')
    write_svg(path, width, height, "\n".join(body))


def scatter_chart(path: Path, rows: list[dict[str, Any]], title: str, x_key: str, y_key: str, group_key: str, xlabel: str, ylabel: str) -> None:
    points = [(safe_float(r[x_key]), safe_float(r[y_key]), str(r.get(group_key, ""))) for r in rows if as_float(r.get(x_key)) is not None and as_float(r.get(y_key)) is not None]
    if not points:
        return
    width, height = 620, 460
    left, right, top, bottom = 70, 125, 45, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, x1 = nice_range([p[0] for p in points])
    y0, y1 = nice_range([p[1] for p in points], lower_zero=True)
    groups = sorted({p[2] for p in points})
    color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    for i in range(5):
        xv = x0 + (x1 - x0) * i / 4
        x = scale(xv, x0, x1, left, left + plot_w)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        body.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="10">{fmt(xv, 2)}</text>')
        yv = y0 + (y1 - y0) * i / 4
        y = scale(yv, y0, y1, top + plot_h, top)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{fmt(yv, 2)}</text>')
    for xv, yv, g in points:
        x = scale(xv, x0, x1, left, left + plot_w)
        y = scale(yv, y0, y1, top + plot_h, top)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color[g]}" fill-opacity="0.42"/>')
    for idx, g in enumerate(groups):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+plot_w+18}" y="{y-9}" width="10" height="10" fill="{color[g]}"/>')
        body.append(f'<text x="{left+plot_w+34}" y="{y}" font-size="11">{svg_escape(g)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    body.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-size="12">{svg_escape(xlabel)}</text>')
    body.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18,{top+plot_h/2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>')
    write_svg(path, width, height, "\n".join(body))


def line_chart(path: Path, rows: list[dict[str, Any]], title: str, x_key: str, y_key: str, group_key: str | None, xlabel: str, ylabel: str) -> None:
    clean = [r for r in rows if as_float(r.get(x_key)) is not None and as_float(r.get(y_key)) is not None]
    if not clean:
        return
    width, height = 660, 430
    left, right, top, bottom = 70, 120, 45, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_vals = [safe_float(r[x_key]) for r in clean]
    y_vals = [safe_float(r[y_key]) for r in clean]
    x0, x1 = nice_range(x_vals)
    y0, y1 = nice_range(y_vals, lower_zero=False)
    groups = sorted({str(r.get(group_key, "all")) for r in clean}) if group_key else ["all"]
    color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        y = scale(yv, y0, y1, top + plot_h, top)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{fmt(yv, 3)}</text>')
    for group in groups:
        sub = [r for r in clean if (str(r.get(group_key, "all")) if group_key else "all") == group]
        sub = sorted(sub, key=lambda r: safe_float(r[x_key]))
        coords = []
        for r in sub:
            x = scale(safe_float(r[x_key]), x0, x1, left, left + plot_w)
            y = scale(safe_float(r[y_key]), y0, y1, top + plot_h, top)
            coords.append(f"{x:.1f},{y:.1f}")
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color[group]}"/>')
        if len(coords) >= 2:
            body.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color[group]}" stroke-width="1.6"/>')
    for idx, group in enumerate(groups):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+plot_w+18}" y="{y-9}" width="10" height="10" fill="{color[group]}"/>')
        body.append(f'<text x="{left+plot_w+34}" y="{y}" font-size="11">{svg_escape(group)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    body.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-size="12">{svg_escape(xlabel)}</text>')
    body.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18,{top+plot_h/2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>')
    write_svg(path, width, height, "\n".join(body))


def heatmap(path: Path, rows: list[dict[str, Any]], title: str, row_key: str, col_key: str, value_key: str, value_range: tuple[float, float] | None = None) -> None:
    clean = [r for r in rows if as_float(r.get(value_key)) is not None]
    if not clean:
        return
    row_labels = sorted({str(r[row_key]) for r in clean})
    col_labels = sorted({str(r[col_key]) for r in clean}, key=lambda x: (as_float(x) is None, as_float(x) if as_float(x) is not None else x))
    table = {(str(r[row_key]), str(r[col_key])): safe_float(r[value_key]) for r in clean}
    values = list(table.values())
    v0, v1 = value_range if value_range else nice_range(values)
    cell_w = 68
    cell_h = 28
    left = max(110, max(len(x) for x in row_labels) * 7 + 20)
    top = 70
    width = left + cell_w * len(col_labels) + 40
    height = top + cell_h * len(row_labels) + 45
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    for j, col in enumerate(col_labels):
        x = left + j * cell_w + cell_w / 2
        body.append(f'<text x="{x:.1f}" y="52" text-anchor="middle" font-size="10">{svg_escape(col)}</text>')
    for i, row in enumerate(row_labels):
        y = top + i * cell_h + cell_h / 2 + 4
        body.append(f'<text x="{left-8}" y="{y:.1f}" text-anchor="end" font-size="10">{svg_escape(row)}</text>')
        for j, col in enumerate(col_labels):
            x = left + j * cell_w
            y0 = top + i * cell_h
            val = table.get((row, col))
            if val is None:
                fill = "#f3f3f3"
                text = ""
            else:
                t = 0.5 if v1 == v0 else max(0.0, min(1.0, (val - v0) / (v1 - v0)))
                r = int(246 * (1 - t) + 89 * t)
                g = int(246 * (1 - t) + 156 * t)
                b = int(246 * (1 - t) + 180 * t)
                fill = f"#{r:02x}{g:02x}{b:02x}"
                text = fmt(val, 3)
            body.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white"/>')
            body.append(f'<text x="{x+cell_w/2:.1f}" y="{y0+cell_h/2+4:.1f}" text-anchor="middle" font-size="9">{text}</text>')
    write_svg(path, width, height, "\n".join(body))


def stacked_bar(path: Path, rows: list[dict[str, Any]], title: str, x_key: str, category_key: str, value_key: str) -> None:
    if not rows:
        return
    labels = sorted({str(r[x_key]) for r in rows})
    cats = sorted({str(r[category_key]) for r in rows})
    values = {(str(r[x_key]), str(r[category_key])): safe_float(r[value_key]) for r in rows}
    width = max(640, 105 * len(labels) + 160)
    height = 430
    left, right, top, bottom = 70, 150, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    body = [f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>']
    bar_w = plot_w / len(labels) * 0.6
    color = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(cats)}
    for idx, label in enumerate(labels):
        cx = left + plot_w * (idx + 0.5) / len(labels)
        y_top = top + plot_h
        total = sum(values.get((label, cat), 0.0) for cat in cats) or 1.0
        for cat in cats:
            frac = values.get((label, cat), 0.0) / total
            h = frac * plot_h
            y_top -= h
            body.append(f'<rect x="{cx-bar_w/2:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color[cat]}" stroke="#333" stroke-width="0.4"/>')
        body.append(f'<text x="{cx:.1f}" y="{height-35}" text-anchor="middle" font-size="11">{svg_escape(label)}</text>')
    for i in range(5):
        y = top + plot_h - plot_h * i / 4
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{i/4:.2f}</text>')
    for idx, cat in enumerate(cats):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+plot_w+20}" y="{y-9}" width="10" height="10" fill="{color[cat]}"/>')
        body.append(f'<text x="{left+plot_w+36}" y="{y}" font-size="11">{svg_escape(cat)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    write_svg(path, width, height, "\n".join(body))


def discover_cell_eval_dataset(config: dict[str, Any]) -> CellEvalDataset:
    path = Path(config["path"])
    layout = config.get("layout", "auto")
    results_root = path
    if path.name != "results_calibrate" and (path / "results_calibrate").exists():
        results_root = path / "results_calibrate"
    dataset = CellEvalDataset(config=config)
    if layout == "onecell_nested":
        for run_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            cell_dir = run_dir / "results_calibrate" / run_dir.name
            result_path = cell_dir / "results.csv"
            if not result_path.exists():
                continue
            add_cell_eval_rows(dataset, result_path, group=run_dir.name)
            real = cell_dir / "real_de.csv"
            pred = cell_dir / "pred_de.csv"
            if real.exists() and pred.exists():
                dataset.de_pairs.append({"group": run_dir.name, "real": str(real), "pred": str(pred)})
        return dataset
    if layout == "split_prefixed":
        for cell_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            prefix = cell_dir.name
            result_path = cell_dir / f"{prefix}_results.csv"
            if not result_path.exists():
                continue
            add_cell_eval_rows(dataset, result_path, group=prefix)
            real = cell_dir / f"{prefix}_real_de.csv"
            pred = cell_dir / f"{prefix}_pred_de.csv"
            if real.exists() and pred.exists():
                dataset.de_pairs.append({"group": prefix, "real": str(real), "pred": str(pred)})
        return dataset
    if layout in ("auto", "nested") and any((p / "results.csv").exists() for p in results_root.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
            result_path = cell_dir / "results.csv"
            if not result_path.exists():
                continue
            add_cell_eval_rows(dataset, result_path, group=cell_dir.name)
            real = cell_dir / "real_de.csv"
            pred = cell_dir / "pred_de.csv"
            if real.exists() and pred.exists():
                dataset.de_pairs.append({"group": cell_dir.name, "real": str(real), "pred": str(pred)})
        return dataset
    for result_path in sorted(results_root.glob("*_results.csv")):
        prefix = result_path.name[: -len("_results.csv")]
        add_cell_eval_rows(dataset, result_path, group=prefix)
        real = results_root / f"{prefix}_real_de.csv"
        pred = results_root / f"{prefix}_pred_de.csv"
        if real.exists() and pred.exists():
            dataset.de_pairs.append({"group": prefix, "real": str(real), "pred": str(pred)})
    return dataset


def add_cell_eval_rows(dataset: CellEvalDataset, result_path: Path, group: str) -> None:
    cfg = dataset.config
    for row in iter_csv_rows(result_path):
        out = {
            "dataset_id": cfg["id"],
            "benchmark": cfg.get("benchmark", cfg["id"]),
            "model_label": cfg.get("label", cfg["id"]),
            "role": cfg.get("role", ""),
            "group": group,
            "split_class": split_class_for(cfg.get("benchmark", cfg["id"]), cfg["id"], group),
            "source_file": str(result_path),
            "perturbation": row.get("perturbation", ""),
        }
        for metric in CORE_METRICS:
            out[metric] = row.get(metric, "")
        if as_float(out.get("mse_delta")) is not None:
            out["rmse_delta"] = math.sqrt(max(0.0, safe_float(out["mse_delta"])))
        dataset.rows.append(out)


def summarize_cell_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_id"],
            row["benchmark"],
            row["model_label"],
            row["role"],
            row.get("group", ""),
        )
        grouped[key].append(row)
    summary = []
    metrics = list(CORE_METRICS) + ["rmse_delta"]
    for key, group_rows in grouped.items():
        dataset_id, benchmark, model_label, role, group = key
        for metric in metrics:
            vals = [as_float(row.get(metric)) for row in group_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            summary.append(
                {
                    "dataset_id": dataset_id,
                    "benchmark": benchmark,
                "model_label": model_label,
                "role": role,
                "group": group,
                "split_class": split_class_for(benchmark, dataset_id, group),
                "metric": metric,
                "n": len(vals),
                "mean": mean(vals),
                    "median": median(vals),
                    "std": stdev(vals),
                    "q25": quantile(vals, 0.25),
                    "q75": quantile(vals, 0.75),
                    "min": min(vals),
                    "max": max(vals),
                }
            )
    return summary


def load_cell_eval_summary(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["path"])
    rows = read_csv_rows(path)
    chosen = None
    for row in rows:
        first = row.get("cell_line") or row.get("statistic")
        if first in {"macro_mean", "mean_of_means"} or row.get("statistic") == "mean_of_means":
            chosen = row
            break
    if chosen is None and rows:
        chosen = rows[-1]
    out = []
    if not chosen:
        return out
    for metric in CORE_METRICS:
        if as_float(chosen.get(metric)) is None:
            continue
        out.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", config["id"]),
                "model_label": config.get("label", config["id"]),
                "role": config.get("role", ""),
                "split_class": split_class_for(config.get("benchmark", config["id"]), config["id"], ""),
                "metric": metric,
                "n": chosen.get("source_file_count", ""),
                "mean": safe_float(chosen[metric]),
                "source_file": str(path),
            }
        )
    if as_float(chosen.get("mse_delta")) is not None:
        out.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", config["id"]),
                "model_label": config.get("label", config["id"]),
                "role": config.get("role", ""),
                "split_class": split_class_for(config.get("benchmark", config["id"]), config["id"], ""),
                "metric": "rmse_delta",
                "n": chosen.get("source_file_count", ""),
                "mean": math.sqrt(max(0.0, safe_float(chosen["mse_delta"]))),
                "source_file": str(path),
            }
        )
    return out


def log2_fc(value: Any) -> float:
    parsed = as_float(value)
    if parsed is None:
        return 0.0
    return math.log(max(parsed, EPS), 2)


def build_de_events(dataset: CellEvalDataset, top_k: int = 100) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cfg = dataset.config
    for pair in dataset.de_pairs:
        group = pair["group"]
        real_by_target: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for row in iter_csv_rows(Path(pair["real"])):
            if safe_float(row.get("fdr"), 1.0) >= 0.05:
                continue
            lfc = log2_fc(row.get("fold_change"))
            real_by_target[row.get("target", "")].append((row.get("feature", ""), lfc, safe_float(row.get("fdr"), 1.0)))
        selected: dict[tuple[str, str], tuple[int, float]] = {}
        for target, genes in real_by_target.items():
            genes = sorted(genes, key=lambda x: (-abs(x[1]), x[2], x[0]))[:top_k]
            for rank, (feature, real_lfc, _fdr) in enumerate(genes, start=1):
                selected[(target, feature)] = (rank, real_lfc)
        pred_lookup: dict[tuple[str, str], float] = {}
        for row in iter_csv_rows(Path(pair["pred"])):
            key = (row.get("target", ""), row.get("feature", ""))
            if key in selected:
                pred_lookup[key] = log2_fc(row.get("fold_change"))
        for (target, feature), (rank, real_lfc) in selected.items():
            pred_lfc = pred_lookup.get((target, feature), 0.0)
            real_sign = 1 if real_lfc > 0 else -1 if real_lfc < 0 else 0
            pred_sign = 1 if pred_lfc > 0 else -1 if pred_lfc < 0 else 0
            direction_match = real_sign == pred_sign
            amp_ratio = (abs(pred_lfc) + EPS) / (abs(real_lfc) + EPS)
            if not direction_match:
                category = "wrong_direction"
            elif amp_ratio < 0.75:
                category = "correct_under"
            elif amp_ratio > 1.25:
                category = "correct_over"
            else:
                category = "correct_matched"
            events.append(
                {
                    "dataset_id": cfg["id"],
                    "benchmark": cfg.get("benchmark", cfg["id"]),
                    "model_label": cfg.get("label", cfg["id"]),
                    "model_group": f"{cfg.get('label', cfg['id'])}:{group}",
                    "group": group,
                    "split_class": split_class_for(cfg.get("benchmark", cfg["id"]), cfg["id"], group),
                    "perturbation": target,
                    "feature": feature,
                    "rank": rank,
                    "real_log2fc": real_lfc,
                    "pred_log2fc": pred_lfc,
                    "direction_match": int(direction_match),
                    "amp_ratio": amp_ratio,
                    "log2_amp_ratio": math.log(amp_ratio, 2),
                    "category": category,
                }
            )
    return events


def summarize_de_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    category_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    gene_group: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        split_class = event.get("split_class", split_class_for(event.get("benchmark", ""), event.get("dataset_id", ""), event.get("group", "")))
        by_group[(event["dataset_id"], event["model_label"], event["group"], split_class)].append(event)
        category_counts[(event["dataset_id"], event["model_label"], event["group"], split_class, event["category"])] += 1
        gene_group[(event["dataset_id"], event["model_label"], event["group"], split_class, event["feature"])].append(event)
    group_summary = []
    for (dataset_id, model_label, group, split_class), vals in by_group.items():
        group_summary.append(
            {
                "dataset_id": dataset_id,
                "model_label": model_label,
                "group": group,
                "split_class": split_class,
                "n_events": len(vals),
                "direction_match_rate": mean([safe_float(v["direction_match"]) for v in vals]),
                "mean_log2_amp_ratio": mean([safe_float(v["log2_amp_ratio"]) for v in vals]),
                "median_log2_amp_ratio": median([safe_float(v["log2_amp_ratio"]) for v in vals]),
            }
        )
    category_summary = []
    totals: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for (dataset_id, model_label, group, split_class, _cat), count in category_counts.items():
        totals[(dataset_id, model_label, group, split_class)] += count
    for (dataset_id, model_label, group, split_class, cat), count in category_counts.items():
        total = totals[(dataset_id, model_label, group, split_class)] or 1
        category_summary.append(
            {
                "dataset_id": dataset_id,
                "model_label": model_label,
                "group": group,
                "split_class": split_class,
                "model_group": f"{model_label}:{group}",
                "category": cat,
                "n": count,
                "fraction": count / total,
            }
        )
    gene_summary = []
    for (dataset_id, model_label, group, split_class, feature), vals in gene_group.items():
        gene_summary.append(
            {
                "dataset_id": dataset_id,
                "model_label": model_label,
                "group": group,
                "split_class": split_class,
                "model_group": f"{model_label}:{group}",
                "feature": feature,
                "n_occurrences": len(vals),
                "direction_match_rate": mean([safe_float(v["direction_match"]) for v in vals]),
                "wrong_direction_fraction": 1.0 - (mean([safe_float(v["direction_match"]) for v in vals]) or 0.0),
                "mean_log2_amp_ratio": mean([safe_float(v["log2_amp_ratio"]) for v in vals]),
            }
        )
    gene_summary.sort(key=lambda r: (-int(r["n_occurrences"]), str(r["feature"]), str(r["group"])))
    return group_summary, category_summary, gene_summary


def load_kang_pdc(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(config["path"])
    csv_path = root / "audit" / "pdc_score_by_fold_lambda.csv"
    summary_path = root / "audit" / "pdc_score_summary.json"
    raw_audit_path = root / "audit" / "raw_h5ad_audit.json"
    rows = []
    if csv_path.exists():
        for row in iter_csv_rows(csv_path):
            out = {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", "Kang"),
                "fold": row.get("fold", ""),
                "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], row.get("fold", "")),
                "lambda": row.get("lambda", ""),
                "mean_delta_pcc": row.get("mean_delta_pcc", ""),
                "median_delta_pcc": row.get("median_delta_pcc", ""),
                "raw_per_group_macro": row.get("raw_per_group_macro", ""),
                "delta_vs_raw_per_group_macro": row.get("delta_vs_raw_per_group_macro", ""),
            }
            rows.append(out)
    summary_rows = []
    if raw_audit_path.exists():
        audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", "Kang"),
                "model_label": "raw weak baseline",
                "role": config.get("raw_audit_role", "non-scale baseline"),
                "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
                "metric": "pearson_delta",
                "mean": audit.get("macro_raw_delta_pcc_mean_over_folds"),
                "source_file": str(raw_audit_path),
            }
        )
    if rows:
        lambda0 = [safe_float(r["mean_delta_pcc"]) for r in rows if str(r["lambda"]) == "0"]
        best_by_fold = {}
        for row in rows:
            fold = row["fold"]
            val = as_float(row.get("mean_delta_pcc"))
            if val is None:
                continue
            if fold not in best_by_fold or val > best_by_fold[fold]:
                best_by_fold[fold] = val
        summary_rows.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", "Kang"),
                "model_label": "lambda0 model",
                "role": config.get("lambda0_role", "diagnostic"),
                "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
                "metric": "pearson_delta",
                "mean": mean(lambda0),
                "source_file": str(csv_path),
            }
        )
        summary_rows.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", "Kang"),
                "model_label": config.get("label", "scale+pdc"),
                "role": config.get("role", "primary"),
                "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
                "metric": "pearson_delta",
                "mean": mean(best_by_fold.values()),
                "source_file": str(csv_path),
            }
        )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best_lambda_by_test_diagnostic", {})
        if best:
            summary_rows.append(
                {
                    "dataset_id": config["id"],
                    "benchmark": config.get("benchmark", "Kang"),
                    "model_label": f"global best lambda={best.get('lambda')}",
                    "role": "diagnostic",
                    "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
                    "metric": "pearson_delta",
                    "mean": best.get("fold_mean_delta_pcc_mean"),
                    "source_file": str(summary_path),
                }
            )
    return rows, summary_rows


def load_kang_summary(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(config["path"])
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for patient, vals in summary.get("by_patient", {}).items():
        rows.append(
            {
                "dataset_id": config["id"],
                "benchmark": config.get("benchmark", "Kang"),
                "patient": patient,
                "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], patient),
                "raw_lambda0_mean": vals.get("raw_lambda0_mean"),
                "selected_mean": vals.get("selected_mean"),
                "pdc_minus_raw": vals.get("pdc_minus_raw"),
            }
        )
    summary_rows = [
        {
            "dataset_id": config["id"],
            "benchmark": config.get("benchmark", "Kang"),
            "model_label": config.get("label", "weak baseline"),
            "role": config.get("role", "baseline"),
            "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
            "metric": "pearson_delta",
            "mean": summary.get("aggregate_selected_mean"),
            "source_file": str(path),
        },
        {
            "dataset_id": config["id"],
            "benchmark": config.get("benchmark", "Kang"),
            "model_label": "raw lambda0",
            "role": config.get("raw_lambda0_role", "non-scale baseline"),
            "split_class": split_class_for(config.get("benchmark", "Kang"), config["id"], ""),
            "metric": "pearson_delta",
            "mean": summary.get("aggregate_raw_lambda0_mean"),
            "source_file": str(path),
        },
    ]
    return rows, summary_rows


def row_matches_filter(row: dict[str, str], config: dict[str, Any]) -> bool:
    include_datasets = set(config.get("include_datasets", []))
    if include_datasets and row.get("dataset", "") not in include_datasets:
        return False
    include_split_ids = set(config.get("include_split_ids", []))
    if include_split_ids and row.get("split_id", "") not in include_split_ids:
        return False
    exclude_split_ids = set(config.get("exclude_split_ids", []))
    if exclude_split_ids and row.get("split_id", "") in exclude_split_ids:
        return False
    for key, expected in config.get("exact_match", {}).items():
        if row.get(key, "") != expected:
            return False
    for key, expected in config.get("contains", {}).items():
        if expected not in row.get(key, ""):
            return False
    for key, forbidden in config.get("not_contains", {}).items():
        if forbidden in row.get(key, ""):
            return False
    return True


def select_numeric_column(row: dict[str, str], columns: list[str], mode: str = "first") -> tuple[str, float] | None:
    candidates = []
    for column in columns:
        val = as_float(row.get(column))
        if val is not None:
            candidates.append((column, val))
    if not candidates:
        return None
    if mode == "max":
        return max(candidates, key=lambda item: item[1])
    if mode == "min":
        return min(candidates, key=lambda item: item[1])
    return candidates[0]


def load_wide_comparison(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(config["path"])
    metric = config.get("metric", "external_pearson_delta")
    external_models = config.get("external_models") or [
        {"label": model, "column": model} for model in DEFAULT_EXTERNAL_BASELINES
    ]
    primary_models = config.get("primary_models", [])
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for row in iter_csv_rows(path):
        if not row_matches_filter(row, config):
            continue
        benchmark = row.get("dataset", config.get("benchmark", config["id"]))
        split_id = row.get("split_id", "")
        split_label = row.get("split_label", "")
        dataset_id = f"{config['id']}__{slugify(benchmark)}"
        base = {
            "dataset_id": dataset_id,
            "benchmark": benchmark,
            "group": split_id,
            "split_label": split_label,
            "split_class": split_class_for(benchmark, row.get("dataset", ""), split_id),
            "metric": metric,
            "metric_protocol": row.get("metric_protocol", ""),
            "source_file": str(path),
            "source_row_metric": row.get("metric", ""),
            "claim_status": row.get("claim_statuses", row.get("claim_status", "")),
            "notes": row.get("notes", ""),
        }
        for model in primary_models:
            label = model["label"]
            columns = model.get("columns", [])
            found = select_numeric_column(row, columns, model.get("select", "first"))
            status = "present" if found else "missing"
            availability_rows.append(
                {
                    **base,
                    "model_label": label,
                    "role": model.get("role", "primary"),
                    "status": status,
                    "source_column": found[0] if found else "",
                    "value": found[1] if found else "",
                    "note": "" if found else "No configured primary model column has a value in this selected row/protocol.",
                }
            )
            if not found:
                continue
            source_column, value = found
            out = {
                **base,
                "model_label": label,
                "role": model.get("role", "primary"),
                "mean": value,
                "source_column": source_column,
            }
            summary_rows.append(out)
            detail_rows.append(out)
        for model in external_models:
            label = model["label"]
            column = model.get("column", label)
            value = as_float(row.get(column))
            status = "present" if value is not None else "missing"
            availability_rows.append(
                {
                    **base,
                    "model_label": label,
                    "role": model.get("role", "external baseline"),
                    "status": status,
                    "source_column": column if value is not None else "",
                    "value": value if value is not None else "",
                    "note": "" if value is not None else f"No {label} value in this selected row/protocol.",
                }
            )
            if value is None:
                continue
            out = {
                **base,
                "model_label": label,
                "role": model.get("role", "external baseline"),
                "mean": value,
                "source_column": column,
            }
            summary_rows.append(out)
            detail_rows.append(out)
    return summary_rows, detail_rows, availability_rows


def summarize_detail_values(
    rows: list[dict[str, Any]],
    key_fields: list[str],
    value_key: str = "value",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    examples: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        value = as_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(field, "") for field in key_fields)
        grouped[key].append(value)
        examples.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for key, vals in grouped.items():
        base = {field: value for field, value in zip(key_fields, key)}
        example = examples[key]
        for field in (
            "dataset_id",
            "benchmark",
            "split_class",
            "group",
            "split_label",
            "model_label",
            "metric",
            "cellclass",
            "source_type",
            "metric_direction",
        ):
            if field not in base and field in example:
                base[field] = example[field]
        base.update(
            {
                "n": len(vals),
                "mean": mean(vals),
                "median": median(vals),
                "std": stdev(vals),
                "q25": quantile(vals, 0.25),
                "q75": quantile(vals, 0.75),
                "min": min(vals),
                "max": max(vals),
            }
        )
        out.append(base)
    out.sort(key=lambda r: tuple(str(r.get(field, "")) for field in key_fields))
    return out


def load_vcbench_metric_details(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    avg_path = Path(config["avg_path"])
    cellclass_path = Path(config["cellclass_path"])
    metrics = set(config.get("metrics", BASELINE_DETAIL_METRICS))
    include_models = {str(model).lower() for model in config.get("include_models", EXTERNAL_MODEL_LABELS.keys())}
    include_datasets = config.get("include_datasets", [])
    dataset_label_map = config.get("dataset_label_map", {})
    label_map = config.get("model_label_map", {})

    def convert(row: dict[str, str], source_type: str) -> dict[str, Any] | None:
        dataset = row.get("dataset", "")
        if not allowed_value(dataset, include_datasets):
            return None
        model = row.get("model", "").lower()
        if include_models and model not in include_models:
            return None
        metric = row.get("metric", "")
        if metric not in metrics:
            return None
        value = as_float(row.get("value"))
        if value is None:
            return None
        benchmark = benchmark_label_from_dataset(dataset, dataset_label_map)
        group = row.get("split_key", "")
        split_class = split_class_for(benchmark, dataset, group)
        return {
            "dataset_id": config["id"],
            "benchmark": benchmark,
            "dataset_key": dataset,
            "group": group,
            "split_label": group,
            "split_class": split_class,
            "model": model,
            "model_label": model_label(model, label_map),
            "role": "external baseline",
            "seed": row.get("seed", ""),
            "metric": metric,
            "metric_label": metric_label(metric),
            "metric_direction": metric_direction(metric),
            "value": value,
            "source_type": source_type,
            "source_file": row.get("source_csv", str(avg_path if source_type == "avg" else cellclass_path)),
            "run_id": row.get("run_id", ""),
            "cellclass": row.get("cellclass", ""),
        }

    avg_rows: list[dict[str, Any]] = []
    for row in iter_csv_rows(avg_path):
        converted = convert(row, "vcbench_avg")
        if converted:
            avg_rows.append(converted)
    cellclass_rows: list[dict[str, Any]] = []
    for row in iter_csv_rows(cellclass_path):
        converted = convert(row, "vcbench_cellclass")
        if converted:
            cellclass_rows.append(converted)

    avg_summary = summarize_detail_values(
        avg_rows,
        ["dataset_id", "benchmark", "split_class", "group", "model_label", "metric"],
    )
    cellclass_summary = summarize_detail_values(
        cellclass_rows,
        ["dataset_id", "benchmark", "split_class", "group", "cellclass", "model_label", "metric"],
    )
    return avg_rows, avg_summary, cellclass_rows, cellclass_summary


def load_matched_eval_details(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detail_path = Path(config["detail_path"])
    group_path = Path(config["group_path"])
    include_models = {str(model) for model in config.get("include_models", [])}
    label_map = config.get("model_label_map", {})
    detail_metrics = config.get("detail_metrics", MATCHED_DETAIL_METRICS)
    group_metrics = config.get("group_metrics", MATCHED_GROUP_METRICS)

    def keep_model(model: str) -> bool:
        return not include_models or model in include_models

    def benchmark_for_task(task: str, split: str) -> str:
        if task == "task04" or split.startswith("task04"):
            return "ComboSciPlex3 Task04"
        return "ComboSciPlex3 Task03"

    detail_rows: list[dict[str, Any]] = []
    for row in iter_csv_rows(detail_path):
        model = row.get("model", "")
        if not keep_model(model):
            continue
        split = row.get("split", "")
        task = "task04" if split.startswith("task04") else "task03"
        benchmark = benchmark_for_task(task, split)
        split_class = split_class_for(benchmark, task, split)
        for metric in detail_metrics:
            value = as_float(row.get(metric))
            if value is None:
                continue
            detail_rows.append(
                {
                    "dataset_id": config["id"],
                    "benchmark": benchmark,
                    "split_class": split_class,
                    "group": split,
                    "split_label": split,
                    "model": model,
                    "model_label": model_label(model, label_map),
                    "role": "primary" if model.startswith("ours") else "external baseline",
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "metric_direction": metric_direction(metric),
                    "value": value,
                    "source_type": "matched_eval_detail",
                    "source_file": str(detail_path),
                    "n_groups_total": row.get("n_groups_total", ""),
                    "min_group_cells": row.get("min_group_cells", ""),
                    "median_group_cells": row.get("median_group_cells", ""),
                    "prediction_path": row.get("prediction_path", ""),
                }
            )
    detail_summary = summarize_detail_values(
        detail_rows,
        ["dataset_id", "benchmark", "split_class", "group", "model_label", "metric"],
    )

    group_rows: list[dict[str, Any]] = []
    for row in iter_csv_rows(group_path):
        model = row.get("model", "")
        if not keep_model(model):
            continue
        split = row.get("split", "")
        task = row.get("task", "task04" if split.startswith("task04") else "task03")
        benchmark = benchmark_for_task(task, split)
        split_class = split_class_for(benchmark, task, split)
        out = {
            "dataset_id": config["id"],
            "benchmark": benchmark,
            "split_class": split_class,
            "group": split,
            "split_label": split,
            "model": model,
            "model_label": model_label(model, label_map),
            "role": "primary" if model.startswith("ours") else "external baseline",
            "source_type": "matched_group_metrics",
            "source_file": str(group_path),
            "prediction_path": row.get("prediction_path", ""),
            "real_path": row.get("real_path", ""),
            "perturbation": row.get("perturbation", ""),
            "n_pred_cells": row.get("n_pred_cells", ""),
            "n_real_cells": row.get("n_real_cells", ""),
        }
        for metric in group_metrics:
            out[metric] = row.get(metric, "")
        group_rows.append(out)

    group_metric_rows: list[dict[str, Any]] = []
    for row in group_rows:
        for metric in group_metrics:
            value = as_float(row.get(metric))
            if value is None:
                continue
            group_metric_rows.append(
                {
                    **{k: row.get(k, "") for k in ("dataset_id", "benchmark", "split_class", "group", "model_label", "role")},
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "metric_direction": metric_direction(metric),
                    "value": value,
                    "source_type": "matched_group_metrics",
                }
            )
    group_summary = summarize_detail_values(
        group_metric_rows,
        ["dataset_id", "benchmark", "split_class", "group", "model_label", "metric"],
    )
    return detail_rows, detail_summary, group_rows, group_summary


def build_gene_level_de_availability(
    config: dict[str, Any],
    cell_datasets: list[CellEvalDataset],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in cell_datasets:
        cfg = dataset.config
        status = "available" if dataset.de_pairs else "missing"
        rows.append(
            {
                "dataset_id": cfg["id"],
                "benchmark": cfg.get("benchmark", cfg["id"]),
                "split_class": split_class_for(cfg.get("benchmark", cfg["id"]), cfg["id"], ""),
                "model_label": cfg.get("label", cfg["id"]),
                "role": cfg.get("role", ""),
                "artifact": "real_de.csv + pred_de.csv",
                "status": status,
                "usable_figure": "figures/de_top100_real_vs_pred_log2fc.svg" if status == "available" else "",
                "note": f"Found {len(dataset.de_pairs)} real/pred DE file pairs." if status == "available" else "No real/pred DE CSV pair found.",
            }
        )
    rows.extend(config.get("gene_level_de_availability", []))
    return rows


def write_matched_perturbation_examples(outputs: AnalysisOutputs, group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_section_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        if as_float(row.get("deltaPCC")) is not None:
            by_section_model[(row.get("split_class", ""), row.get("model_label", ""))].append(row)
    for (split_class, model), rows in by_section_model.items():
        ordered = sorted(rows, key=lambda r: safe_float(r.get("deltaPCC")))
        if len(ordered) <= 10:
            selected = [(row, "all_examples") for row in ordered]
        else:
            selected = [(row, "low_deltaPCC") for row in ordered[:5]] + [(row, "high_deltaPCC") for row in ordered[-5:]]
        seen_examples: set[tuple[str, str, str]] = set()
        for row, bucket in selected:
            dedupe_key = (row.get("group", ""), row.get("model_label", ""), row.get("perturbation", ""))
            if dedupe_key in seen_examples:
                continue
            seen_examples.add(dedupe_key)
            examples.append(
                {
                    "split_class": split_class,
                    "benchmark": row.get("benchmark", ""),
                    "group": row.get("group", ""),
                    "model_label": model,
                    "bucket": bucket,
                    "perturbation": row.get("perturbation", ""),
                    "deltaPCC": row.get("deltaPCC", ""),
                    "rmse_average": row.get("rmse_average", ""),
                    "pearson_average": row.get("pearson_average", ""),
                    "n_real_cells": row.get("n_real_cells", ""),
                    "n_pred_cells": row.get("n_pred_cells", ""),
                }
            )
    write_csv(outputs.tables / "matched_perturbation_examples.csv", examples)
    return examples


def summarize_primary_vs_external(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_split: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if as_float(row.get("mean")) is None:
            continue
        key = (
            row["dataset_id"],
            row["benchmark"],
            row.get("group", ""),
            row.get("split_class", split_class_for(row.get("benchmark", ""), row.get("dataset_id", ""), row.get("group", ""))),
            row["metric"],
        )
        by_split[key].append(row)
    split_summary: list[dict[str, Any]] = []
    for (dataset_id, benchmark, group, split_class, metric), vals in by_split.items():
        primary = [r for r in vals if r.get("role") == "primary"]
        external = [r for r in vals if r.get("role") == "external baseline"]
        if not primary or not external:
            split_summary.append(
                {
                    "dataset_id": dataset_id,
                    "benchmark": benchmark,
                    "group": group,
                    "split_class": split_class,
                    "metric": metric,
                    "primary_model": primary[0]["model_label"] if primary else "",
                    "primary_value": primary[0]["mean"] if primary else "",
                    "best_external_model": external[0]["model_label"] if external else "",
                    "best_external_value": external[0]["mean"] if external else "",
                    "margin": "",
                    "status": "missing_external" if primary else "missing_primary",
                }
            )
            continue
        best_primary = max(primary, key=lambda r: safe_float(r.get("mean")))
        best_external = max(external, key=lambda r: safe_float(r.get("mean")))
        margin = safe_float(best_primary["mean"]) - safe_float(best_external["mean"])
        if margin >= 0.02:
            status = "primary_better"
        elif margin <= -0.02:
            status = "external_better"
        else:
            status = "near_tie"
        split_summary.append(
            {
                "dataset_id": dataset_id,
                "benchmark": benchmark,
                "group": group,
                "split_class": split_class,
                "metric": metric,
                "primary_model": best_primary["model_label"],
                "primary_value": best_primary["mean"],
                "best_external_model": best_external["model_label"],
                "best_external_value": best_external["mean"],
                "margin": margin,
                "status": status,
            }
        )
    benchmark_summary: list[dict[str, Any]] = []
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_summary:
        if as_float(row.get("margin")) is not None:
            by_benchmark[row["benchmark"]].append(row)
    for benchmark, vals in by_benchmark.items():
        margins = [safe_float(v["margin"]) for v in vals]
        benchmark_summary.append(
            {
                "benchmark": benchmark,
                "n_splits": len(vals),
                "mean_primary": mean([safe_float(v["primary_value"]) for v in vals]),
                "mean_best_external": mean([safe_float(v["best_external_value"]) for v in vals]),
                "mean_margin": mean(margins),
                "win_rate": mean([1.0 if safe_float(v["margin"]) > 0 else 0.0 for v in vals]),
                "best_external_models": ";".join(sorted({v["best_external_model"] for v in vals})),
            }
        )
    benchmark_summary.sort(key=lambda r: str(r["benchmark"]))
    return split_summary, benchmark_summary


def comparison_rows(cell_summary: list[dict[str, Any]], summary_only: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in cell_summary:
        if row.get("group"):
            key = (row["dataset_id"], row["benchmark"], row["model_label"], row["role"], row["metric"], row.get("split_class", ""))
            if as_float(row.get("mean")) is not None:
                grouped[key].append(safe_float(row["mean"]))
    for key, vals in grouped.items():
        dataset_id, benchmark, model_label, role, metric, split_class = key
        rows.append(
            {
                "dataset_id": dataset_id,
                "benchmark": benchmark,
                "model_label": model_label,
                "role": role,
                "metric": metric,
                "split_class": split_class,
                "mean": mean(vals),
                "n_groups": len(vals),
                "source": "cell_eval_groups",
            }
        )
    summary_grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    summary_sources: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    for row in summary_only:
        if as_float(row.get("mean")) is None:
            continue
        if row.get("group"):
            key = (row["dataset_id"], row["benchmark"], row["model_label"], row.get("role", ""), row["metric"], row.get("split_class", ""))
            summary_grouped[key].append(safe_float(row["mean"]))
            if row.get("source_file"):
                summary_sources[key].add(str(row["source_file"]))
        else:
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "benchmark": row["benchmark"],
                    "model_label": row["model_label"],
                    "role": row.get("role", ""),
                    "metric": row["metric"],
                    "split_class": row.get("split_class", ""),
                    "mean": safe_float(row["mean"]),
                    "n_groups": row.get("n", ""),
                    "source": row.get("source_file", ""),
                }
            )
    for key, vals in summary_grouped.items():
        dataset_id, benchmark, model_label, role, metric, split_class = key
        rows.append(
            {
                "dataset_id": dataset_id,
                "benchmark": benchmark,
                "model_label": model_label,
                "role": role,
                "metric": metric,
                "split_class": split_class,
                "mean": mean(vals),
                "n_groups": len(vals),
                "source": "; ".join(sorted(summary_sources.get(key, []))),
            }
        )
    return rows


def inspect_latent_availability(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        root = Path(cfg.get("h5ad_root", cfg.get("path", "")))
        h5ads = []
        if root.exists():
            h5ads = list(root.glob("*.h5ad"))[:5]
        rows.append(
            {
                "dataset_id": cfg["id"],
                "benchmark": cfg.get("benchmark", cfg["id"]),
                "latent_status": "not_checked_runtime_dependency_free",
                "note": "Use h5py/anndata-enabled environment to inspect obsm; v1 skips latent figures unless embeddings are provided.",
                "example_h5ad_count": len(h5ads),
            }
        )
    return rows


def make_cell_eval_figures(outputs: AnalysisOutputs, datasets: list[CellEvalDataset], de_category: list[dict[str, Any]], de_summary: list[dict[str, Any]], de_events: list[dict[str, Any]], gene_summary: list[dict[str, Any]], top_gene_limit: int) -> None:
    for dataset in datasets:
        rows = dataset.rows
        if not rows:
            continue
        dataset_id = dataset.config["id"]
        label = dataset.config.get("label", dataset_id)
        groups = defaultdict(list)
        for row in rows:
            val = as_float(row.get("pearson_delta"))
            if val is not None:
                groups[row["group"]].append(val)
        box_chart(outputs.figures / f"{dataset_id}_delta_pcc_distribution.svg", dict(groups), f"{label}: deltaPCC distribution", "deltaPCC", ylim=(-0.1, 1.02))
        for k in TOP_KS:
            metric = f"precision_at_{k}"
            groups = defaultdict(list)
            for row in rows:
                val = as_float(row.get(metric))
                if val is not None:
                    groups[row["group"]].append(val)
            box_chart(outputs.figures / f"{dataset_id}_de_precision_at_{k}_distribution.svg", dict(groups), f"{label}: DE precision@{k}", f"precision@{k}", ylim=(0.0, 1.02))
        scatter_chart(
            outputs.figures / f"{dataset_id}_delta_pcc_vs_precision_at_100.svg",
            rows,
            f"{label}: deltaPCC vs precision@100",
            "pearson_delta",
            "precision_at_100",
            "group",
            "deltaPCC",
            "DE precision@100",
        )
    if de_category:
        stacked_bar(outputs.figures / "de_top100_direction_magnitude_categories.svg", de_category, "Real top100 DEG direction/magnitude classes", "model_group", "category", "fraction")
    if de_summary:
        heatmap(outputs.figures / "de_top100_direction_match_heatmap.svg", de_summary, "Top100 DEG direction match", "group", "model_label", "direction_match_rate", value_range=(0, 1))
        heatmap(outputs.figures / "de_top100_amplitude_bias_heatmap.svg", de_summary, "Top100 DEG amplitude bias", "group", "model_label", "mean_log2_amp_ratio")
    if de_events:
        sample = de_events[:: max(1, len(de_events) // 4000)]
        scatter_chart(
            outputs.figures / "de_top100_real_vs_pred_log2fc.svg",
            sample,
            "Real top100 DEG signed log2FC",
            "real_log2fc",
            "pred_log2fc",
            "model_label",
            "Real log2FC",
            "Predicted log2FC",
        )
    if gene_summary:
        top_genes = sorted({row["feature"] for row in gene_summary[:top_gene_limit]})
        top_rows = [row for row in gene_summary if row["feature"] in top_genes]
        heatmap(outputs.figures / "de_top100_gene_amplitude_bias_top_genes.svg", top_rows, "Frequent top100 genes: amplitude bias", "feature", "model_group", "mean_log2_amp_ratio")


def make_kang_figures(outputs: AnalysisOutputs, kang_rows: list[dict[str, Any]], kang_patient_rows: list[dict[str, Any]]) -> None:
    if kang_rows:
        fold_means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in kang_rows:
            fold_means[row["lambda"]]["mean_delta_pcc"].append(safe_float(row["mean_delta_pcc"]))
        lambda_rows = []
        for lam, vals in fold_means.items():
            lambda_rows.append({"lambda": lam, "mean_delta_pcc": mean(vals["mean_delta_pcc"])})
        line_chart(outputs.figures / "kang_lambda_sweep_delta_pcc.svg", lambda_rows, "Kang lambda sweep", "lambda", "mean_delta_pcc", None, "lambda", "mean deltaPCC")
        heatmap(outputs.figures / "kang_fold_lambda_delta_pcc_heatmap.svg", kang_rows, "Kang fold x lambda deltaPCC", "fold", "lambda", "mean_delta_pcc", value_range=(0.85, 1.0))
    if kang_patient_rows:
        bar_rows = []
        for row in kang_patient_rows:
            bar_rows.append({"label": f"{row['patient']} raw", "value": row.get("raw_lambda0_mean")})
            bar_rows.append({"label": f"{row['patient']} pdc", "value": row.get("selected_mean")})
        bar_chart(outputs.figures / "kang_patient_raw_vs_pdc.svg", bar_rows, "Kang patient raw vs PDC", "label", "value", "mean deltaPCC")


def make_comparison_figures(outputs: AnalysisOutputs, rows: list[dict[str, Any]]) -> None:
    delta_rows = [row for row in rows if row["metric"] in {"pearson_delta", "external_pearson_delta", "matched_delta_pcc"}]
    if delta_rows:
        plot_rows = []
        for row in delta_rows:
            plot_rows.append({"label": f"{row['benchmark']} {row['model_label']}", "value": row["mean"]})
        bar_chart(outputs.figures / "comparison_delta_pcc.svg", plot_rows, "Model comparison: deltaPCC", "label", "value", "mean deltaPCC")
    de_rows = [row for row in rows if row["metric"] in {"precision_at_100", "de_direction_match"}]
    if de_rows:
        for metric in sorted({row["metric"] for row in de_rows}):
            plot_rows = [
                {"label": f"{row['benchmark']} {row['model_label']}", "value": row["mean"]}
                for row in de_rows
                if row["metric"] == metric
            ]
            bar_chart(outputs.figures / f"comparison_{metric}.svg", plot_rows, f"Model comparison: {metric_label(metric)}", "label", "value", metric_label(metric))


def make_external_figures(
    outputs: AnalysisOutputs,
    external_rows: list[dict[str, Any]],
    split_summary: list[dict[str, Any]],
    benchmark_summary: list[dict[str, Any]],
) -> None:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in external_rows:
        if as_float(row.get("mean")) is not None:
            by_benchmark[row["benchmark"]].append(row)
    for benchmark, rows in by_benchmark.items():
        heatmap(
            outputs.figures / f"external_{slugify(benchmark)}_delta_pcc_heatmap.svg",
            rows,
            f"{benchmark}: primary vs external baselines",
            "group",
            "model_label",
            "mean",
            value_range=(-0.15, 1.0),
        )
    margin_rows = [
        {
            "label": f"{row['benchmark']} {row['group']}",
            "value": row["margin"],
        }
        for row in split_summary
        if as_float(row.get("margin")) is not None
    ]
    if margin_rows:
        bar_chart(
            outputs.figures / "primary_vs_best_external_margin_by_split.svg",
            margin_rows,
            "Primary minus best external baseline",
            "label",
            "value",
            "deltaPCC margin",
        )
    if benchmark_summary:
        bar_chart(
            outputs.figures / "primary_vs_best_external_mean_margin_by_benchmark.svg",
            [{"label": row["benchmark"], "value": row["mean_margin"]} for row in benchmark_summary],
            "Mean primary minus best external baseline",
            "label",
            "value",
            "mean deltaPCC margin",
        )


def aggregate_plot_rows(rows: list[dict[str, Any]], keys: list[str], value_key: str = "mean") -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    examples: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        value = as_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(field, "") for field in keys)
        grouped[key].append(value)
        examples.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for key, values in grouped.items():
        row = dict(examples[key])
        for field, value in zip(keys, key):
            row[field] = value
        row[value_key] = mean(values)
        row["n_plot_values"] = len(values)
        out.append(row)
    return out


def make_baseline_detail_figures(
    outputs: AnalysisOutputs,
    vcbench_metric_summary: list[dict[str, Any]],
    vcbench_cellclass_summary: list[dict[str, Any]],
    matched_detail_summary: list[dict[str, Any]],
    matched_group_rows: list[dict[str, Any]],
) -> None:
    higher_metrics = {"pearson_delta", "pearson_average", "cosine_logfc", "spearman_logfc", "deg_precision", "deg_recall", "deg_iou"}
    error_metrics = {"rmse_average", "rmse_average_mean", "row_rmse_to_gene_mean"}

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vcbench_metric_summary:
        by_section[row.get("split_class", "")].append(row)
    for split_class, rows in by_section.items():
        section_rows = aggregate_plot_rows(rows, ["split_class", "model_label", "metric"])
        high = [row for row in section_rows if row.get("metric") in higher_metrics]
        err = [row for row in section_rows if row.get("metric") in error_metrics]
        heatmap(
            outputs.figures / f"baseline_detail_{split_class}_vcbench_higher_metrics.svg",
            high,
            f"{split_class}: VCBench baseline biological metrics",
            "model_label",
            "metric",
            "mean",
        )
        heatmap(
            outputs.figures / f"baseline_detail_{split_class}_vcbench_error_metrics.svg",
            err,
            f"{split_class}: VCBench baseline error metrics",
            "model_label",
            "metric",
            "mean",
        )

    by_section_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vcbench_cellclass_summary:
        by_section_cell[row.get("split_class", "")].append(row)
    for split_class, rows in by_section_cell.items():
        for metric in ("pearson_delta", "cosine_logfc", "spearman_logfc", "deg_precision"):
            metric_rows = [row for row in rows if row.get("metric") == metric]
            plot_rows = aggregate_plot_rows(metric_rows, ["split_class", "cellclass", "model_label"])
            heatmap(
                outputs.figures / f"baseline_detail_{split_class}_cellclass_{metric}.svg",
                plot_rows,
                f"{split_class}: cellclass {metric_label(metric)}",
                "cellclass",
                "model_label",
                "mean",
            )

    matched_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched_detail_summary:
        matched_by_section[row.get("split_class", "")].append(row)
    for split_class, rows in matched_by_section.items():
        plot_rows = aggregate_plot_rows(rows, ["split_class", "model_label", "metric"])
        high = [row for row in plot_rows if row.get("metric") in {"deltaPCC_mean", "pearson_average_mean"}]
        err = [row for row in plot_rows if row.get("metric") in error_metrics or row.get("metric") == "per_gene_std_mean"]
        heatmap(
            outputs.figures / f"baseline_detail_{split_class}_matched_higher_metrics.svg",
            high,
            f"{split_class}: matched baseline biological metrics",
            "model_label",
            "metric",
            "mean",
        )
        heatmap(
            outputs.figures / f"baseline_detail_{split_class}_matched_error_metrics.svg",
            err,
            f"{split_class}: matched baseline error/dispersion metrics",
            "model_label",
            "metric",
            "mean",
        )

    group_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched_group_rows:
        group_by_section[row.get("split_class", "")].append(row)
    for split_class, rows in group_by_section.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            val = as_float(row.get("deltaPCC"))
            if val is not None:
                groups[row.get("model_label", "")].append(val)
        box_chart(
            outputs.figures / f"baseline_detail_{split_class}_matched_perturbation_deltaPCC_distribution.svg",
            dict(groups),
            f"{split_class}: per-perturbation deltaPCC",
            "deltaPCC",
            ylim=(-0.35, 1.02),
        )
        scatter_rows = [row for row in rows if as_float(row.get("deltaPCC")) is not None and as_float(row.get("rmse_average")) is not None]
        if len(scatter_rows) > 2500:
            scatter_rows = scatter_rows[:: max(1, len(scatter_rows) // 2500)]
        scatter_chart(
            outputs.figures / f"baseline_detail_{split_class}_matched_deltaPCC_vs_rmse.svg",
            scatter_rows,
            f"{split_class}: deltaPCC vs RMSE",
            "deltaPCC",
            "rmse_average",
            "model_label",
            "deltaPCC",
            "RMSE average",
        )


def write_figure_book(outputs: AnalysisOutputs, config: dict[str, Any]) -> None:
    lines = [f"# {config.get('title', 'Perturbation benchmark analysis')} figure book", ""]
    lines.append("This file embeds every SVG generated by the analysis suite.")
    lines.append("")
    for fig in sorted(outputs.figures.glob("*.svg")):
        lines.append(f"## {fig.stem}")
        lines.append("")
        lines.append(f"![{fig.stem}](figures/{fig.name})")
        lines.append("")
    (outputs.outdir / "figure_book.md").write_text("\n".join(lines), encoding="utf-8")


def split_section_figures(outputs: AnalysisOutputs, section: dict[str, Any]) -> list[Path]:
    keywords = section.get("figure_keywords") or [section["id"]]
    figures = []
    for fig in sorted(outputs.figures.glob("*.svg")):
        name = fig.name
        if any(keyword in name for keyword in keywords):
            figures.append(fig)
    return figures


def model_metric_table(rows: list[dict[str, Any]], metrics: list[str], value_key: str = "mean") -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get("metric") in metrics and as_float(row.get(value_key)) is not None]
    return aggregate_plot_rows(filtered, ["model_label", "metric"], value_key=value_key)


def best_metric_sentence(rows: list[dict[str, Any]], metric: str, value_key: str = "mean") -> str:
    metric_rows = [row for row in rows if row.get("metric") == metric and as_float(row.get(value_key)) is not None]
    if not metric_rows:
        return ""
    best = min(metric_rows, key=lambda r: safe_float(r.get(value_key))) if metric_direction(metric) == "lower" else max(metric_rows, key=lambda r: safe_float(r.get(value_key)))
    direction_text = "lowest" if metric_direction(metric) == "lower" else "highest"
    return f"{metric_label(metric)}: {best.get('model_label', '')} has the {direction_text} value ({fmt(best.get(value_key))})."


def write_report(
    outputs: AnalysisOutputs,
    config: dict[str, Any],
    comparison: list[dict[str, Any]],
    latent_availability: list[dict[str, Any]],
    baseline_availability: list[dict[str, Any]],
    external_split_summary: list[dict[str, Any]],
    external_benchmark_summary: list[dict[str, Any]],
    vcbench_metric_summary: list[dict[str, Any]],
    vcbench_cellclass_summary: list[dict[str, Any]],
    matched_detail_summary: list[dict[str, Any]],
    matched_group_summary: list[dict[str, Any]],
    matched_examples: list[dict[str, Any]],
    gene_de_availability: list[dict[str, Any]],
) -> None:
    title = config.get("title", "Perturbation benchmark analysis")
    lines = [f"# {title}", ""]
    lines.append("This regenerated report is organized by split class. Each split section contains the headline scale+pdc comparison, detailed non-scale baseline analyses when artifacts exist, DE/logFC availability, figures, and an interpretation paragraph.")
    lines.append("")
    policy = config.get("comparison_policy")
    if policy:
        lines.append("## Baseline policy")
        lines.append("")
        lines.append(policy.get("summary", "Primary baseline comparisons should use external non-scale models."))
        lines.append("")
        names = policy.get("non_scale_model_names", [])
        if names:
            lines.append("Main external baseline model set: " + ", ".join(names) + ".")
            lines.append("")
        diagnostics = policy.get("diagnostic_only", [])
        if diagnostics:
            lines.append("Diagnostic-only artifacts are not used as the main baseline comparison:")
            lines.append("")
            for name in diagnostics:
                lines.append(f"- {name}")
            lines.append("")

    if external_benchmark_summary:
        lines.append("## Headline overview")
        lines.append("")
        lines.append("| Benchmark | Splits | Mean primary | Mean best non-scale baseline | Mean margin | Win rate | Best external models |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in external_benchmark_summary:
            lines.append(
                f"| {md_cell(row['benchmark'])} | {row['n_splits']} | {fmt(row['mean_primary'])} | {fmt(row['mean_best_external'])} | {fmt(row['mean_margin'])} | {fmt(row['win_rate'])} | {md_cell(row['best_external_models'])} |"
            )
        lines.append("")

    section_configs = config.get("split_sections", [])
    if not section_configs:
        known = sorted(
            {
                row.get("split_class", "")
                for rows in (external_split_summary, vcbench_metric_summary, matched_detail_summary, gene_de_availability)
                for row in rows
                if row.get("split_class", "")
            }
        )
        section_configs = [{"id": section_id, "title": section_id} for section_id in known]

    for section in section_configs:
        section_id = section["id"]
        lines.append(f"## {section.get('title', section_id)}")
        lines.append("")
        if section.get("question"):
            lines.append(f"Question: {section['question']}")
        if section.get("object"):
            lines.append(f"Object: {section['object']}")
        if section.get("method"):
            lines.append(f"Method: {section['method']}")
        if section.get("question") or section.get("object") or section.get("method"):
            lines.append("")

        ext_rows = [row for row in external_split_summary if row.get("split_class") == section_id]
        vcb_rows = [row for row in vcbench_metric_summary if row.get("split_class") == section_id]
        vcb_cell_rows = [row for row in vcbench_cellclass_summary if row.get("split_class") == section_id]
        matched_rows = [row for row in matched_detail_summary if row.get("split_class") == section_id]
        matched_group_rows = [row for row in matched_group_summary if row.get("split_class") == section_id]
        example_rows = [row for row in matched_examples if row.get("split_class") == section_id]
        de_rows = [row for row in gene_de_availability if row.get("split_class") == section_id]

        interpretation: list[str] = []
        margins = [safe_float(row["margin"]) for row in ext_rows if as_float(row.get("margin")) is not None]
        if margins:
            wins = sum(1 for margin in margins if margin > 0)
            losses = sum(1 for margin in margins if margin < 0)
            interpretation.append(f"Headline scale+pdc margin over the best non-scale baseline averages {fmt(mean(margins))} across {len(margins)} split rows ({wins} positive, {losses} negative).")
        for metric in ("pearson_delta", "deltaPCC_mean", "cosine_logfc", "spearman_logfc", "deg_precision", "rmse_average", "rmse_average_mean"):
            sentence = best_metric_sentence(vcb_rows + matched_rows, metric)
            if sentence:
                interpretation.append(sentence)
        if de_rows:
            missing_de = [row for row in de_rows if row.get("status") != "available"]
            available_de = [row for row in de_rows if row.get("status") == "available"]
            if available_de:
                interpretation.append("Gene-level real_vs_pred log2FC scatter is available for " + ", ".join(sorted({row.get("model_label", "") for row in available_de})) + ".")
            if missing_de:
                interpretation.append("For models marked missing in the DE table, current artifacts do not contain paired real_de/pred_de CSVs; the section therefore uses VCBench logFC/DE proxy metrics or matched per-perturbation group metrics instead of fabricating a DE scatter.")
        if interpretation:
            lines.append("Interpretation:")
            lines.append("")
            for item in interpretation[:8]:
                lines.append(f"- {item}")
            lines.append("")

        if ext_rows:
            lines.append("Headline scale+pdc vs best non-scale baseline:")
            lines.append("")
            lines.append("| Benchmark | Split | Primary | Value | Best non-scale baseline | Value | Margin | Status |")
            lines.append("|---|---|---|---:|---|---:|---:|---|")
            for row in sorted(ext_rows, key=lambda r: (r.get("benchmark", ""), r.get("group", ""))):
                lines.append(
                    f"| {md_cell(row['benchmark'])} | {md_cell(row['group'])} | {md_cell(row['primary_model'])} | {fmt(row['primary_value'])} | {md_cell(row['best_external_model'])} | {fmt(row['best_external_value'])} | {fmt(row['margin'])} | {md_cell(row['status'])} |"
                )
            lines.append("")

        metric_rows = model_metric_table(
            vcb_rows,
            ["pearson_delta", "pearson_average", "rmse_average", "cosine_logfc", "spearman_logfc", "deg_precision", "deg_recall", "deg_iou"],
        )
        if metric_rows:
            lines.append("Detailed external baseline metric profile:")
            lines.append("")
            lines.append("| Model | Metric | Mean | n | Direction |")
            lines.append("|---|---|---:|---:|---|")
            for row in sorted(metric_rows, key=lambda r: (r.get("model_label", ""), r.get("metric", ""))):
                lines.append(f"| {md_cell(row['model_label'])} | {md_cell(metric_label(row['metric']))} | {fmt(row['mean'])} | {md_cell(row.get('n_plot_values',''))} | {md_cell(row.get('metric_direction',''))} |")
            lines.append("")

        matched_metric_rows = model_metric_table(
            matched_rows,
            ["deltaPCC_mean", "pearson_average_mean", "rmse_average_mean", "per_gene_std_mean", "row_rmse_to_gene_mean"],
        )
        if matched_metric_rows:
            lines.append("Matched held-out-group baseline detail:")
            lines.append("")
            lines.append("| Model | Metric | Mean | n | Direction |")
            lines.append("|---|---|---:|---:|---|")
            for row in sorted(matched_metric_rows, key=lambda r: (r.get("model_label", ""), r.get("metric", ""))):
                lines.append(f"| {md_cell(row['model_label'])} | {md_cell(metric_label(row['metric']))} | {fmt(row['mean'])} | {md_cell(row.get('n_plot_values',''))} | {md_cell(row.get('metric_direction',''))} |")
            lines.append("")

        cell_metric_rows = model_metric_table(
            vcb_cell_rows,
            ["pearson_delta", "cosine_logfc", "spearman_logfc", "deg_precision"],
        )
        if cell_metric_rows:
            lines.append("Cellclass-level baseline detail is included as heatmaps; aggregate cellclass means:")
            lines.append("")
            lines.append("| Model | Metric | Mean across cellclasses/splits | n |")
            lines.append("|---|---|---:|---:|")
            for row in sorted(cell_metric_rows, key=lambda r: (r.get("model_label", ""), r.get("metric", ""))):
                lines.append(f"| {md_cell(row['model_label'])} | {md_cell(metric_label(row['metric']))} | {fmt(row['mean'])} | {md_cell(row.get('n_plot_values',''))} |")
            lines.append("")

        if matched_group_rows:
            lines.append("Per-perturbation matched group summary:")
            lines.append("")
            lines.append("| Split | Model | Metric | Mean | Median | q25 | q75 | n |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|")
            for row in sorted(matched_group_rows, key=lambda r: (r.get("group", ""), r.get("model_label", ""), r.get("metric", ""))):
                if row.get("metric") not in {"deltaPCC", "pearson_average", "rmse_average"}:
                    continue
                lines.append(f"| {md_cell(row.get('group',''))} | {md_cell(row['model_label'])} | {md_cell(metric_label(row['metric']))} | {fmt(row['mean'])} | {fmt(row['median'])} | {fmt(row['q25'])} | {fmt(row['q75'])} | {md_cell(row.get('n',''))} |")
            lines.append("")

        if example_rows:
            lines.append("Example perturbations from matched per-perturbation analysis:")
            lines.append("")
            lines.append("| Split | Model | Bucket | Perturbation | deltaPCC | RMSE | Real cells |")
            lines.append("|---|---|---|---|---:|---:|---:|")
            for row in example_rows[:30]:
                lines.append(f"| {md_cell(row.get('group',''))} | {md_cell(row['model_label'])} | {md_cell(row['bucket'])} | {md_cell(row['perturbation'])} | {fmt(row['deltaPCC'])} | {fmt(row['rmse_average'])} | {md_cell(row['n_real_cells'])} |")
            lines.append("")

        if de_rows:
            lines.append("Gene-level DE/logFC artifact availability:")
            lines.append("")
            lines.append("| Model | Artifact | Status | Usable figure | Note |")
            lines.append("|---|---|---|---|---|")
            for row in sorted(de_rows, key=lambda r: (r.get("model_label", ""), r.get("artifact", ""))):
                lines.append(f"| {md_cell(row.get('model_label',''))} | {md_cell(row.get('artifact',''))} | {md_cell(row.get('status',''))} | {md_cell(row.get('usable_figure',''))} | {md_cell(row.get('note',''))} |")
            lines.append("")

        figs = split_section_figures(outputs, section)
        if figs:
            lines.append("Figures:")
            lines.append("")
            for fig in figs:
                lines.append(f"![{fig.stem}](figures/{fig.name})")
                lines.append("")
        if not (ext_rows or vcb_rows or vcb_cell_rows or matched_rows or matched_group_rows or de_rows or figs):
            lines.append("No compatible artifact was found for this split class in the configured inputs.")
            lines.append("")

    missing = [row for row in baseline_availability if row.get("status") == "missing"]
    if missing:
        lines.append("## Missing non-scale baselines")
        lines.append("")
        lines.append("These missing rows are reported explicitly and are not replaced by scale-family diagnostics.")
        lines.append("")
        lines.append("| Benchmark | Split | Model | Note |")
        lines.append("|---|---|---|---|")
        for row in sorted(missing, key=lambda r: (r.get("benchmark", ""), r.get("group", ""), r.get("model_label", ""))):
            lines.append(f"| {md_cell(row.get('benchmark',''))} | {md_cell(row.get('group',''))} | {md_cell(row.get('model_label',''))} | {md_cell(row.get('note') or row.get('notes',''))} |")
        lines.append("")

    lines.append("## Analysis availability")
    lines.append("")
    lines.append("| Angle | Mathematical object | Current status |")
    lines.append("|---|---|---|")
    angle_rows = [
        ("Gene-space perturbation effect", "delta expression vectors", "Available for scale+pdc Replogle cell_eval and VCBench/matched score tables."),
        ("DE recovery", "real/pred logFC, DEG overlap proxies", "Replogle has paired real_de/pred_de; VCBench baselines currently expose logFC/DE proxy metrics, not paired DE CSVs."),
        ("Context-specific response", "response by patient/cellclass/split", "Available for Kang cellclasses and Task03/04 split-level matched groups."),
        ("Distribution shift", "RMSE/pearson average/per-gene std", "Available in VCBench and matched summaries as scalar proxies; full MMD/Wasserstein requires h5ad loading."),
        ("Latent geometry", "z = E(x)", "Skipped in this dependency-free run unless embeddings are provided separately."),
    ]
    for angle, obj, status in angle_rows:
        lines.append(f"| {md_cell(angle)} | {md_cell(obj)} | {md_cell(status)} |")
    lines.append("")
    for row in latent_availability:
        lines.append(f"- {row['dataset_id']}: {row['latent_status']}; {row['note']}")
    lines.append("")
    (outputs.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_legacy(
    outputs: AnalysisOutputs,
    config: dict[str, Any],
    comparison: list[dict[str, Any]],
    latent_availability: list[dict[str, Any]],
    baseline_availability: list[dict[str, Any]],
    external_split_summary: list[dict[str, Any]],
    external_benchmark_summary: list[dict[str, Any]],
) -> None:
    title = config.get("title", "Perturbation benchmark analysis")
    lines = [f"# {title}", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("This report compares available perturbation benchmark artifacts using expression-delta and DE-recovery analyses that can run from existing CSV/JSON outputs. The primary baseline set is restricted to external non-scale models when compatible artifacts exist.")
    lines.append("")
    policy = config.get("comparison_policy")
    if policy:
        lines.append("## Baseline policy")
        lines.append("")
        lines.append(policy.get("summary", "Primary baseline comparisons should use non-scale models."))
        lines.append("")
        names = policy.get("non_scale_model_names", [])
        if names:
            lines.append("External non-scale baseline candidates in this run:")
            lines.append("")
            for name in names:
                lines.append(f"- {name}")
            lines.append("")
        diagnostics = policy.get("diagnostic_only", [])
        if diagnostics:
            lines.append("Diagnostic-only entries, not counted as non-scale baselines:")
            lines.append("")
            for name in diagnostics:
                lines.append(f"- {name}")
            lines.append("")
    lines.append("## Model comparison")
    lines.append("")
    delta = [row for row in comparison if row["metric"] in {"pearson_delta", "external_pearson_delta", "matched_delta_pcc"}]
    if delta:
        lines.append("| Benchmark | Metric | Model | Role | Mean | n groups | Source |")
        lines.append("|---|---|---|---|---:|---:|---|")
        for row in sorted(delta, key=lambda r: (r["benchmark"], r["metric"], r["role"], r["model_label"])):
            lines.append(f"| {md_cell(row['benchmark'])} | {md_cell(metric_label(row['metric']))} | {md_cell(row['model_label'])} | {md_cell(row.get('role',''))} | {fmt(row['mean'])} | {md_cell(row.get('n_groups',''))} | {md_cell(row.get('source',''))} |")
        lines.append("")
    if external_benchmark_summary:
        lines.append("## External-baseline conclusion")
        lines.append("")
        lines.append("| Benchmark | Splits | Mean primary | Mean best external | Mean margin | Win rate | Best external models |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in external_benchmark_summary:
            lines.append(
                f"| {md_cell(row['benchmark'])} | {row['n_splits']} | {fmt(row['mean_primary'])} | {fmt(row['mean_best_external'])} | {fmt(row['mean_margin'])} | {fmt(row['win_rate'])} | {md_cell(row['best_external_models'])} |"
            )
        lines.append("")
        lines.append("| Benchmark | Split | Primary | Value | Best external | Value | Margin | Status |")
        lines.append("|---|---|---|---:|---|---:|---:|---|")
        for row in sorted(external_split_summary, key=lambda r: (r["benchmark"], r.get("group", ""))):
            if as_float(row.get("margin")) is None:
                continue
            lines.append(
                f"| {md_cell(row['benchmark'])} | {md_cell(row['group'])} | {md_cell(row['primary_model'])} | {fmt(row['primary_value'])} | {md_cell(row['best_external_model'])} | {fmt(row['best_external_value'])} | {fmt(row['margin'])} | {md_cell(row['status'])} |"
            )
        lines.append("")
    missing = [row for row in baseline_availability if row.get("status") == "missing"]
    if missing:
        lines.append("## Missing External Baselines")
        lines.append("")
        lines.append("Missing rows are not replaced by prior LLM, train delta-memory, memory anchors, lambda0, or other scale-family diagnostics.")
        lines.append("")
        lines.append("| Benchmark | Split | Model | Role | Note |")
        lines.append("|---|---|---|---|---|")
        for row in sorted(missing, key=lambda r: (r.get("benchmark", ""), r.get("group", ""), r.get("model_label", ""))):
            note = row.get("note") or row.get("notes", "")
            lines.append(f"| {md_cell(row.get('benchmark',''))} | {md_cell(row.get('group',''))} | {md_cell(row.get('model_label',''))} | {md_cell(row.get('role',''))} | {md_cell(note)} |")
        lines.append("")
    lines.append("## Analysis angle availability")
    lines.append("")
    angle_rows = [
        ("Gene-space perturbation effect", "delta expression vectors from results.csv or score tables", "Available for Replogle cell_eval and VCBench external wide tables"),
        ("DE recovery", "precision@K, direction match, top100 log2FC", "Available for Replogle cell_eval outputs; external VCBench rows use deltaPCC unless DE columns are present"),
        ("Perturbation direction", "per-perturbation deltaPCC and response vectors", "Available as scalar deltaPCC; vector plots require h5ad expression loading"),
        ("Response manifold", "perturbation deltas across contexts", "Partial; current reusable report uses split/context heatmaps"),
        ("Context-specific response", "delta response by cell line, patient, split, or cell class", "Available for Replogle cell lines, Kang patients, and split-level VCBench rows"),
        ("Latent geometry", "model embeddings in obsm or external embedding table", "Optional; skipped unless embeddings are provided"),
    ]
    lines.append("| Angle | Mathematical object | v1 status |")
    lines.append("|---|---|---|")
    for angle, obj, status in angle_rows:
        lines.append(f"| {md_cell(angle)} | {md_cell(obj)} | {md_cell(status)} |")
    lines.append("")
    lines.append("## Latent module")
    lines.append("")
    for row in latent_availability:
        lines.append(f"- {row['dataset_id']}: {row['latent_status']}; {row['note']}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in sorted(outputs.figures.glob("*.svg")):
        lines.append(f"- `figures/{fig.name}`")
    lines.append("")
    (outputs.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    outputs = ensure_outputs(Path(args.outdir))

    cell_datasets: list[CellEvalDataset] = []
    all_cell_rows: list[dict[str, Any]] = []
    cell_summary: list[dict[str, Any]] = []
    summary_only: list[dict[str, Any]] = []
    kang_rows: list[dict[str, Any]] = []
    kang_patient_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    baseline_availability: list[dict[str, Any]] = []
    vcbench_metric_rows: list[dict[str, Any]] = []
    vcbench_metric_summary: list[dict[str, Any]] = []
    vcbench_cellclass_rows: list[dict[str, Any]] = []
    vcbench_cellclass_summary: list[dict[str, Any]] = []
    matched_detail_rows: list[dict[str, Any]] = []
    matched_detail_summary: list[dict[str, Any]] = []
    matched_group_rows: list[dict[str, Any]] = []
    matched_group_summary: list[dict[str, Any]] = []
    de_events: list[dict[str, Any]] = []

    for dataset_cfg in config.get("datasets", []):
        dtype = dataset_cfg.get("type")
        if dtype == "cell_eval":
            dataset = discover_cell_eval_dataset(dataset_cfg)
            cell_datasets.append(dataset)
            all_cell_rows.extend(dataset.rows)
            cell_summary.extend(summarize_cell_eval(dataset.rows))
            if not args.skip_de and dataset.de_pairs and dataset_cfg.get("include_de_events", True):
                de_events.extend(build_de_events(dataset, top_k=100))
        elif dtype == "cell_eval_summary":
            summary_only.extend(load_cell_eval_summary(dataset_cfg))
        elif dtype == "kang_pdc":
            rows, summary = load_kang_pdc(dataset_cfg)
            kang_rows.extend(rows)
            summary_only.extend(summary)
        elif dtype == "kang_summary":
            rows, summary = load_kang_summary(dataset_cfg)
            kang_patient_rows.extend(rows)
            summary_only.extend(summary)
        elif dtype == "wide_comparison":
            rows, detail, availability_rows = load_wide_comparison(dataset_cfg)
            summary_only.extend(rows)
            external_rows.extend(detail)
            baseline_availability.extend(availability_rows)
        elif dtype == "vcbench_metric_details":
            raw, summary, cell_raw, cell_summary_rows = load_vcbench_metric_details(dataset_cfg)
            vcbench_metric_rows.extend(raw)
            vcbench_metric_summary.extend(summary)
            vcbench_cellclass_rows.extend(cell_raw)
            vcbench_cellclass_summary.extend(cell_summary_rows)
        elif dtype == "matched_eval_details":
            detail_rows, detail_summary, group_rows, group_summary = load_matched_eval_details(dataset_cfg)
            matched_detail_rows.extend(detail_rows)
            matched_detail_summary.extend(detail_summary)
            matched_group_rows.extend(group_rows)
            matched_group_summary.extend(group_summary)
        else:
            raise ValueError(f"Unsupported dataset type: {dtype}")

    baseline_availability.extend(config.get("manual_baseline_availability", []))
    de_group_summary, de_category, de_gene_summary = summarize_de_events(de_events)
    comparison = comparison_rows(cell_summary, summary_only)
    external_split_summary, external_benchmark_summary = summarize_primary_vs_external(external_rows)
    latent_availability = inspect_latent_availability(config.get("datasets", []))
    gene_de_availability = build_gene_level_de_availability(config, cell_datasets)

    write_csv(outputs.tables / "cell_eval_perturbation_metrics.csv", all_cell_rows)
    write_csv(outputs.tables / "cell_eval_group_metric_summary.csv", cell_summary)
    write_csv(outputs.tables / "summary_only_metrics.csv", summary_only)
    write_csv(outputs.tables / "model_comparison_summary.csv", comparison)
    write_csv(outputs.tables / "external_baseline_metrics.csv", external_rows)
    write_csv(outputs.tables / "external_baseline_availability.csv", baseline_availability)
    write_csv(outputs.tables / "primary_vs_best_external_by_split.csv", external_split_summary)
    write_csv(outputs.tables / "external_benchmark_summary.csv", external_benchmark_summary)
    write_csv(outputs.tables / "vcbench_baseline_metric_rows.csv", vcbench_metric_rows)
    write_csv(outputs.tables / "vcbench_baseline_metric_summary.csv", vcbench_metric_summary)
    write_csv(outputs.tables / "vcbench_baseline_cellclass_rows.csv", vcbench_cellclass_rows)
    write_csv(outputs.tables / "vcbench_baseline_cellclass_summary.csv", vcbench_cellclass_summary)
    write_csv(outputs.tables / "matched_detail_metric_rows.csv", matched_detail_rows)
    write_csv(outputs.tables / "matched_detail_metric_summary.csv", matched_detail_summary)
    write_csv(outputs.tables / "matched_group_metrics.csv", matched_group_rows)
    write_csv(outputs.tables / "matched_group_metric_summary.csv", matched_group_summary)
    write_csv(outputs.tables / "gene_level_de_availability.csv", gene_de_availability)
    write_csv(outputs.tables / "kang_lambda_scores.csv", kang_rows)
    write_csv(outputs.tables / "kang_patient_summary.csv", kang_patient_rows)
    write_csv(outputs.tables / "de_top100_events.csv", de_events)
    write_csv(outputs.tables / "de_top100_group_summary.csv", de_group_summary)
    write_csv(outputs.tables / "de_top100_category_summary.csv", de_category)
    write_csv(outputs.tables / "de_top100_gene_summary.csv", de_gene_summary)
    write_csv(outputs.tables / "analysis_availability.csv", latent_availability)

    make_cell_eval_figures(outputs, cell_datasets, de_category, de_group_summary, de_events, de_gene_summary, args.top_de_gene_limit)
    make_kang_figures(outputs, kang_rows, kang_patient_rows)
    make_comparison_figures(outputs, comparison)
    make_external_figures(outputs, external_rows, external_split_summary, external_benchmark_summary)
    make_baseline_detail_figures(outputs, vcbench_metric_summary, vcbench_cellclass_summary, matched_detail_summary, matched_group_rows)
    matched_examples = write_matched_perturbation_examples(outputs, matched_group_rows)
    write_report(
        outputs,
        config,
        comparison,
        latent_availability,
        baseline_availability,
        external_split_summary,
        external_benchmark_summary,
        vcbench_metric_summary,
        vcbench_cellclass_summary,
        matched_detail_summary,
        matched_group_summary,
        matched_examples,
        gene_de_availability,
    )
    write_figure_book(outputs, config)

    outputs.manifest = {
        "config": str(config_path),
        "outdir": str(outputs.outdir),
        "tables": sorted(p.name for p in outputs.tables.glob("*.csv")),
        "figures": sorted(p.name for p in outputs.figures.glob("*.svg")),
        "n_cell_eval_rows": len(all_cell_rows),
        "n_de_events": len(de_events),
        "n_kang_lambda_rows": len(kang_rows),
        "n_external_rows": len(external_rows),
        "n_external_split_summaries": len(external_split_summary),
        "n_vcbench_metric_rows": len(vcbench_metric_rows),
        "n_vcbench_cellclass_rows": len(vcbench_cellclass_rows),
        "n_matched_group_rows": len(matched_group_rows),
        "n_comparison_rows": len(comparison),
    }
    (outputs.outdir / "run_manifest.json").write_text(json.dumps(outputs.manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote analysis report to {outputs.outdir / 'report.md'}")


if __name__ == "__main__":
    main()
