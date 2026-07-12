#!/usr/bin/env python3
"""Generate a figure book for the Xinjie Replogle analysis figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CATALOG = [
    (
        "distribution_delta_pcc_by_cell_line",
        "Gene-space perturbation effect",
        "Per-perturbation deltaPCC distributions by K562, HepG2, Jurkat, and RPE1.",
        "RPE1 has the highest mean deltaPCC (0.8190), K562 the lowest (0.6888); expression-effect direction/pattern is best recovered in RPE1.",
    ),
    (
        "mean_delta_pcc_by_cell_line",
        "Gene-space perturbation effect",
        "Cell-line mean deltaPCC bar plot.",
        "The mean view confirms RPE1 > Jurkat/HepG2 > K562 on effect-vector correlation.",
    ),
    (
        "distribution_effect_rmse_by_cell_line",
        "Gene-space perturbation effect",
        "Per-perturbation RMSE on delta/effect vectors by cell line.",
        "K562 has the lowest effect RMSE (0.0493), RPE1 the highest (0.0661); correlation and absolute error are different failure modes.",
    ),
    (
        "mean_effect_rmse_by_cell_line",
        "Gene-space perturbation effect",
        "Cell-line mean effect RMSE bar plot.",
        "K562 is most accurate in absolute effect scale, while RPE1 is least accurate despite highest deltaPCC.",
    ),
    (
        "distribution_rmse_delta_by_cell_line",
        "Gene-space perturbation effect",
        "Per-perturbation sqrt(mse_delta) distributions by cell line.",
        "This confirms the absolute delta-error ranking: K562 lowest and RPE1 highest.",
    ),
    (
        "mean_rmse_delta_by_cell_line",
        "Gene-space perturbation effect",
        "Cell-line mean sqrt(mse_delta) bar plot.",
        "The mean RMSE ranking is K562 < Jurkat < HepG2 < RPE1.",
    ),
    (
        "scatter_delta_pcc_vs_effect_rmse",
        "Evaluation metric alignment",
        "Per-perturbation deltaPCC versus effect RMSE, colored by cell line.",
        "High directional recovery does not guarantee low absolute error; RPE1 can be high deltaPCC and high RMSE.",
    ),
    (
        "scatter_delta_pcc_vs_rmse_delta",
        "Evaluation metric alignment",
        "Per-perturbation deltaPCC versus sqrt(mse_delta).",
        "This checks the same direction-vs-magnitude tension using the delta RMSE metric.",
    ),
    (
        "distribution_deg_overlap_precision_at_100_by_cell_line",
        "DE recovery",
        "Predicted/real top100 DEG overlap precision distributions by cell line.",
        "HepG2/RPE1/Jurkat are above K562 at top100; K562 has the weakest top100 DEG-set recovery.",
    ),
    (
        "distribution_deg_overlap_precision_at_200_by_cell_line",
        "DE recovery",
        "Top200 DEG overlap precision distributions by cell line.",
        "HepG2 and RPE1 improve strongly at K=200, while K562 remains lower.",
    ),
    (
        "distribution_deg_overlap_precision_at_500_by_cell_line",
        "DE recovery",
        "Top500 DEG overlap precision distributions by cell line.",
        "RPE1 (0.4412) and HepG2 (0.4333) are best at broader DEG recovery; K562 remains lowest.",
    ),
    (
        "distribution_deg_overlap_precision_at_100_200_500_by_cell_line",
        "DE recovery",
        "Combined top100/top200/top500 DEG overlap precision distributions by cell line.",
        "Increasing K helps HepG2 and RPE1, so the model often recovers broader DEG neighborhoods even when top100 is imperfect.",
    ),
    (
        "mean_deg_overlap_precision_at_100_by_cell_line",
        "DE recovery",
        "Mean precision@100 by cell line.",
        "HepG2 is highest at top100 (0.3443), RPE1 is close (0.3390), and K562 is lowest.",
    ),
    (
        "mean_deg_overlap_precision_at_200_by_cell_line",
        "DE recovery",
        "Mean precision@200 by cell line.",
        "HepG2 (0.3977) and RPE1 (0.3966) are essentially tied and ahead of Jurkat/K562.",
    ),
    (
        "mean_deg_overlap_precision_at_500_by_cell_line",
        "DE recovery",
        "Mean precision@500 by cell line.",
        "RPE1 is best (0.4412), HepG2 second (0.4333), and K562 lowest.",
    ),
    (
        "scatter_delta_pearson_vs_deg_overlap_precision_at_100",
        "Evaluation metric alignment",
        "Per-perturbation deltaPCC versus DEG overlap precision@100.",
        "The positive relationship (r = 0.7335) shows deltaPCC tracks DEG recovery, but cell-line residuals remain.",
    ),
    (
        "scatter_delta_pearson_vs_deg_overlap_precision_at_200",
        "Evaluation metric alignment",
        "Per-perturbation deltaPCC versus DEG overlap precision@200.",
        "Directional effect recovery tracks broader DEG recovery, with residual cell-line-specific behavior.",
    ),
    (
        "scatter_delta_pearson_vs_deg_overlap_precision_at_500",
        "Evaluation metric alignment",
        "Per-perturbation deltaPCC versus DEG overlap precision@500.",
        "Higher deltaPCC generally means better broad DEG coverage, supporting the use of both metrics.",
    ),
    (
        "heatmap_top_deg_direction_match_rate_by_cell_line_k",
        "DE recovery",
        "Direction-match rate on real topK DEGs for K=100/200/500 by cell line.",
        "Top-DEG direction is usually correct: top100 direction match is 0.9186-0.9504, with RPE1 highest.",
    ),
    (
        "heatmap_top_deg_abs_lfc_spearman_by_cell_line_k",
        "DE recovery",
        "Spearman correlation between real and predicted absolute logFC ranks on real topK DEGs.",
        "Magnitude ranking improves with larger K; top100 ranking is weaker, especially in HepG2.",
    ),
    (
        "heatmap_top_deg_log2_amplitude_bias_by_cell_line_k",
        "DE recovery",
        "Mean log2 amplitude bias log2(|pred log2FC| / |real log2FC|) on real topK DEGs.",
        "Bias is negative in all cell lines and K values, so predicted DE effects are systematically under-amplified.",
    ),
    (
        "distribution_top100_log2_amplitude_ratio_by_cell_line",
        "DE recovery",
        "Distribution of log2 amplitude ratio on real top100 DEGs.",
        "The distribution is shifted below zero in every cell line; amplitude shrinkage is the dominant residual error.",
    ),
    (
        "scatter_top100_real_vs_pred_signed_log2fc",
        "DE recovery",
        "Real versus predicted signed log2FC for real top100 DEG events.",
        "Most points stay in the correct sign quadrant but shrink toward zero; direction is mostly right while magnitude is compressed.",
    ),
    (
        "stacked_top100_direction_magnitude_categories_by_cell_line",
        "DE recovery",
        "Fractions of real top100 DEG events classified as wrong direction, correct-under, correct-matched, or correct-over.",
        "Correct-under dominates all cell lines; RPE1 has low wrong-direction rate but high under-amplification.",
    ),
    (
        "heatmap_gene_top100_direction_match_rate_top30_by_cell_line",
        "Biological alignment / DE recovery",
        "Direction-match rate by cell line for the 30 most frequent real top100 genes.",
        "High-frequency DEG directions are mostly recovered; direction failure is gene/cell-specific rather than global.",
    ),
    (
        "heatmap_gene_top100_wrong_direction_fraction_top30_by_cell_line",
        "Biological alignment / DE recovery",
        "Wrong-direction fraction by cell line for the same top30 frequent real top100 genes.",
        "Wrong direction is generally low, with relatively higher local failure hotspots in K562.",
    ),
    (
        "heatmap_gene_top100_log2_amplitude_bias_top30_by_cell_line",
        "Biological alignment / DE recovery",
        "Mean log2 amplitude bias by cell line for the same top30 frequent real top100 genes.",
        "Most high-frequency genes are under-amplified, strongest in RPE1; direction can be right while effect size is too small.",
    ),
    (
        "bar_gene_top100_highest_wrong_direction_fraction",
        "Biological alignment / DE recovery",
        "Genes with the highest wrong-direction fraction among genes with enough top100 occurrences.",
        "This highlights gene-specific failure cases for mechanistic follow-up and should be used as a diagnostic figure.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="benchmark/workspace/replogle_main_corrected_fullcell_remote_ep129_20260623/analyzing",
    )
    parser.add_argument(
        "--outdir",
        default="benchmark/workspace/replogle_xinjie_figure_book_20260629",
    )
    return parser.parse_args()


def copy_figure(src_figures: Path, dst_figures: Path, stem: str) -> dict[str, str]:
    copied: dict[str, str] = {}
    for suffix in [".png", ".pdf"]:
        src = src_figures / f"{stem}{suffix}"
        if src.exists():
            dst = dst_figures / src.name
            shutil.copy2(src, dst)
            copied[suffix.lstrip(".")] = f"figures/{dst.name}"
    return copied


def write_external_baseline_availability(tables: Path) -> pd.DataFrame:
    tables.mkdir(parents=True, exist_ok=True)
    external = pd.DataFrame(
        [
            {
                "model": model,
                "replogle_artifact_status": "not_found_in_current_workspace",
                "interpretation": (
                    "Do not claim a Replogle model comparison until compatible non-scale "
                    "score rows or h5ad predictions are generated for this model."
                ),
            }
            for model in ["CPA", "BioLord", "State", "GEARS", "scLambda"]
        ]
    )
    external.to_csv(tables / "replogle_external_baseline_availability.csv", index=False)
    return external


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir)
    src_figures = source / "figures"
    outdir = Path(args.outdir)
    dst_figures = outdir / "figures"
    tables = outdir / "tables"
    dst_figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    external = write_external_baseline_availability(tables)

    rows = []
    lines = [
        "# Replogle Xinjie Full-Cell Figure Book",
        "",
        "Purpose: include the complete Xinjie Replogle diagnostic figure set for `SCALE+PDC` from the corrected full-cell run.",
        "",
        f"Source: `{source}`",
        "",
        "External-baseline availability: compatible Replogle CPA/BioLord/State/GEARS/scLambda score rows or h5ad predictions were not found in the current workspace. Therefore this Replogle document is a single-model biological diagnostic figure book, not a paper-facing model-comparison claim.",
        "",
        "Integrated conclusion: the Replogle scale+pdc run recovers perturbation-effect direction well, especially in RPE1, and top-DEG direction is usually correct across cell lines. The main residual error is amplitude calibration: predicted signed log2FC values are systematically shrunk toward zero, so many real top100 DEG events are `correct_under` rather than direction-wrong.",
        "",
        "## External Baseline Availability",
        "",
        "Question: can Replogle be used for a direct paper-facing comparison against the approved non-scale baselines?",
        "",
        "Mathematical object: same split/protocol model predictions or score rows for CPA, BioLord, State, GEARS, and scLambda.",
        "",
        "Concrete procedure: scan the current workspace for compatible Replogle score rows or h5ad predictions from the approved external baselines.",
        "",
        "Result interpretation: because no compatible approved non-scale Replogle artifacts are currently available, Replogle is kept as biological diagnostics only. Base-model, memory, PDC-sweep, and scale-family artifacts are not substituted as baselines.",
        "",
        "| Model | Replogle artifact status | Interpretation |",
        "|---|---|---|",
    ]
    for _, row in external.iterrows():
        lines.append(f"| {row['model']} | {row['replogle_artifact_status']} | {row['interpretation']} |")

    lines += [
        "",
        "## Figure Index",
        "",
        "| Figure | Analysis angle | What it plots | Main conclusion |",
        "|---|---|---|---|",
    ]

    for stem, angle, what, conclusion in CATALOG:
        copied = copy_figure(src_figures, dst_figures, stem)
        rows.append(
            {
                "figure": stem,
                "analysis_angle": angle,
                "what_it_plots": what,
                "main_conclusion": conclusion,
                "png": copied.get("png", ""),
                "pdf": copied.get("pdf", ""),
                "present": bool(copied),
            }
        )
        lines.append(f"| `{stem}` | {angle} | {what} | {conclusion} |")

    lines += ["", "## Figures", ""]
    current_angle = ""
    for row in rows:
        if row["analysis_angle"] != current_angle:
            current_angle = row["analysis_angle"]
            lines += [f"### {current_angle}", ""]
        lines += [
            f"#### `{row['figure']}`",
            "",
            f"What it plots: {row['what_it_plots']}",
            "",
            f"Interpretation: {row['main_conclusion']}",
            "",
        ]
        if row["png"]:
            lines += [f"![{row['figure']}]({row['png']})", ""]
        else:
            lines += ["Missing PNG in source directory.", ""]
        if row["pdf"]:
            lines += [f"PDF: [{row['figure']}.pdf]({row['pdf']})", ""]

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source),
                "n_catalog_figures": len(CATALOG),
                "n_present_png": sum(1 for row in rows if row["png"]),
                "n_present_pdf": sum(1 for row in rows if row["pdf"]),
                "n_comparison_rows": 0,
                "n_comparison_figures": 0,
                "external_baseline_availability": "tables/replogle_external_baseline_availability.csv",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Replogle Xinjie figure book to {outdir}")


if __name__ == "__main__":
    main()
