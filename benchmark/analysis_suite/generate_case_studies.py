#!/usr/bin/env python3
"""Generate biological case-study figures from paired-DE benchmark tables."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["SCALE+PDC", "State", "scLambda", "GEARS", "CPA", "BioLord"]
MODEL_COLORS = {
    "SCALE+PDC": "#C25759",
    "State": "#6C8EBF",
    "scLambda": "#8C6D31",
    "GEARS": "#B696B6",
    "CPA": "#599CB4",
    "BioLord": "#5B8C5A",
    "True": "#222222",
}
CASE_COLORS = {
    "IFN immune response": "#C25759",
    "Combinatorial perturbation": "#6C8EBF",
    "Developmental response": "#5B8C5A",
}
ISG_GENES = {
    "IFIT1",
    "IFIT2",
    "IFIT3",
    "ISG15",
    "MX1",
    "MX2",
    "OAS1",
    "OAS2",
    "OAS3",
    "OASL",
    "RSAD2",
    "CXCL10",
    "IRF7",
    "STAT1",
    "STAT2",
    "IFI6",
    "IFI27",
    "IFI35",
    "IFI44",
    "IFI44L",
    "IFIH1",
    "DDX58",
    "GBP1",
    "GBP5",
    "HERC5",
    "USP18",
    "XAF1",
    "BST2",
    "IFITM1",
    "IFITM2",
    "IFITM3",
}
NORMAN_CASES = ["CBL_UBASH3A", "FOXA1_HOXB9", "PTPN12_ZBTB25"]
ZSCAPE_CASES = ["smo", "tbxta", "tbx16-tbx16l"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-dir",
        default="benchmark/workspace/paired_de_external_baselines_plus_scale_pdc_boxjitter_20260629",
    )
    parser.add_argument(
        "--consolidated-dir",
        default="benchmark/workspace/final_paper_ready_figure_book_20260629",
    )
    parser.add_argument("--outdir", default="benchmark/workspace/biological_case_studies_20260630")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def normalize_target(split_class: str, target: object) -> str:
    text = str(target)
    if split_class == "kang_task05_lopo_patients":
        if "ifn" in text.lower() or text.upper() == "IFNB":
            return "IFNB"
    if split_class == "norman_task07_heldout_combo_genes":
        return text.replace("+", "_")
    return text


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else 99, model)


def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    figures = outdir / "figures"
    tables = outdir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def add_identity(ax: plt.Axes, values: pd.DataFrame) -> None:
    vals = pd.concat([values["real_log2fc"], values["pred_log2fc"]]).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return
    lo = float(vals.quantile(0.01))
    hi = float(vals.quantile(0.99))
    pad = max(1.0, (hi - lo) * 0.08)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#444444", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def kang_case(
    events: pd.DataFrame,
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    figures: Path,
    tables: Path,
) -> list[dict[str, str]]:
    sub = events[events["split_class"] == "kang_task05_lopo_patients"].copy()
    sub["target_norm"] = sub["target"].map(lambda x: normalize_target("kang_task05_lopo_patients", x))
    sub = sub[sub["target_norm"] == "IFNB"].copy()
    sub["is_isg"] = sub["feature"].astype(str).str.upper().isin(ISG_GENES)
    models = sorted(sub["model_label"].dropna().unique(), key=model_sort_key)

    real_sig = real_de[real_de["split_class"] == "kang_task05_lopo_patients"].copy()
    pred_sig = pred_de[pred_de["split_class"] == "kang_task05_lopo_patients"].copy()
    for frame in (real_sig, pred_sig):
        frame["target_norm"] = frame["target"].map(lambda x: normalize_target("kang_task05_lopo_patients", x))
        frame["is_isg"] = frame["feature"].astype(str).str.upper().isin(ISG_GENES)
        frame["signature_delta"] = frame["target_mean"].astype(float) - frame["reference_mean"].astype(float)
    key_cols = ["model_label", "split", "cellclass", "target_norm", "context", "feature"]
    sig = real_sig[real_sig["is_isg"] & (real_sig["target_norm"] == "IFNB")][key_cols + ["signature_delta"]].merge(
        pred_sig[pred_sig["is_isg"] & (pred_sig["target_norm"] == "IFNB")][key_cols + ["signature_delta"]],
        on=key_cols,
        suffixes=("_real", "_pred"),
        how="inner",
    )
    signature = (
        sig.groupby("model_label", as_index=False)
        .agg(
            real_signature_shift=("signature_delta_real", "mean"),
            pred_signature_shift=("signature_delta_pred", "mean"),
            n_isg_events=("feature", "size"),
            n_isg_genes=("feature", "nunique"),
        )
    )
    signature["model_label"] = pd.Categorical(signature["model_label"], categories=models, ordered=True)
    signature = signature.sort_values("model_label")
    signature.to_csv(tables / "kang_ifn_isg_signature_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(signature))
    width = 0.36
    ax.bar(x - width / 2, signature["real_signature_shift"], width, label="True ISG shift", color="#444444", alpha=0.76)
    ax.bar(x + width / 2, signature["pred_signature_shift"], width, label="Predicted ISG shift", color="#C25759", alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(signature["model_label"], rotation=30, ha="right")
    ax.set_ylabel("Mean expression shift on curated ISG genes")
    ax.set_title("Kang IFN-beta immune-response signature shift")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    save(fig, figures / "kang_ifn_isg_signature_shift.png")

    top_features = (
        sub.groupby("feature", as_index=False)["abs_real_log2fc"]
        .mean()
        .sort_values("abs_real_log2fc", ascending=False)
        .head(30)["feature"]
        .tolist()
    )
    scatter = (
        sub[sub["feature"].isin(top_features)]
        .groupby(["model_label", "feature"], as_index=False)
        .agg(real_log2fc=("real_log2fc", "mean"), pred_log2fc=("pred_log2fc", "mean"), direction_match=("direction_match", "mean"))
    )
    scatter.to_csv(tables / "kang_ifn_topdeg_feature_scatter.csv", index=False)
    ncols = 3
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        rows = scatter[scatter["model_label"] == model]
        ax.scatter(
            rows["real_log2fc"],
            rows["pred_log2fc"],
            s=32,
            alpha=0.78,
            color=MODEL_COLORS.get(model, "#777777"),
            edgecolors="white",
            linewidths=0.35,
        )
        add_identity(ax, rows)
        ax.set_title(model)
        ax.set_xlabel("True top-DE log2FC")
        ax.set_ylabel("Predicted top-DE log2FC")
        ax.grid(alpha=0.18)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    fig.suptitle("Kang IFN-beta top real-DE gene logFC scatter", y=1.01)
    save(fig, figures / "kang_ifn_topdeg_logfc_scatter.png")

    direction = (
        sub.assign(gene_set=np.where(sub["is_isg"], "curated ISG genes", "all real top100 DE genes"))
        .groupby(["model_label", "gene_set"], as_index=False)
        .agg(direction_accuracy=("direction_match", "mean"), n_events=("feature", "size"))
    )
    direction_all = (
        sub.groupby("model_label", as_index=False)
        .agg(direction_accuracy=("direction_match", "mean"), n_events=("feature", "size"))
        .assign(gene_set="all real top100 DE genes")
    )
    direction_isg = direction[direction["gene_set"] == "curated ISG genes"]
    direction = pd.concat([direction_all, direction_isg], ignore_index=True)
    direction.to_csv(tables / "kang_ifn_direction_accuracy.csv", index=False)
    pivot = direction.pivot(index="model_label", columns="gene_set", values="direction_accuracy").reindex(models)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(pivot))
    labels = ["all real top100 DE genes", "curated ISG genes"]
    colors = ["#6C8EBF", "#C25759"]
    for idx, label in enumerate(labels):
        ax.bar(x + (idx - 0.5) * 0.34, pivot[label], width=0.32, color=colors[idx], alpha=0.82, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Direction-correct fraction")
    ax.set_title("Kang IFN-beta direction agreement by model")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    save(fig, figures / "kang_ifn_direction_accuracy.png")

    return [
        {
            "figure": "figures/kang_ifn_isg_signature_shift.png",
            "title": "Kang IFN/immune-response signature shift",
            "caption": "Mean true versus predicted expression shift (`target_mean - matched_control_mean`) on curated ISG genes among paired top real-DE events.",
        },
        {
            "figure": "figures/kang_ifn_topdeg_logfc_scatter.png",
            "title": "Kang IFN top real-DE logFC scatter",
            "caption": "Feature-level true versus predicted log2FC for the strongest IFN-beta response genes, faceted by model.",
        },
        {
            "figure": "figures/kang_ifn_direction_accuracy.png",
            "title": "Kang IFN direction accuracy",
            "caption": "Direction-correct fraction for all real top100 DE genes and the curated ISG subset.",
        },
    ]


def norman_case(events: pd.DataFrame, metrics: pd.DataFrame, figures: Path, tables: Path) -> list[dict[str, str]]:
    sub = events[events["split_class"] == "norman_task07_heldout_combo_genes"].copy()
    sub["target_norm"] = sub["target"].map(lambda x: normalize_target("norman_task07_heldout_combo_genes", x))
    available = set(sub["target_norm"])
    cases = [case for case in NORMAN_CASES if case in available]
    if len(cases) < 3:
        extra = (
            sub.groupby("target_norm")["abs_real_log2fc"].mean().sort_values(ascending=False).index.tolist()
        )
        for target in extra:
            if target not in cases:
                cases.append(target)
            if len(cases) >= 3:
                break
    cases = cases[:3]
    models = sorted(sub["model_label"].dropna().unique(), key=model_sort_key)
    rows = sub[sub["target_norm"].isin(cases)].copy()
    rows.to_csv(tables / "norman_combo_selected_topdeg_events.csv", index=False)

    fig, axes = plt.subplots(1, len(cases), figsize=(5.1 * len(cases), 4.7), squeeze=False)
    for ax, target in zip(axes.ravel(), cases):
        target_rows = rows[rows["target_norm"] == target]
        for model in models:
            m = target_rows[target_rows["model_label"] == model]
            ax.scatter(
                m["real_log2fc"],
                m["pred_log2fc"],
                s=18,
                alpha=0.55,
                label=model,
                color=MODEL_COLORS.get(model, "#777777"),
                edgecolors="none",
            )
        add_identity(ax, target_rows)
        ax.set_title(target)
        ax.set_xlabel("True top-DE log2FC")
        ax.set_ylabel("Predicted top-DE log2FC")
        ax.grid(alpha=0.18)
    axes.ravel()[-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Norman combinatorial perturbation top-DE recovery", y=1.02)
    save(fig, figures / "norman_combo_topdeg_logfc_scatter.png")

    metric_rows = metrics[metrics["split_class"] == "norman_task07_heldout_combo_genes"].copy()
    metric_rows["target_norm"] = metric_rows["target"].map(lambda x: normalize_target("norman_task07_heldout_combo_genes", x))
    metric_rows = metric_rows[metric_rows["target_norm"].isin(cases)]
    heat = metric_rows.pivot_table(index="model_label", columns="target_norm", values="direction_agreement_topk", aggfunc="mean").reindex(models)
    heat.to_csv(tables / "norman_combo_direction_heatmap_values.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(heat.values.astype(float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Norman combo direction-correct fraction on real top100 DE genes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Direction agreement")
    save(fig, figures / "norman_combo_direction_heatmap.png")

    return [
        {
            "figure": "figures/norman_combo_topdeg_logfc_scatter.png",
            "title": "Norman combinatorial perturbation scatter",
            "caption": "True versus predicted top real-DE log2FC for three representative gene-pair perturbations.",
        },
        {
            "figure": "figures/norman_combo_direction_heatmap.png",
            "title": "Norman combo direction heatmap",
            "caption": "Direction-correct fraction by model and selected gene-pair perturbation.",
        },
    ]


def zscape_case(events: pd.DataFrame, figures: Path, tables: Path) -> list[dict[str, str]]:
    sub = events[events["split_class"] == "zscape_seen_cell_unseen_gene"].copy()
    sub = sub[sub["target"].isin(ZSCAPE_CASES)].copy()
    models = sorted(sub["model_label"].dropna().unique(), key=model_sort_key)
    top_rows = []
    for target, target_rows in sub.groupby("target"):
        top_features = (
            target_rows.groupby("feature", as_index=False)["abs_real_log2fc"]
            .mean()
            .sort_values("abs_real_log2fc", ascending=False)
            .head(7)["feature"]
            .tolist()
        )
        top_rows.append(target_rows[target_rows["feature"].isin(top_features)])
    selected = pd.concat(top_rows, ignore_index=True)
    heat_rows = []
    for (target, feature), g in selected.groupby(["target", "feature"]):
        row = {"target_feature": f"{target} | {feature}", "target": target, "feature": feature}
        row["True"] = g["real_log2fc"].mean()
        for model in models:
            row[model] = g[g["model_label"] == model]["pred_log2fc"].mean()
        heat_rows.append(row)
    heat = pd.DataFrame(heat_rows)
    heat["sort_abs"] = heat["True"].abs()
    heat = heat.sort_values(["target", "sort_abs"], ascending=[True, False]).drop(columns=["sort_abs"])
    heat.to_csv(tables / "zscape_developmental_marker_proxy_heatmap_values.csv", index=False)
    columns = ["True", *models]
    values = heat[columns].astype(float).values
    vmax = float(np.nanpercentile(np.abs(values), 95)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1.0)
    fig_h = max(5.5, 0.33 * len(heat) + 1.5)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels(heat["target_feature"], fontsize=7)
    ax.set_title("ZSCAPE developmental target response: real/predicted top-DE logFC")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("log2FC")
    save(fig, figures / "zscape_developmental_marker_logfc_heatmap.png")
    return [
        {
            "figure": "figures/zscape_developmental_marker_logfc_heatmap.png",
            "title": "ZSCAPE developmental marker-response heatmap",
            "caption": "Real and predicted log2FC for top real-DE genes under comparable developmental targets smo, tbxta, and tbx16-tbx16l.",
        }
    ]


def jaccard_case(metrics: pd.DataFrame, figures: Path, tables: Path) -> list[dict[str, str]]:
    rows = metrics.copy()
    rows["jaccard_topk"] = rows["de_overlap_at_k"] / (2.0 - rows["de_overlap_at_k"])
    summary = (
        rows.groupby(["split_class", "model_label"], as_index=False)
        .agg(jaccard_topk=("jaccard_topk", "mean"), n_targets=("target", "size"))
    )
    summary.to_csv(tables / "de_top100_jaccard_summary.csv", index=False)
    split_order = [
        "kang_task05_lopo_patients",
        "norman_task07_heldout_combo_genes",
        "zscape_seen_cell_unseen_gene",
    ]
    model_order = [m for m in MODEL_ORDER if m in set(summary["model_label"])]
    heat = summary.pivot(index="split_class", columns="model_label", values="jaccard_topk").reindex(split_order)[model_order]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    im = ax.imshow(heat.values.astype(float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.35, np.nanmax(heat.values)))
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([s.replace("_", " ") for s in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Top100 predicted/real DE gene-set Jaccard")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard from top100 overlap")
    save(fig, figures / "de_top100_jaccard_heatmap.png")
    return [
        {
            "figure": "figures/de_top100_jaccard_heatmap.png",
            "title": "DE gene-set Jaccard heatmap",
            "caption": "Model-level top100 predicted-vs-real DE gene-set overlap converted from overlap@K to Jaccard.",
        }
    ]


def delta_pca_case(real_de: pd.DataFrame, pred_de: pd.DataFrame, figures: Path, tables: Path) -> list[dict[str, str]]:
    real = real_de.copy()
    pred = pred_de.copy()
    for df in (real, pred):
        df["target_norm"] = [normalize_target(sc, target) for sc, target in zip(df["split_class"], df["target"])]
        df["case_key"] = df["split_class"] + "|" + df["target_norm"].astype(str)
        df["delta"] = df["target_mean"].astype(float) - df["reference_mean"].astype(float)
    allowed = {
        "kang_task05_lopo_patients|IFNB",
        *{f"norman_task07_heldout_combo_genes|{x}" for x in NORMAN_CASES},
        *{f"zscape_seen_cell_unseen_gene|{x}" for x in ZSCAPE_CASES},
    }
    real = real[real["case_key"].isin(allowed)]
    pred = pred[pred["case_key"].isin(allowed)]
    rows = []
    feature_set = sorted(set(real["feature"].astype(str)) | set(pred["feature"].astype(str)))
    if not feature_set:
        return []
    feature_index = {gene: idx for idx, gene in enumerate(feature_set)}

    def append_vectors(df: pd.DataFrame, source: str) -> None:
        group_cols = ["split_class", "target_norm", "model_label"] if source == "Pred" else ["split_class", "target_norm"]
        for key, g in df.groupby(group_cols):
            vec = np.zeros(len(feature_set), dtype=float)
            counts = np.zeros(len(feature_set), dtype=float)
            for _, row in g.iterrows():
                idx = feature_index[str(row["feature"])]
                vec[idx] += float(row["delta"])
                counts[idx] += 1.0
            nz = counts > 0
            vec[nz] /= counts[nz]
            if source == "Pred":
                split_class, target_norm, model = key
            else:
                split_class, target_norm = key
                model = "True"
            rows.append({"split_class": split_class, "target_norm": target_norm, "model_label": model, "source": source, "vector": vec})

    append_vectors(real, "True")
    append_vectors(pred, "Pred")
    matrix = np.vstack([row["vector"] for row in rows])
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    coords = u[:, :2] * s[:2]
    denom = float((s**2).sum()) if s.size else 1.0
    explained = (s[:2] ** 2 / denom) if denom > 0 else np.zeros(2)
    out_rows = []
    for idx, row in enumerate(rows):
        if row["split_class"] == "kang_task05_lopo_patients":
            case_group = "IFN immune response"
        elif row["split_class"] == "norman_task07_heldout_combo_genes":
            case_group = "Combinatorial perturbation"
        else:
            case_group = "Developmental response"
        out_rows.append(
            {
                "split_class": row["split_class"],
                "target_norm": row["target_norm"],
                "case_group": case_group,
                "model_label": row["model_label"],
                "source": row["source"],
                "PC1": coords[idx, 0],
                "PC2": coords[idx, 1] if coords.shape[1] > 1 else 0.0,
            }
        )
    coords_df = pd.DataFrame(out_rows)
    coords_df.to_csv(tables / "response_delta_pca_proxy_coordinates.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.9), sharex=True, sharey=True)
    ax = axes[0]
    for model, g in coords_df.groupby("model_label"):
        marker = "*" if model == "True" else "o"
        size = 150 if model == "True" else 54
        ax.scatter(
            g["PC1"],
            g["PC2"],
            s=size,
            marker=marker,
            color=MODEL_COLORS.get(model, "#777777"),
            alpha=0.84,
            edgecolors="white",
            linewidths=0.5,
            label=model,
        )
    for _, row in coords_df.iterrows():
        if row["model_label"] in {"True", "SCALE+PDC"}:
            label = str(row["target_norm"])
            if len(label) > 12:
                label = label[:11] + "."
            ax.text(row["PC1"], row["PC2"], label, fontsize=7, alpha=0.75)
    ax.set_title("Colored by model/source")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)" if explained.size > 1 else "PC2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncols=2)

    ax = axes[1]
    for group, g in coords_df.groupby("case_group"):
        ax.scatter(
            g["PC1"],
            g["PC2"],
            s=np.where(g["source"] == "True", 132, 52),
            marker="o",
            color=CASE_COLORS.get(group, "#777777"),
            alpha=0.78,
            edgecolors=np.where(g["source"] == "True", "#111111", "white"),
            linewidths=np.where(g["source"] == "True", 1.1, 0.45),
            label=group,
        )
    for _, row in coords_df.iterrows():
        if row["model_label"] in {"True", "SCALE+PDC"}:
            label = str(row["target_norm"])
            if len(label) > 12:
                label = label[:11] + "."
            ax.text(row["PC1"], row["PC2"], label, fontsize=7, alpha=0.75)
    ax.set_title("Colored by perturbation/response class")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("Expression-response delta PCA proxy on top real-DE genes", y=1.02)
    save(fig, figures / "response_delta_pca_proxy.png")
    return [
        {
            "figure": "figures/response_delta_pca_proxy.png",
            "title": "Response delta PCA proxy",
            "caption": "PCA of top-DE response deltas `target_mean - matched_control_mean`; left panel is colored by model/source, right panel by immune/combo/developmental response class. This is an expression-response manifold proxy, not latent UMAP.",
        }
    ]


def write_report(outdir: Path, entries: list[dict[str, str]], notes: list[str]) -> None:
    lines = [
        "# Biological Case Study Figure Book",
        "",
        "Purpose: add paper-facing biological case studies on top of the model-level benchmark figures.",
        "",
        "All figures use existing paired-DE tables. Strict pathway enrichment is not run here because no local GMT/pathway file was provided.",
        "",
        "## Figures",
        "",
    ]
    for entry in entries:
        lines += [
            f"### {entry['title']}",
            "",
            f"![{Path(entry['figure']).stem}]({entry['figure']})",
            "",
            f"Interpretation: {entry['caption']}",
            "",
        ]
    if notes:
        lines += ["## Notes", ""]
        for note in notes:
            lines.append(f"- {note}")
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_into_consolidated(outdir: Path, consolidated: Path, entries: list[dict[str, str]]) -> None:
    if not consolidated.exists():
        return
    figdir = consolidated / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    report = consolidated / "report.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    insert = [
        "## Biological Case Studies",
        "",
        "These figures provide case-study views requested for immune response, combinatorial perturbation, developmental response, DE gene-set overlap, and response-manifold structure.",
        "",
    ]
    for entry in entries:
        src = outdir / entry["figure"]
        dst = figdir / src.name
        shutil.copy2(src, dst)
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, figdir / pdf.name)
        insert += [
            f"### {entry['title']}",
            "",
            f"![{src.stem}](figures/{src.name})",
            "",
            f"Interpretation: {entry['caption']}",
            "",
        ]
    block = "\n".join(insert)
    marker = "## Provenance"
    text = re.sub(r"\n## Biological Case Studies\n.*?(?=\n## Provenance|\Z)", "\n", text, flags=re.S)
    if marker in text:
        text = text.replace(marker, block + "\n" + marker)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    report.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paired = Path(args.paired_dir)
    outdir = Path(args.outdir)
    figures, tables = ensure_dirs(outdir)
    events = pd.read_csv(paired / "tables" / "paired_de_top100_events.csv")
    metrics = pd.read_csv(paired / "tables" / "paired_de_target_metrics.csv")
    real_de = pd.read_csv(paired / "tables" / "paired_real_de_top100.csv")
    pred_de = pd.read_csv(paired / "tables" / "paired_pred_de_top100.csv")
    entries: list[dict[str, str]] = []
    notes = [
        "ZSCAPE feature names in the paired-DE table did not match the curated developmental marker list directly; the ZSCAPE heatmap therefore uses top real-DE genes for comparable developmental targets as a marker-response proxy.",
        "DE gene-set Jaccard is derived from existing overlap@K as Jaccard = overlap_fraction / (2 - overlap_fraction) for K=100.",
        "Response delta PCA uses top real-DE gene mean shifts from paired-DE tables, not model latent embeddings.",
    ]
    entries.extend(kang_case(events, real_de, pred_de, figures, tables))
    entries.extend(norman_case(events, metrics, figures, tables))
    entries.extend(zscape_case(events, figures, tables))
    entries.extend(jaccard_case(metrics, figures, tables))
    entries.extend(delta_pca_case(real_de, pred_de, figures, tables))
    write_report(outdir, entries, notes)
    copy_into_consolidated(outdir, Path(args.consolidated_dir), entries)
    (outdir / "manifest.json").write_text(
        json.dumps({"paired_dir": str(paired), "figures": [entry["figure"] for entry in entries], "notes": notes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote biological case studies to {outdir}")


if __name__ == "__main__":
    main()
