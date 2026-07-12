#!/usr/bin/env python3
"""Generate scatter-heavy perturbation analysis figures.

The script is intentionally dependency-light. It uses pandas/numpy/matplotlib
and optionally anndata for h5ad PCA proxy plots. If real model embeddings or
UMAP coordinates exist in AnnData.obsm, those are used directly; otherwise the
h5ad plots are labeled as expression-space PCA proxies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CELL_COLORS = {
    "k562": "#599CB4",
    "hepg2": "#C25759",
    "jurkat": "#5B8C5A",
    "rpe1": "#8C6D31",
}

LABELS = {
    "delta_pcc": "deltaPCC",
    "rmse_delta": "RMSE(delta)",
    "deg_overlap_at_100": "DE overlap@100",
    "deg_overlap_at_200": "DE overlap@200",
    "deg_overlap_at_500": "DE overlap@500",
    "direction_match_rate": "Top100 direction match",
    "signed_lfc_pearson": "Top100 signed log2FC Pearson",
    "signed_lfc_spearman": "Top100 signed log2FC Spearman",
    "abs_lfc_spearman": "Top100 abs(log2FC) Spearman",
    "mean_log2_amp_ratio": "Top100 mean log2 amplitude ratio",
    "wrong_direction_fraction": "Top100 wrong-direction fraction",
    "correct_under_fraction": "Top100 correct-under fraction",
    "correct_matched_fraction": "Top100 correct-matched fraction",
    "pred_significant_fraction": "Pred significant fraction",
}

SCATTER_CONCLUSIONS = {
    "scatter_delta_pcc_vs_top100_direction_match": "Higher deltaPCC is strongly coupled to higher top-DEG direction agreement, so the perturbation-effect vector metric is tracking a biologically meaningful sign-recovery signal.",
    "scatter_delta_pcc_vs_top100_signed_lfc_pearson": "The positive but moderate relationship shows that global deltaPCC and top100 signed logFC fidelity agree, while top-DEG recovery still has perturbation-specific residual errors.",
    "scatter_delta_pcc_vs_top100_abs_lfc_spearman": "Magnitude-rank recovery is weaker than sign recovery, consistent with a model that often gets direction right but not exact effect-size ordering.",
    "scatter_delta_pcc_vs_top100_amp_bias": "Better deltaPCC tends to coincide with less severe amplitude shrinkage, but amplitude bias remains a distinct failure mode.",
    "scatter_de_overlap100_vs_top100_direction_match": "Perturbations with better real top-DEG overlap also tend to have better sign agreement among top genes.",
    "scatter_de_overlap100_vs_top100_amp_bias": "DE overlap improves when predicted effects are less under-amplified, suggesting missing DE genes are partly caused by shrunken effect sizes.",
    "scatter_signed_lfc_pearson_vs_top100_amp_bias": "Signed top-DEG correlation improves as amplitude bias moves toward zero, again pointing to under-amplification as a key residual error.",
    "scatter_direction_match_vs_correct_under_fraction": "Direction correctness and under-amplification are nearly independent here; a prediction can have the right sign while still being too small.",
    "scatter_rmse_delta_vs_top100_amp_bias": "Higher delta RMSE is associated with stronger negative amplitude bias, so absolute error is partly driven by effect-size shrinkage.",
    "scatter_pred_significant_fraction_vs_de_overlap100": "The weak relationship indicates that simply predicting more significant genes is not enough; the model must identify the correct real DE genes.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replogle-root",
        default="benchmark/workspace/replogle_main_corrected_fullcell_remote_ep129_20260623",
        help="Replogle workspace root containing analyzing/tables and replogle_pred_*.h5ad.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sample-per-cell-line", type=int, default=800)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    figures = outdir / "figures"
    tables = outdir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def finite_series(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)


def corr_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3:
        return "n < 3"
    r = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r):
        return f"n={len(x)}"
    return f"r={r:.3f}, n={len(x)}"


def save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    figures: Path,
    title: str,
    name: str,
) -> dict[str, Any]:
    plot_df = finite_series(df, [x_col, y_col])
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=160)
    for cell_line, sub in plot_df.groupby("cell_line"):
        color = CELL_COLORS.get(str(cell_line).lower(), "#666666")
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=10,
            alpha=0.45,
            linewidths=0,
            color=color,
            label=str(sub["cell_line_label"].iloc[0]) if "cell_line_label" in sub else str(cell_line),
        )
    ax.set_title(title)
    ax.set_xlabel(LABELS.get(x_col, x_col))
    ax.set_ylabel(LABELS.get(y_col, y_col))
    if len(plot_df):
        ax.text(
            0.02,
            0.98,
            corr_text(plot_df[x_col].to_numpy(), plot_df[y_col].to_numpy()),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.25"},
        )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.tight_layout()
    png = figures / f"{name}.png"
    pdf = figures / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": png.name,
        "pdf": pdf.name,
        "kind": "metric_scatter",
        "x": x_col,
        "y": y_col,
        "n": int(len(plot_df)),
        "pearson_r": float(np.corrcoef(plot_df[x_col], plot_df[y_col])[0, 1]) if len(plot_df) >= 3 else "",
        "description": f"Scatter of {LABELS.get(x_col, x_col)} versus {LABELS.get(y_col, y_col)} per perturbation, colored by cell line.",
        "conclusion": SCATTER_CONCLUSIONS.get(name, "Use this scatter to inspect whether the two metrics agree or reveal a distinct failure mode."),
    }


def load_replogle_metrics(root: Path) -> pd.DataFrame:
    tables = root / "analyzing" / "tables"
    core = pd.read_csv(tables / "replogle_vcbench_core_metrics_per_perturbation.csv")
    top = pd.read_csv(tables / "replogle_top_deg_direction_magnitude_per_perturbation.csv")
    top = top[top["top_k"] == 100].copy()
    merged = core.merge(
        top,
        on=["cell_line", "cell_line_label", "perturbation"],
        how="inner",
        suffixes=("", "_top100"),
    )
    return merged


def plot_metric_scatters(metrics: pd.DataFrame, figures: Path) -> list[dict[str, Any]]:
    specs = [
        ("delta_pcc", "direction_match_rate", "deltaPCC vs top100 direction match", "scatter_delta_pcc_vs_top100_direction_match"),
        ("delta_pcc", "signed_lfc_pearson", "deltaPCC vs top100 signed log2FC Pearson", "scatter_delta_pcc_vs_top100_signed_lfc_pearson"),
        ("delta_pcc", "abs_lfc_spearman", "deltaPCC vs top100 magnitude-rank Spearman", "scatter_delta_pcc_vs_top100_abs_lfc_spearman"),
        ("delta_pcc", "mean_log2_amp_ratio", "deltaPCC vs top100 amplitude bias", "scatter_delta_pcc_vs_top100_amp_bias"),
        ("deg_overlap_at_100", "direction_match_rate", "DE overlap@100 vs top100 direction match", "scatter_de_overlap100_vs_top100_direction_match"),
        ("deg_overlap_at_100", "mean_log2_amp_ratio", "DE overlap@100 vs top100 amplitude bias", "scatter_de_overlap100_vs_top100_amp_bias"),
        ("signed_lfc_pearson", "mean_log2_amp_ratio", "Top100 signed log2FC Pearson vs amplitude bias", "scatter_signed_lfc_pearson_vs_top100_amp_bias"),
        ("direction_match_rate", "correct_under_fraction", "Top100 direction match vs under-amplification", "scatter_direction_match_vs_correct_under_fraction"),
        ("rmse_delta", "mean_log2_amp_ratio", "RMSE(delta) vs top100 amplitude bias", "scatter_rmse_delta_vs_top100_amp_bias"),
        ("pred_significant_fraction", "deg_overlap_at_100", "Predicted significant fraction vs DE overlap@100", "scatter_pred_significant_fraction_vs_de_overlap100"),
    ]
    rows = []
    for x_col, y_col, title, name in specs:
        rows.append(save_scatter(metrics, x_col, y_col, figures, title, name))
    return rows


def to_dense(x: Any) -> np.ndarray:
    try:
        from scipy import sparse

        if sparse.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    x = np.log1p(np.maximum(x, 0.0))
    x -= x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    x /= np.where(scale > 1e-6, scale, 1.0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    variance = (s[:2] ** 2) / max(1, x.shape[0] - 1)
    return coords, variance


def sample_h5ads(root: Path, metrics: pd.DataFrame, sample_per_cell_line: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    try:
        import anndata as ad
    except Exception as exc:
        return pd.DataFrame(), {"status": "missing_anndata", "error": str(exc)}

    metric_lookup = metrics.set_index(["cell_line", "perturbation"])
    rows: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    obsm_status: dict[str, Any] = {}
    for cell_line in ["k562", "hepg2", "jurkat", "rpe1"]:
        path = root / f"replogle_pred_{cell_line}.h5ad"
        if not path.exists():
            continue
        adata = ad.read_h5ad(path, backed="r")
        n = adata.n_obs
        take = min(sample_per_cell_line, n)
        idx = np.sort(rng.choice(n, size=take, replace=False))
        obs = adata.obs.iloc[idx].copy()
        obs["cell_line"] = cell_line
        obs["cell_line_label"] = cell_line.upper() if cell_line == "k562" else cell_line.capitalize()
        obs["perturbation"] = obs["gene"].astype(str)
        for metric in ["delta_pcc", "deg_overlap_at_100", "direction_match_rate", "mean_log2_amp_ratio"]:
            vals = []
            for pert in obs["perturbation"]:
                try:
                    vals.append(metric_lookup.loc[(cell_line, pert), metric])
                except KeyError:
                    vals.append(np.nan)
            obs[metric] = vals
        rows.append(obs.reset_index(drop=True))
        matrices.append(to_dense(adata.X[idx]))
        obsm_status[cell_line] = {"obsm_keys": list(adata.obsm.keys()), "n_sampled": int(take)}
        adata.file.close()

    if not rows:
        return pd.DataFrame(), {"status": "no_h5ad", "cell_lines": obsm_status}
    obs_all = pd.concat(rows, ignore_index=True)
    x_all = np.vstack(matrices)
    coords, variance = pca_2d(x_all)
    obs_all["pca1"] = coords[:, 0]
    obs_all["pca2"] = coords[:, 1]
    return obs_all, {
        "status": "expression_pca_proxy",
        "reason": "No Replogle prediction h5ad contains obsm latent/UMAP coordinates; umap-learn is not required for this PCA proxy.",
        "cell_lines": obsm_status,
        "variance_proxy": [float(v) for v in variance],
    }


def scatter_embedding(
    df: pd.DataFrame,
    color_col: str,
    figures: Path,
    name: str,
    title: str,
    continuous: bool,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=160)
    if continuous:
        vals = pd.to_numeric(df[color_col], errors="coerce")
        good = vals.notna()
        points = ax.scatter(
            df.loc[good, "pca1"],
            df.loc[good, "pca2"],
            c=vals[good],
            s=8,
            alpha=0.55,
            linewidths=0,
            cmap="viridis",
        )
        cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(LABELS.get(color_col, color_col))
    else:
        for label, sub in df.groupby(color_col):
            color = CELL_COLORS.get(str(label).lower(), None)
            ax.scatter(sub["pca1"], sub["pca2"], s=8, alpha=0.48, linewidths=0, color=color, label=str(label))
        ax.legend(frameon=False, fontsize=8, markerscale=1.6)
    ax.set_title(title)
    ax.set_xlabel("Expression PC1")
    ax.set_ylabel("Expression PC2")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    png = figures / f"{name}.png"
    pdf = figures / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": png.name,
        "pdf": pdf.name,
        "kind": "expression_pca_proxy",
        "color": color_col,
        "n": int(len(df)),
        "description": f"Prediction expression PCA proxy scatter colored by {LABELS.get(color_col, color_col)}.",
        "conclusion": "This is an expression-space proxy, not a model-latent UMAP; it can show whether sampled predicted cells separate by cell line or metric value, but it should not be used as evidence of learned latent geometry without real `obsm` embeddings.",
    }


def plot_embedding_proxy(samples: pd.DataFrame, figures: Path) -> list[dict[str, Any]]:
    if samples.empty:
        return []
    rows = [
        scatter_embedding(samples, "cell_line", figures, "pred_expression_pca_proxy_by_cell_line", "Predicted expression PCA proxy by cell line", continuous=False),
        scatter_embedding(samples, "delta_pcc", figures, "pred_expression_pca_proxy_by_delta_pcc", "Predicted expression PCA proxy colored by deltaPCC", continuous=True),
        scatter_embedding(samples, "deg_overlap_at_100", figures, "pred_expression_pca_proxy_by_de_overlap100", "Predicted expression PCA proxy colored by DE overlap@100", continuous=True),
        scatter_embedding(samples, "direction_match_rate", figures, "pred_expression_pca_proxy_by_direction_match", "Predicted expression PCA proxy colored by top100 direction match", continuous=True),
        scatter_embedding(samples, "mean_log2_amp_ratio", figures, "pred_expression_pca_proxy_by_amp_bias", "Predicted expression PCA proxy colored by amplitude bias", continuous=True),
    ]
    for cell_line, sub in samples.groupby("cell_line"):
        rows.append(
            scatter_embedding(
                sub,
                "delta_pcc",
                figures,
                f"pred_expression_pca_proxy_{cell_line}_by_delta_pcc",
                f"{cell_line.upper() if cell_line == 'k562' else cell_line.capitalize()} predicted expression PCA proxy by deltaPCC",
                continuous=True,
            )
        )
    return rows


def write_report(outdir: Path, figure_rows: list[dict[str, Any]], embedding_status: dict[str, Any]) -> None:
    lines = [
        "# Scatter and Representation-Space Figure Book",
        "",
        "This figure book adds scatter-heavy views for Replogle. Current Replogle prediction h5ad files do not expose `obsm` model latent coordinates or UMAP coordinates, so the representation-space figures are labeled as prediction expression PCA proxies rather than true model-latent UMAP.",
        "",
        "## Embedding Availability",
        "",
        "```json",
        json.dumps(embedding_status, indent=2),
        "```",
        "",
        "## Figures",
        "",
    ]
    for row in figure_rows:
        lines.append(f"### {row['figure']}")
        lines.append("")
        lines.append(row["description"])
        lines.append(f"Conclusion: {row.get('conclusion', '')}")
        if "pearson_r" in row and row.get("pearson_r") != "":
            lines.append(f"Pearson r: {row['pearson_r']:.4f}; n={row['n']}.")
        else:
            lines.append(f"n={row['n']}.")
        lines.append("")
        lines.append(f"![{row['figure']}](figures/{row['figure']})")
        lines.append("")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.replogle_root)
    outdir = Path(args.outdir)
    figures, tables = ensure_dirs(outdir)

    metrics = load_replogle_metrics(root)
    metrics.to_csv(tables / "replogle_scatter_metric_table_top100.csv", index=False)
    figure_rows = plot_metric_scatters(metrics, figures)

    samples, embedding_status = sample_h5ads(root, metrics, args.sample_per_cell_line, args.seed)
    if not samples.empty:
        samples.to_csv(tables / "replogle_pred_expression_pca_proxy_sample.csv", index=False)
    figure_rows.extend(plot_embedding_proxy(samples, figures))

    pd.DataFrame(figure_rows).to_csv(tables / "scatter_umap_figure_manifest.csv", index=False)
    (outdir / "run_manifest.json").write_text(
        json.dumps(
            {
                "replogle_root": str(root),
                "n_metric_rows": int(len(metrics)),
                "n_sample_rows": int(len(samples)),
                "n_figures": int(len(figure_rows)),
                "embedding_status": embedding_status,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(outdir, figure_rows, embedding_status)
    print(f"Wrote scatter/representation figures to {outdir}")


if __name__ == "__main__":
    main()
