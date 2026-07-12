#!/usr/bin/env python3
"""Generate expression-space PCA proxy scatter plots for h5ad predictions.

These figures are not latent UMAPs. They are a dependency-light fallback for
artifact-backed runs that do not expose model `obsm` embeddings and where
`umap-learn` is not installed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


RUNS_ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/vcbench_comparison_20260611/vcbench_runs")
PREPARED_ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/reports/vcbench_comparison_20260611/prepared_data")
MODELS = ["cpa", "biolord", "state", "gears", "sclambda"]
MODEL_LABELS = {
    "cpa": "CPA",
    "biolord": "BioLord",
    "state": "State",
    "gears": "GEARS",
    "sclambda": "scLambda",
}
COLORS = {
    "real_control": "#666666",
    "real_perturbed": "#111111",
    "CPA": "#599CB4",
    "BioLord": "#5B8C5A",
    "State": "#6C8EBF",
    "GEARS": "#B696B6",
    "scLambda": "#8C6D31",
    "SCALE+PDC": "#C25759",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-per-group", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def bool_mask(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr
    lowered = np.char.lower(arr.astype(str))
    return np.isin(lowered, ["true", "1", "yes", "control"])


def context_mask(adata: ad.AnnData, context: dict[str, str]) -> np.ndarray:
    mask = np.ones(adata.n_obs, dtype=bool)
    for col, value in context.items():
        if col in adata.obs.columns:
            mask &= np.asarray(adata.obs[col].astype(str)) == str(value)
    return mask


def split_mask(adata: ad.AnnData, split_key: str, split_value: str) -> np.ndarray:
    if split_key and split_key in adata.obs.columns:
        return np.asarray(adata.obs[split_key].astype(str)) == split_value
    return np.ones(adata.n_obs, dtype=bool)


def to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def sample_indices(mask: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= max_n:
        return idx
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def read_sample(adata: ad.AnnData, mask: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = sample_indices(mask, max_n, rng)
    if idx.size == 0:
        return np.zeros((0, adata.n_vars), dtype=np.float32)
    return to_dense(adata.X[idx]).astype(np.float32, copy=False)


def pca2(matrix: np.ndarray) -> np.ndarray:
    x = matrix.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def write_svg(path: Path, coords: np.ndarray, labels: list[str], title: str) -> None:
    if coords.shape[0] == 0:
        return
    xs = coords[:, 0]
    ys = coords[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.08
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad
    width, height = 820, 600
    left, right, top, bottom = 78, 170, 48, 70
    pw, ph = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return top + ph - (y - ymin) / (ymax - ymin) * ph

    order = ["real_control", "real_perturbed", "CPA", "BioLord", "State", "GEARS", "scLambda", "SCALE+PDC"]
    groups = [g for g in order if g in set(labels)]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#e8e8e8}.axis{stroke:#333}</style>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="16">{html.escape(title)}</text>',
    ]
    for i in range(5):
        xv = xmin + (xmax - xmin) * i / 4
        yv = ymin + (ymax - ymin) * i / 4
        x = sx(xv)
        y = sy(yv)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        body.append(f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" font-size="10">{xv:.1f}</text>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{yv:.1f}</text>')
    for group in groups:
        color = COLORS.get(group, "#666666")
        radius = 3.0 if group.startswith("real_") else 2.5
        opacity = 0.42 if group.startswith("real_") else 0.34
        for x, y, label in zip(xs, ys, labels):
            if label != group:
                continue
            body.append(f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}"/>')
    for idx, group in enumerate(groups):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+pw+24}" y="{y-9}" width="10" height="10" fill="{COLORS.get(group, "#666666")}"/>')
        body.append(f'<text x="{left+pw+40}" y="{y}" font-size="11">{html.escape(group)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    body.append(f'<text x="{left+pw/2}" y="{height-22}" text-anchor="middle" font-size="12">Expression PC1</text>')
    body.append(f'<text x="22" y="{top+ph/2}" transform="rotate(-90 22,{top+ph/2})" text-anchor="middle" font-size="12">Expression PC2</text>')
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def build_spec_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": "kang_patient101_cd4t",
            "dataset": "kang",
            "benchmark": "Kang Task05",
            "split": "split_lopo_test_patient_101",
            "split_class": "kang_task05_lopo_patients",
            "cellclass": "CD4 T",
            "real_path": str(PREPARED_ROOT / "kang2018_task05_pbmc2000_vcbench.h5ad"),
            "context": {"patient": "patient_101", "eval_cell_type": "CD4 T"},
            "split_key": "split_lopo_test_patient_101",
            "split_value": "test",
            "extra_predictions": [
                {
                    "label": "SCALE+PDC",
                    "pred_path": "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/benchmark/workspace/kang_baseline_pdc_20260625/patient_101_val_patient_1015/lambda_0p9/patient_101_val_patient_1015_test_pred.h5ad",
                    "context": {"celltype": "CD4 Memory"},
                    "target_col": "condition",
                    "control_col": "control",
                    "control_value": "PBS",
                    "context_note": "SCALE+PDC Kang h5ad uses CD4 Memory while VCBench external proxy uses CD4 T.",
                }
            ],
        },
        {
            "name": "norman_k562_task07",
            "dataset": "norman",
            "benchmark": "Norman Task07",
            "split": "split_task07_seed42",
            "split_class": "norman_task07_heldout_combo_genes",
            "cellclass": "K562",
            "real_path": str(PREPARED_ROOT / "norman_task07_seed42_hvg2000_vcbench.h5ad"),
            "context": {"eval_cell_type": "K562"},
            "split_key": "split_task07_seed42",
            "split_value": "test",
            "extra_predictions": [
                {
                    "label": "SCALE+PDC",
                    "pred_path": "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/benchmark/workspace/small_pdc_val_selected_20260624/task07_norman/task07_norman_test_pdc_lam0p7122_pred.h5ad",
                    "context": {"celltype": "K562"},
                    "target_col": "condition",
                    "control_col": "control",
                    "control_value": "control",
                }
            ],
        },
    ]


def predicted_noncontrol_mask(pred: ad.AnnData, extra: dict[str, Any]) -> np.ndarray:
    mask = context_mask(pred, extra.get("context", {}))
    mask &= split_mask(pred, extra.get("split_key", ""), extra.get("split_value", ""))
    control_col = extra.get("control_col", "control")
    target_col = extra.get("target_col", "perturbation")
    control_value = str(extra.get("control_value", ""))
    if control_col in pred.obs.columns:
        mask &= ~bool_mask(pred.obs[control_col])
    elif control_value and target_col in pred.obs.columns:
        mask &= np.asarray(pred.obs[target_col].astype(str)) != control_value
    return mask


def generate_spec(spec: dict[str, Any], outdir: Path, max_per_group: int, rng: np.random.Generator) -> dict[str, Any]:
    real = ad.read_h5ad(spec["real_path"], backed="r")
    control_col = "control"
    target_col = "perturbation"
    real_context = context_mask(real, spec["context"])
    real_control_mask = real_context & bool_mask(real.obs[control_col])
    real_target_mask = real_context & split_mask(real, spec["split_key"], spec["split_value"]) & ~bool_mask(real.obs[control_col])
    matrices = [read_sample(real, real_control_mask, max_per_group, rng), read_sample(real, real_target_mask, max_per_group, rng)]
    labels = ["real_control"] * matrices[0].shape[0] + ["real_perturbed"] * matrices[1].shape[0]
    n_by_group = {"real_control": int(matrices[0].shape[0]), "real_perturbed": int(matrices[1].shape[0])}
    real_var = [str(x) for x in real.var_names]

    for model in MODELS:
        pred_path = RUNS_ROOT / f"{spec['dataset']}__{model}__seed44__{spec['split']}" / "hydra" / "cellclass_evaluation" / spec["cellclass"] / "predictions.h5ad"
        if not pred_path.exists():
            continue
        pred = ad.read_h5ad(pred_path, backed="r")
        pred_var = [str(x) for x in pred.var_names]
        if pred_var != real_var:
            pred.file.close()
            continue
        pred_mask = context_mask(pred, spec["context"]) & split_mask(pred, spec["split_key"], spec["split_value"])
        if control_col in pred.obs.columns:
            pred_mask &= ~bool_mask(pred.obs[control_col])
        matrix = read_sample(pred, pred_mask, max_per_group, rng)
        matrices.append(matrix)
        label = MODEL_LABELS[model]
        labels.extend([label] * matrix.shape[0])
        n_by_group[label] = int(matrix.shape[0])
        pred.file.close()

    for extra in spec.get("extra_predictions", []):
        pred_path = Path(extra["pred_path"])
        if not pred_path.exists():
            continue
        pred = ad.read_h5ad(pred_path, backed="r")
        pred_var = [str(x) for x in pred.var_names]
        if pred_var != real_var:
            pred.file.close()
            continue
        matrix = read_sample(pred, predicted_noncontrol_mask(pred, extra), max_per_group, rng)
        matrices.append(matrix)
        label = str(extra.get("label", "SCALE+PDC"))
        labels.extend([label] * matrix.shape[0])
        n_by_group[label] = int(matrix.shape[0])
        pred.file.close()

    real.file.close()
    nonempty = [m for m in matrices if m.shape[0] > 0]
    if not nonempty:
        return {**spec, "figure": "", "n_by_group": n_by_group, "n_points": 0, "status": "no_matched_cells"}
    combined = np.vstack(nonempty)
    coords = pca2(combined)
    fig = outdir / "figures" / f"expression_pca_proxy_{spec['name']}_all_models.svg"
    write_svg(fig, coords, labels, f"{spec['benchmark']} {spec['cellclass']}: expression PCA proxy")
    return {**spec, "figure": str(fig), "n_by_group": n_by_group, "n_points": int(coords.shape[0]), "status": "ok"}


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = [generate_spec(spec, outdir, args.max_per_group, rng) for spec in build_spec_rows()]
    lines = [
        "# Expression-Space PCA Proxy Scatter",
        "",
        "These are not model latent UMAPs. Current baseline h5ad artifacts do not expose `obsm` model embeddings, and the active environment lacks `umap-learn`/`sklearn`. The figures instead place real control, real perturbed, and predicted perturbed cells in a shared expression PCA space using sampled cells.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['name']}")
        lines.append("")
        lines.append(f"Benchmark: {row['benchmark']}; split: `{row['split']}`; cellclass: `{row['cellclass']}`.")
        lines.append("")
        lines.append("Point counts: " + ", ".join(f"{k}={v}" for k, v in sorted(row["n_by_group"].items())))
        lines.append("")
        notes = [extra.get("context_note", "") for extra in row.get("extra_predictions", []) if extra.get("context_note")]
        for note in notes:
            lines.append(f"Note: {note}")
            lines.append("")
        if row.get("figure"):
            rel = Path(row["figure"]).relative_to(outdir)
            lines.append(f"![{Path(row['figure']).stem}]({rel})")
        else:
            lines.append(f"No figure generated: {row.get('status', 'unknown')}.")
        lines.append("")
    (outdir / "expression_pca_proxy_report.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "expression_pca_proxy_manifest.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote expression PCA proxy outputs to {outdir}")


if __name__ == "__main__":
    main()
