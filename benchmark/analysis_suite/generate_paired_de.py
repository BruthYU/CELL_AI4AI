#!/usr/bin/env python3
"""Generate paired real/predicted DE summaries from h5ad predictions.

This script intentionally lives outside the dependency-free report generator.
It requires an environment with anndata, numpy, scipy, and pandas. It computes
paper-facing paired logFC objects:

- real DE: real perturbed cells vs matched real control cells
- predicted DE: predicted perturbed cells vs the same matched real controls
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import rankdata, spearmanr


EPS = 1e-9
LOGFC_MEAN_POLICY = "clip negative expression means to zero before fold-change/log2FC ratios"
DEFAULT_MODELS = ("cpa", "biolord", "state", "gears", "sclambda")
MODEL_LABELS = {
    "cpa": "CPA",
    "biolord": "BioLord",
    "state": "State",
    "gears": "GEARS",
    "sclambda": "scLambda",
    "scale_pdc": "SCALE+PDC",
    "ours_task03_aivc_flow": "SCALE task03 flow",
    "ours_task04_aivc_flow": "SCALE task04 flow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def iter_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_config_value(path: Path, key: str) -> str:
    pat = re.compile(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pat.match(line)
        if match:
            return match.group(1)
    return ""


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def split_class_for(benchmark: str, group: str) -> str:
    text = f"{benchmark} {group}".lower()
    if "zscape" in text:
        return "zscape_seen_cell_unseen_gene"
    if "kang" in text or "lopo" in text:
        return "kang_task05_lopo_patients"
    if "norman" in text or "task07" in text:
        return "norman_task07_heldout_combo_genes"
    if "task04" in text or "seen_single_unseen_combo" in text:
        return "task04_seen_single_unseen_combo"
    if "seen_drug_seen_cell_unseen_cells" in text:
        return "task03_seen_drug_seen_cell_unseen_cells"
    if "unseen_cell_line" in text:
        return "task03_unseen_cell_lines"
    if "unseen_dose" in text:
        return "task03_unseen_dose_extrapolation"
    return slugify(f"{benchmark}_{group}")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def bool_mask(series: Any) -> np.ndarray:
    values = np.asarray(series)
    if values.dtype == bool:
        return values
    lowered = np.char.lower(values.astype(str))
    return np.isin(lowered, ["true", "1", "yes", "control"])


def to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def mean_vec(adata: ad.AnnData, mask: np.ndarray) -> np.ndarray | None:
    if int(mask.sum()) == 0:
        return None
    matrix = to_dense(adata.X[mask])
    return np.asarray(matrix.mean(axis=0)).reshape(-1)


def mean_vec_indices(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray | None:
    if indices.size == 0:
        return None
    matrix = to_dense(adata.X[np.sort(indices)])
    return np.asarray(matrix.mean(axis=0)).reshape(-1)


def nonnegative_mean(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def fold_change(target_mean: np.ndarray, reference_mean: np.ndarray) -> np.ndarray:
    target = nonnegative_mean(target_mean)
    reference = nonnegative_mean(reference_mean)
    return (target + EPS) / (reference + EPS)


def log2fc(target_mean: np.ndarray, reference_mean: np.ndarray) -> np.ndarray:
    return np.log2(fold_change(target_mean, reference_mean))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return math.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = math.sqrt(float((x0 * x0).sum()) * float((y0 * y0).sum()))
    if denom <= EPS:
        return math.nan
    return float((x0 * y0).sum() / denom)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * sorted_labels).sum() / positives)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores)
    pos_rank_sum = float(ranks[labels.astype(bool)].sum())
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def finite(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def common_context_columns(real: ad.AnnData, pred: ad.AnnData, requested: list[str]) -> list[str]:
    return [col for col in requested if col in real.obs.columns and col in pred.obs.columns]


def context_mask(adata: ad.AnnData, context: dict[str, Any]) -> np.ndarray:
    mask = np.ones(adata.n_obs, dtype=bool)
    for col, value in context.items():
        if col in adata.obs.columns:
            mask &= np.asarray(adata.obs[col].astype(str)) == str(value)
    return mask


def reference_mask(
    real: ad.AnnData,
    control_col: str,
    context: dict[str, Any],
    target_col: str | None = None,
    control_value: str | None = None,
) -> np.ndarray:
    if control_col in real.obs.columns:
        mask = bool_mask(real.obs[control_col])
    elif control_value and target_col and target_col in real.obs.columns:
        mask = np.asarray(real.obs[target_col].astype(str)) == str(control_value)
    else:
        mask = np.zeros(real.n_obs, dtype=bool)
    return mask & context_mask(real, context)


def target_mask(adata: ad.AnnData, target_col: str, target: str, context: dict[str, Any], control_col: str | None = None) -> np.ndarray:
    mask = np.asarray(adata.obs[target_col].astype(str)) == str(target)
    if control_col and control_col in adata.obs.columns:
        mask &= ~bool_mask(adata.obs[control_col])
    return mask & context_mask(adata, context)


def is_control_target(target: Any, control_value: str | None = None) -> bool:
    lowered = str(target).lower()
    control_values = {
        "control",
        str(control_value or "").lower(),
        "[('control', 0.0, 'um')]",
        "[('control', 0.0, 'uM')]".lower(),
    }
    control_values.discard("")
    return lowered in control_values


def target_context_records(
    pred: ad.AnnData,
    pred_base_mask: np.ndarray,
    target_col: str,
    control_col: str,
    control_value: str,
    context_cols: list[str],
    max_targets: int,
) -> list[tuple[str, dict[str, Any]]]:
    if target_col not in pred.obs.columns:
        raise ValueError(f"target_col {target_col!r} is missing from prediction obs")
    candidate_mask = pred_base_mask.copy()
    if control_col in pred.obs.columns:
        candidate_mask &= ~bool_mask(pred.obs[control_col])
    elif control_value:
        candidate_mask &= np.asarray(pred.obs[target_col].astype(str)) != str(control_value)
    cols = [target_col, *context_cols]
    frame = pred.obs.loc[candidate_mask, cols].astype(str).drop_duplicates()
    records: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    for row in frame.to_dict("records"):
        target = str(row[target_col])
        if is_control_target(target, control_value):
            continue
        context = {col: row[col] for col in context_cols}
        key = (target, tuple(sorted(context.items())))
        if key in seen:
            continue
        seen.add(key)
        records.append((target, context))
    records.sort(key=lambda item: (item[0], tuple(str(item[1].get(col, "")) for col in context_cols)))
    if max_targets > 0:
        records = records[:max_targets]
    return records


def target_group_indices(
    adata: ad.AnnData,
    base_mask: np.ndarray,
    target_col: str,
    control_col: str,
    control_value: str,
    context_cols: list[str],
) -> dict[tuple[str, tuple[tuple[str, Any], ...]], np.ndarray]:
    if target_col not in adata.obs.columns:
        raise ValueError(f"target_col {target_col!r} is missing from obs")
    candidate_mask = base_mask.copy()
    if control_col in adata.obs.columns:
        candidate_mask &= ~bool_mask(adata.obs[control_col])
    elif control_value:
        candidate_mask &= np.asarray(adata.obs[target_col].astype(str)) != str(control_value)
    positions = np.flatnonzero(candidate_mask)
    if positions.size == 0:
        return {}
    cols = [target_col, *context_cols]
    frame = adata.obs.iloc[positions][cols].astype(str).reset_index(drop=True)
    groups: dict[tuple[str, tuple[tuple[str, Any], ...]], np.ndarray] = {}
    for key, local_positions in frame.groupby(cols, sort=True).indices.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        target = str(key_tuple[0])
        if is_control_target(target, control_value):
            continue
        context = {col: str(key_tuple[idx + 1]) for idx, col in enumerate(context_cols)}
        groups[(target, tuple(sorted(context.items())))] = positions[np.asarray(local_positions, dtype=int)]
    return groups


def reference_indices(
    real: ad.AnnData,
    control_col: str,
    target_col: str,
    control_value: str,
    context: dict[str, Any],
) -> np.ndarray:
    if control_col in real.obs.columns:
        mask = bool_mask(real.obs[control_col])
    elif control_value and target_col in real.obs.columns:
        mask = np.asarray(real.obs[target_col].astype(str)) == str(control_value)
    else:
        mask = np.zeros(real.n_obs, dtype=bool)
    for col, value in context.items():
        if col in real.obs.columns:
            mask &= np.asarray(real.obs[col].astype(str)) == str(value)
    return np.flatnonzero(mask)


def compute_unit(
    unit: dict[str, Any],
    top_k: int,
    real_cache: dict[str, ad.AnnData] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    real_owned = False
    real_path = str(unit["real_path"])
    if real_cache is not None:
        real = real_cache.get(real_path)
        if real is None:
            real = ad.read_h5ad(real_path, backed="r")
            real_cache[real_path] = real
    else:
        real = ad.read_h5ad(real_path, backed="r")
        real_owned = True
    pred = ad.read_h5ad(unit["pred_path"], backed="r")
    target_col = unit.get("target_col", "perturbation")
    control_col = unit.get("control_col", "control")
    control_value = unit.get("control_value", "")
    split_key = unit.get("split_key", "")
    split_value = unit.get("split_value", "test")
    context_cols = common_context_columns(real, pred, unit.get("context_cols", []))
    feature_names = [str(x) for x in real.var_names]
    pred_feature_names = [str(x) for x in pred.var_names]
    if len(feature_names) != pred.n_vars:
        raise ValueError(f"Gene dimension mismatch: {unit['real_path']} vs {unit['pred_path']}")
    if feature_names != pred_feature_names:
        raise ValueError(f"Gene name/order mismatch: {unit['real_path']} vs {unit['pred_path']}")

    pred_base_mask = np.ones(pred.n_obs, dtype=bool)
    real_base_mask = np.ones(real.n_obs, dtype=bool)
    if split_key and split_key in real.obs.columns:
        real_base_mask &= np.asarray(real.obs[split_key].astype(str)) == split_value
    if split_key and split_key in pred.obs.columns:
        pred_base_mask &= np.asarray(pred.obs[split_key].astype(str)) == split_value

    max_targets = int(unit.get("max_targets", 0) or 0)
    pred_groups = target_group_indices(pred, pred_base_mask, target_col, control_col, control_value, context_cols)
    real_groups = target_group_indices(real, real_base_mask, target_col, control_col, control_value, context_cols)
    target_records = sorted(pred_groups)
    if max_targets > 0:
        target_records = target_records[:max_targets]
    event_rows: list[dict[str, Any]] = []
    real_de_rows: list[dict[str, Any]] = []
    pred_de_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    ref_cache: dict[tuple[tuple[str, Any], ...], tuple[int, np.ndarray | None]] = {}

    for target, context_key in target_records:
        context = dict(context_key)
        if context_key in ref_cache:
            n_real_control, real_ref = ref_cache[context_key]
        else:
            real_ref_idx = reference_indices(real, control_col, target_col, control_value, context)
            n_real_control = int(real_ref_idx.size)
            real_ref = mean_vec_indices(real, real_ref_idx)
            ref_cache[context_key] = (n_real_control, real_ref)
        real_target_idx = real_groups.get((target, context_key), np.asarray([], dtype=int))
        pred_target_idx = pred_groups.get((target, context_key), np.asarray([], dtype=int))
        if n_real_control == 0 or real_target_idx.size == 0 or pred_target_idx.size == 0:
            continue
        real_target = mean_vec_indices(real, real_target_idx)
        pred_target = mean_vec_indices(pred, pred_target_idx)
        if real_ref is None or real_target is None or pred_target is None:
            continue
        real_lfc = log2fc(real_target, real_ref)
        pred_lfc = log2fc(pred_target, real_ref)
        real_order = np.argsort(-np.abs(real_lfc))
        pred_order = np.argsort(-np.abs(pred_lfc))
        real_top = real_order[:top_k]
        pred_top = set(int(i) for i in pred_order[:top_k])
        real_top_set = set(int(i) for i in real_top)
        labels = np.zeros(real_lfc.shape[0], dtype=int)
        labels[list(real_top_set)] = 1
        scores = np.abs(pred_lfc)
        direction = np.sign(real_lfc[real_top]) == np.sign(pred_lfc[real_top])
        overlap = len(real_top_set & pred_top) / float(top_k)
        target_summaries.append(
            {
                **unit,
                "target": target,
                "context": ";".join(f"{k}={v}" for k, v in sorted(context.items())),
                "n_real_target": int(real_target_idx.size),
                "n_real_control": n_real_control,
                "n_pred_target": int(pred_target_idx.size),
                "top_k": top_k,
                "pearson_logfc_topk": finite(pearson(real_lfc[real_top], pred_lfc[real_top])),
                "spearman_logfc_topk": finite(float(spearmanr(real_lfc[real_top], pred_lfc[real_top]).statistic)),
                "direction_agreement_topk": finite(float(direction.mean())),
                "de_overlap_at_k": finite(overlap),
                "auprc_real_topk": finite(average_precision(labels, scores)),
                "auroc_real_topk": finite(auroc(labels, scores)),
            }
        )
        for rank, idx in enumerate(real_top, start=1):
            feature = feature_names[int(idx)]
            real_fc = float(fold_change(real_target[idx : idx + 1], real_ref[idx : idx + 1])[0])
            pred_fc = float(fold_change(pred_target[idx : idx + 1], real_ref[idx : idx + 1])[0])
            common = {
                **{k: unit.get(k, "") for k in ("dataset_id", "benchmark", "split_class", "split", "model", "model_label", "seed", "cellclass")},
                "target": target,
                "reference": "matched_control",
                "feature": feature,
                "rank": rank,
                "context": ";".join(f"{k}={v}" for k, v in sorted(context.items())),
            }
            event_rows.append(
                {
                    **common,
                    "real_log2fc": float(real_lfc[idx]),
                    "pred_log2fc": float(pred_lfc[idx]),
                    "direction_match": int(np.sign(real_lfc[idx]) == np.sign(pred_lfc[idx])),
                    "abs_real_log2fc": abs(float(real_lfc[idx])),
                    "abs_pred_log2fc": abs(float(pred_lfc[idx])),
                    "n_real_target": int(real_target_idx.size),
                    "n_real_control": n_real_control,
                    "n_pred_target": int(pred_target_idx.size),
                }
            )
            real_de_rows.append(
                {
                    **common,
                    "target_mean": float(real_target[idx]),
                    "reference_mean": float(real_ref[idx]),
                    "percent_change": real_fc - 1.0,
                    "fold_change": real_fc,
                    "p_value": "",
                    "statistic": "",
                    "fdr": "",
                }
            )
            pred_de_rows.append(
                {
                    **common,
                    "target_mean": float(pred_target[idx]),
                    "reference_mean": float(real_ref[idx]),
                    "percent_change": pred_fc - 1.0,
                    "fold_change": pred_fc,
                    "p_value": "",
                    "statistic": "",
                    "fdr": "",
                }
            )

    if real_owned:
        real.file.close()
    pred.file.close()
    return event_rows, real_de_rows, pred_de_rows, target_summaries


def discover_vcbench_units(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["runs_root"])
    models = set(config.get("models", DEFAULT_MODELS))
    seeds = {str(x) for x in config.get("seeds", [42])}
    units: list[dict[str, Any]] = []
    for dataset_cfg in config.get("datasets", []):
        dataset = dataset_cfg["dataset"]
        splits = dataset_cfg.get("splits", [])
        for split in splits:
            for model in models:
                for seed in seeds:
                    run_dir = root / f"{dataset}__{model}__seed{seed}__{split}"
                    hydra = run_dir / "hydra"
                    cfg_path = hydra / ".hydra" / "config.yaml"
                    if not cfg_path.exists():
                        continue
                    real_path = parse_config_value(cfg_path, "data_path")
                    split_key = parse_config_value(cfg_path, "split_key")
                    pred_paths = sorted((hydra / "cellclass_evaluation").glob("*/predictions.h5ad"))
                    max_cellclasses = int(dataset_cfg.get("max_cellclasses", 0) or 0)
                    if max_cellclasses > 0:
                        pred_paths = pred_paths[:max_cellclasses]
                    for pred_path in pred_paths:
                        cellclass = pred_path.parent.name
                        benchmark = dataset_cfg.get("benchmark", dataset)
                        units.append(
                            {
                                "dataset_id": dataset,
                                "benchmark": benchmark,
                                "split_class": split_class_for(benchmark, split),
                                "split": split,
                                "model": model,
                                "model_label": model_label(model),
                                "seed": seed,
                                "cellclass": cellclass,
                                "real_path": real_path,
                                "pred_path": str(pred_path),
                                "target_col": dataset_cfg.get("target_col", "perturbation"),
                                "control_col": dataset_cfg.get("control_col", "control"),
                                "split_key": split_key,
                                "split_value": dataset_cfg.get("split_value", "test"),
                                "context_cols": dataset_cfg.get("context_cols", []),
                            }
                        )
    return units


def discover_matched_units(config: dict[str, Any]) -> list[dict[str, Any]]:
    detail_rows = iter_csv(Path(config["detail_path"]))
    group_rows = iter_csv(Path(config["group_path"]))
    include_models = set(config.get("models", []))
    real_by_split = {}
    for row in group_rows:
        if row.get("real_path"):
            real_by_split[row["split"]] = row["real_path"]
    units = []
    seen = set()
    for row in detail_rows:
        model = row.get("model", "")
        if include_models and model not in include_models:
            continue
        split = row.get("split", "")
        pred_path = row.get("prediction_path", "")
        real_path = real_by_split.get(split, "")
        if not pred_path or not real_path:
            continue
        key = (split, model, pred_path)
        if key in seen:
            continue
        seen.add(key)
        task = "task04" if split.startswith("task04") else "task03"
        benchmark = "ComboSciPlex3 Task04" if task == "task04" else "ComboSciPlex3 Task03"
        units.append(
            {
                "dataset_id": task,
                "benchmark": benchmark,
                "split_class": split_class_for(benchmark, split),
                "split": split,
                "model": model,
                "model_label": model_label(model),
                "seed": "",
                "cellclass": "",
                "real_path": real_path,
                "pred_path": pred_path,
                "target_col": config.get("target_col", "perturbation"),
                "control_col": config.get("control_col", "control"),
                "split_key": "",
                "split_value": "",
                "context_cols": config.get("context_cols", ["cell_line"]),
            }
        )
    return units


def discover_scale_pdc_units(config: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    defaults = {
        "model": "scale_pdc",
        "model_label": model_label("scale_pdc"),
        "seed": "42",
    }
    for unit in config.get("manual_units", []):
        merged = {**defaults, **unit}
        merged["model_label"] = merged.get("model_label") or model_label(str(merged.get("model", "scale_pdc")))
        units.append(merged)

    kang_path = config.get("kang_h5ad_list_path")
    if kang_path:
        kang_cfg = config.get("kang", {})
        for row in iter_csv(Path(kang_path)):
            pred_path = row.get("pred_h5ad_path", "")
            real_path = row.get("real_h5ad_path", "")
            if not pred_path or not real_path:
                continue
            test_patient = row.get("test_patient", "")
            split = kang_cfg.get("split", "")
            if not split and test_patient:
                split = f"split_lopo_test_{test_patient}"
            units.append(
                {
                    **defaults,
                    "dataset_id": kang_cfg.get("dataset_id", "kang"),
                    "benchmark": kang_cfg.get("benchmark", "Kang Task05"),
                    "split_class": kang_cfg.get("split_class", "kang_task05_lopo_patients"),
                    "split": split or row.get("split_id", ""),
                    "seed": kang_cfg.get("seed", "42"),
                    "cellclass": row.get("fold", ""),
                    "real_path": real_path,
                    "pred_path": pred_path,
                    "target_col": kang_cfg.get("target_col", "condition"),
                    "control_col": kang_cfg.get("control_col", "control"),
                    "control_value": kang_cfg.get("control_value", "PBS"),
                    "split_key": kang_cfg.get("split_key", ""),
                    "split_value": kang_cfg.get("split_value", ""),
                    "context_cols": kang_cfg.get("context_cols", ["celltype"]),
                    "source_table": kang_path,
                    "selected_lambda": row.get("selected_lambda", ""),
                    "selected_metric_value": row.get("selected_metric_value", ""),
                    "test_patient": test_patient,
                    "val_patient": row.get("val_patient", ""),
                }
            )
    return units


def summarize_targets_by_keys(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    summaries = []
    metrics = ["pearson_logfc_topk", "spearman_logfc_topk", "direction_agreement_topk", "de_overlap_at_k", "auprc_real_topk", "auroc_real_topk"]
    for key, vals in grouped.items():
        out = {k: v for k, v in zip(keys, key)}
        out["n_targets"] = len(vals)
        out["n_events"] = len(vals) * int(vals[0].get("top_k", 0) or 0)
        for metric in metrics:
            parsed = [float(v[metric]) for v in vals if v.get(metric) not in ("", None)]
            out[metric + "_mean"] = sum(parsed) / len(parsed) if parsed else ""
        summaries.append(out)
    summaries.sort(key=lambda r: tuple(str(r.get(k, "")) for k in keys))
    return summaries


def summarize_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_targets_by_keys(rows, ["dataset_id", "benchmark", "split_class", "split", "model_label", "seed", "cellclass"])


def write_svg_scatter(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    points = []
    for row in rows:
        try:
            points.append((float(row["real_log2fc"]), float(row["pred_log2fc"]), str(row.get("model_label", ""))))
        except Exception:
            pass
    if not points:
        return
    if len(points) > 5000:
        step = max(1, len(points) // 5000)
        points = points[::step]
    models = sorted({p[2] for p in points})
    palette = ["#599CB4", "#C25759", "#5B8C5A", "#B696B6", "#8C6D31", "#6C8EBF"]
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo = min(xs + ys)
    hi = max(xs + ys)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad
    width, height = 720, 520
    left, right, top, bottom = 75, 145, 45, 65
    pw, ph = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - lo) / (hi - lo) * pw

    def sy(y: float) -> float:
        return top + ph - (y - lo) / (hi - lo) * ph

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#e7e7e7}.axis{stroke:#333}</style>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16">{title}</text>',
    ]
    for i in range(5):
        val = lo + (hi - lo) * i / 4
        x = sx(val)
        y = sy(val)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        body.append(f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" font-size="10">{val:.1f}</text>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{val:.1f}</text>')
    body.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    for x, y, model in points:
        body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2" fill="{color[model]}" fill-opacity="0.35"/>')
    for idx, model in enumerate(models):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+pw+20}" y="{y-9}" width="10" height="10" fill="{color[model]}"/>')
        body.append(f'<text x="{left+pw+36}" y="{y}" font-size="11">{model}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    body.append(f'<text x="{left+pw/2}" y="{height-20}" text-anchor="middle" font-size="12">Real log2FC</text>')
    body.append(f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20,{top+ph/2})" text-anchor="middle" font-size="12">Predicted log2FC</text>')
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_svg_metric_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    points = []
    for row in rows:
        try:
            x = float(row[x_metric])
            y = float(row[y_metric])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y, str(row.get("model_label", "")), str(row.get("split_class", ""))))
    if not points:
        return
    models = sorted({p[2] for p in points})
    palette = ["#599CB4", "#C25759", "#5B8C5A", "#B696B6", "#8C6D31", "#6C8EBF"]
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 0.05
        xmax += 0.05
    if ymin == ymax:
        ymin -= 0.05
        ymax += 0.05
    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.08
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad
    width, height = 760, 540
    left, right, top, bottom = 80, 150, 45, 70
    pw, ph = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return top + ph - (y - ymin) / (ymax - ymin) * ph

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#e7e7e7}.axis{stroke:#333}</style>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16">{title}</text>',
    ]
    for i in range(5):
        xv = xmin + (xmax - xmin) * i / 4
        yv = ymin + (ymax - ymin) * i / 4
        x = sx(xv)
        y = sy(yv)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        body.append(f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" font-size="10">{xv:.2f}</text>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{yv:.2f}</text>')
    for x, y, model, split_class in points:
        body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color[model]}" fill-opacity="0.8"><title>{model} {split_class}: {x:.4f}, {y:.4f}</title></circle>')
    for idx, model in enumerate(models):
        y = top + 18 + idx * 18
        body.append(f'<rect x="{left+pw+20}" y="{y-9}" width="10" height="10" fill="{color[model]}"/>')
        body.append(f'<text x="{left+pw+36}" y="{y}" font-size="11">{model}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    body.append(f'<text x="{left+pw/2}" y="{height-22}" text-anchor="middle" font-size="12">{xlabel}</text>')
    body.append(f'<text x="22" y="{top+ph/2}" transform="rotate(-90 22,{top+ph/2})" text-anchor="middle" font-size="12">{ylabel}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def best_by_model(rows: list[dict[str, Any]], metric: str, higher_is_better: bool = True) -> tuple[str, float] | None:
    vals: list[tuple[str, float]] = []
    for row in rows:
        try:
            val = float(row.get(metric, ""))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append((str(row.get("model_label", "")), val))
    if not vals:
        return None
    return max(vals, key=lambda x: x[1]) if higher_is_better else min(vals, key=lambda x: x[1])


def write_report(outdir: Path, config: dict[str, Any], model_summaries: list[dict[str, Any]], cell_summaries: list[dict[str, Any]], figures: list[Path]) -> None:
    lines = ["# Paired DE paper-ready analysis", ""]
    lines.append("This report uses paired real/predicted logFC objects. Real DE is computed from real perturbed cells versus matched real controls; predicted DE is computed from predicted perturbed cells versus the same matched real controls.")
    lines.append("Held-out target cells respect the configured test split. Matched real controls are selected by context from the real control pool and are not forced to be in the test split, because some VCBench protocols, such as Norman Task07, store controls in the train partition.")
    lines.append(f"LogFC mean policy: {LOGFC_MEAN_POLICY}. This is required because some external baseline prediction matrices contain negative expression means, which are outside the fold-change domain.")
    lines.append("")
    lines.append("## Analysis Prompt")
    lines.append("")
    prompt_path = config.get("analysis_prompt_path")
    if prompt_path and Path(prompt_path).exists():
        lines.append(f"Full prompt source: `{prompt_path}`")
        prompt_text = Path(prompt_path).read_text(encoding="utf-8")
        (outdir / "analysis_prompt.md").write_text(prompt_text, encoding="utf-8")
        lines.append("")
        lines.append(prompt_text)
    else:
        lines.append("No prompt file was found in the config.")
    lines.append("")
    lines.append("## Paired DE Overview")
    lines.append("")
    lines.append("| Split class | Split | Model | Seed | Targets | Events | Pearson logFC | Spearman logFC | DirAgr | DEOver@K | AUPRC | AUROC |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in model_summaries:
        lines.append(
            "| {split_class} | {split} | {model_label} | {seed} | {n_targets} | {n_events} | {pearson} | {spearman} | {diragr} | {overlap} | {auprc} | {auroc} |".format(
                split_class=row.get("split_class", ""),
                split=row.get("split", ""),
                model_label=row.get("model_label", ""),
                seed=row.get("seed", ""),
                n_targets=row.get("n_targets", ""),
                n_events=row.get("n_events", ""),
                pearson=fmt(row.get("pearson_logfc_topk_mean")),
                spearman=fmt(row.get("spearman_logfc_topk_mean")),
                diragr=fmt(row.get("direction_agreement_topk_mean")),
                overlap=fmt(row.get("de_overlap_at_k_mean")),
                auprc=fmt(row.get("auprc_real_topk_mean")),
                auroc=fmt(row.get("auroc_real_topk_mean")),
            )
        )
    lines.append("")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_summaries:
        by_split[str(row.get("split_class", ""))].append(row)
    model_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_summaries:
        model_by_split[str(row.get("split_class", ""))].append(row)
    combined_figures = [fig for fig in figures if "__" not in fig.stem and not fig.stem.startswith("paired_de_metric_")]
    model_figures = [fig for fig in figures if "__" in fig.stem and not fig.stem.startswith("paired_de_metric_")]
    metric_figures = [fig for fig in figures if fig.stem.startswith("paired_de_metric_")]
    figures_by_split = {fig.stem.replace("paired_de_", "").replace("_top100_real_vs_pred_log2fc", ""): fig for fig in combined_figures}
    model_figures_by_split: dict[str, list[Path]] = defaultdict(list)
    for fig in model_figures:
        split_part = fig.stem.replace("paired_de_", "").split("__", 1)[0]
        model_figures_by_split[split_part].append(fig)
    if metric_figures:
        lines.append("## Metric Summary Figures")
        lines.append("")
        lines.append("These figures compare paired-DE metrics across models with grouped box plots and jittered/strip scatter overlays. Boxes summarize target-level metric distributions, while points show individual perturbation/context rows.")
        lines.append("")
        for fig in sorted(metric_figures):
            rel = fig.relative_to(outdir)
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
            metric_name = fig.stem.replace("paired_de_metric_box_jitter_", "").replace("_", " ")
            lines.append(f"Brief interpretation: this plot shows the model-level distribution of {metric_name}; higher values indicate better paired-DE recovery for that metric.")
            lines.append("")
    lines.append("## Split Sections")
    lines.append("")
    for split_class in sorted(by_split):
        rows = by_split[split_class]
        model_rows = model_by_split.get(split_class, [])
        split_names = sorted({str(row.get("split", "")) for row in rows if row.get("split")})
        lines.append(f"## {split_class}")
        lines.append("")
        lines.append("Question: On this split class, does SCALE+PDC recover top real-DE gene directions and amplitudes better than the non-scale baselines CPA, BioLord, State, GEARS, and scLambda under the same matched-control DE construction?")
        lines.append("")
        lines.append("Mathematical object: paired `G_DE, logFC` where real logFC is real perturbed vs matched real control, and predicted logFC is baseline predicted perturbed vs the same matched real control.")
        lines.append("")
        lines.append("Concrete procedure: for each model/cellclass/perturbation, select top real-effect genes by absolute real log2FC, plot real vs predicted top100 log2FC, and compute Pearson/Spearman logFC, direction agreement, DE overlap@K, AUPRC, and AUROC.")
        lines.append("")
        pearson_best = best_by_model(model_rows, "pearson_logfc_topk_mean")
        dir_best = best_by_model(model_rows, "direction_agreement_topk_mean")
        overlap_best = best_by_model(model_rows, "de_overlap_at_k_mean")
        interp = []
        if pearson_best:
            interp.append(f"highest top100 logFC Pearson: {pearson_best[0]} ({pearson_best[1]:.4f})")
        if dir_best:
            interp.append(f"highest direction agreement: {dir_best[0]} ({dir_best[1]:.4f})")
        if overlap_best:
            interp.append(f"highest DE overlap@K: {overlap_best[0]} ({overlap_best[1]:.4f})")
        if interp:
            lines.append("Result interpretation: " + "; ".join(interp) + ".")
        else:
            lines.append("Result interpretation: no successful paired DE rows were generated for this split class.")
        lines.append("")
        if split_names:
            lines.append("Splits included: " + ", ".join(split_names))
            lines.append("")
        lines.append("Model-level summary:")
        lines.append("")
        lines.append("| Split | Model | Seed | Targets | Events | Pearson logFC | Spearman logFC | DirAgr | DEOver@K | AUPRC | AUROC |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(model_rows, key=lambda r: (str(r.get("split", "")), str(r.get("model_label", "")), str(r.get("seed", "")))):
            lines.append(
                "| {split} | {model_label} | {seed} | {n_targets} | {n_events} | {pearson} | {spearman} | {diragr} | {overlap} | {auprc} | {auroc} |".format(
                    split=row.get("split", ""),
                    model_label=row.get("model_label", ""),
                    seed=row.get("seed", ""),
                    n_targets=row.get("n_targets", ""),
                    n_events=row.get("n_events", ""),
                    pearson=fmt(row.get("pearson_logfc_topk_mean")),
                    spearman=fmt(row.get("spearman_logfc_topk_mean")),
                    diragr=fmt(row.get("direction_agreement_topk_mean")),
                    overlap=fmt(row.get("de_overlap_at_k_mean")),
                    auprc=fmt(row.get("auprc_real_topk_mean")),
                    auroc=fmt(row.get("auroc_real_topk_mean")),
                )
            )
        lines.append("")
        lines.append("Cellclass-level detail:")
        lines.append("")
        lines.append("| Split | Model | Seed | Cellclass | Targets | Pearson logFC | Spearman logFC | DirAgr | DEOver@K | AUPRC | AUROC |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(rows, key=lambda r: (str(r.get("split", "")), str(r.get("model_label", "")), str(r.get("cellclass", "")))):
            lines.append(
                "| {split} | {model_label} | {seed} | {cellclass} | {n_targets} | {pearson} | {spearman} | {diragr} | {overlap} | {auprc} | {auroc} |".format(
                    split=row.get("split", ""),
                    model_label=row.get("model_label", ""),
                    seed=row.get("seed", ""),
                    cellclass=row.get("cellclass", ""),
                    n_targets=row.get("n_targets", ""),
                    pearson=fmt(row.get("pearson_logfc_topk_mean")),
                    spearman=fmt(row.get("spearman_logfc_topk_mean")),
                    diragr=fmt(row.get("direction_agreement_topk_mean")),
                    overlap=fmt(row.get("de_overlap_at_k_mean")),
                    auprc=fmt(row.get("auprc_real_topk_mean")),
                    auroc=fmt(row.get("auroc_real_topk_mean")),
                )
            )
        lines.append("")
        fig = figures_by_split.get(split_class)
        if fig:
            rel = fig.relative_to(outdir)
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
            lines.append("Brief interpretation: this diagnostic scatter compares real top-DE log2FC to predicted log2FC; points near the diagonal indicate better recovery of both sign and effect size.")
            lines.append("")
        model_figs = sorted(model_figures_by_split.get(split_class, []))
        if model_figs:
            lines.append("Model-specific top100 logFC scatter figures:")
            lines.append("")
            lines.append("Brief interpretation: use these model-specific diagnostic scatters to inspect whether a model is direction-correct, under-amplified, or producing off-diagonal effects for real top-DE genes.")
            lines.append("")
            for model_fig in model_figs:
                rel = model_fig.relative_to(outdir)
                model_name = model_fig.stem.split("__", 1)[1].replace("_top100_real_vs_pred_log2fc", "")
                lines.append(f"- `{model_name}`: [{model_fig.name}]({rel})")
            lines.append("")
    text = "\n".join(lines)
    outdir.joinpath("paired_de_report.md").write_text(text, encoding="utf-8")
    outdir.joinpath("report.md").write_text(text, encoding="utf-8")


def fmt(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.4f}"


def main() -> None:
    args = parse_args()
    config = read_json(Path(args.config))
    outdir = Path(args.outdir)
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    units: list[dict[str, Any]] = []
    if "vcbench" in config:
        units.extend(discover_vcbench_units(config["vcbench"]))
    if "matched" in config:
        units.extend(discover_matched_units(config["matched"]))
    if "scale_pdc" in config:
        units.extend(discover_scale_pdc_units(config["scale_pdc"]))

    all_events: list[dict[str, Any]] = []
    all_real_de: list[dict[str, Any]] = []
    all_pred_de: list[dict[str, Any]] = []
    all_targets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, unit in enumerate(units, start=1):
        print(f"[{idx}/{len(units)}] {unit['benchmark']} {unit['split']} {unit['model_label']} {unit.get('cellclass','')}", flush=True)
        try:
            events, real_de, pred_de, target_rows = compute_unit(unit, args.top_k)
        except Exception as exc:
            failures.append({**unit, "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_events.extend(events)
        all_real_de.extend(real_de)
        all_pred_de.extend(pred_de)
        all_targets.extend(target_rows)

    summaries = summarize_targets(all_targets)
    model_summaries = summarize_targets_by_keys(all_targets, ["dataset_id", "benchmark", "split_class", "split", "model_label", "seed"])
    write_csv(tables / "paired_de_top100_events.csv", all_events)
    write_csv(tables / "paired_real_de_top100.csv", all_real_de)
    write_csv(tables / "paired_pred_de_top100.csv", all_pred_de)
    write_csv(tables / "paired_de_target_metrics.csv", all_targets)
    write_csv(tables / "paired_de_metric_summary.csv", summaries)
    write_csv(tables / "paired_de_model_metric_summary.csv", model_summaries)
    write_csv(tables / "paired_de_failures.csv", failures)

    figure_paths: list[Path] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_events:
        by_split[row["split_class"]].append(row)
    for split_class, rows in by_split.items():
        fig = figures / f"paired_de_{split_class}_top100_real_vs_pred_log2fc.svg"
        write_svg_scatter(fig, rows, f"{split_class}: paired top100 real vs predicted log2FC")
        if fig.exists():
            figure_paths.append(fig)
        rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_model[str(row.get("model_label", "model"))].append(row)
        for model_name, model_rows in sorted(rows_by_model.items()):
            model_slug = slugify(model_name)
            model_fig = figures / f"paired_de_{split_class}__{model_slug}_top100_real_vs_pred_log2fc.svg"
            write_svg_scatter(model_fig, model_rows, f"{split_class} {model_name}: top100 real vs predicted log2FC")
            if model_fig.exists():
                figure_paths.append(model_fig)

    metric_specs = [
        (
            "pearson_logfc_topk_mean",
            "de_overlap_at_k_mean",
            "paired_de_metric_alignment_pearson_logfc_vs_deover.svg",
            "Model-level paired DE: logFC Pearson vs DE overlap",
            "Mean top100 logFC Pearson",
            "Mean DE overlap@100",
        ),
        (
            "pearson_logfc_topk_mean",
            "direction_agreement_topk_mean",
            "paired_de_metric_alignment_pearson_logfc_vs_diragr.svg",
            "Model-level paired DE: logFC Pearson vs direction agreement",
            "Mean top100 logFC Pearson",
            "Mean direction agreement@100",
        ),
        (
            "de_overlap_at_k_mean",
            "direction_agreement_topk_mean",
            "paired_de_metric_alignment_deover_vs_diragr.svg",
            "Model-level paired DE: DE overlap vs direction agreement",
            "Mean DE overlap@100",
            "Mean direction agreement@100",
        ),
    ]
    for x_metric, y_metric, filename, title, xlabel, ylabel in metric_specs:
        fig = figures / filename
        write_svg_metric_scatter(fig, model_summaries, x_metric, y_metric, title, xlabel, ylabel)
        if fig.exists():
            figure_paths.append(fig)

    write_report(outdir, config, model_summaries, summaries, figure_paths)
    manifest = {
        "n_units": len(units),
        "n_failures": len(failures),
        "n_events": len(all_events),
        "n_targets": len(all_targets),
        "n_summaries": len(summaries),
        "n_model_summaries": len(model_summaries),
        "top_k": args.top_k,
        "n_figures": len(figure_paths),
        "logfc_mean_policy": LOGFC_MEAN_POLICY,
    }
    (outdir / "paired_de_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote paired DE outputs to {outdir}")


if __name__ == "__main__":
    main()
