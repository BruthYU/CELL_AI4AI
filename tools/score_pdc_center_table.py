#!/usr/bin/env python3
"""Score PDC centers at mean level without writing full-cell predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from apply_prior_delta_centering import grouped_means, pearson, split_control_means


def load_center_table(path: Path, n_vars: int) -> dict[tuple[str, str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    contexts = [str(x) for x in payload["contexts"]]
    pert_names = [str(x) for x in payload["pert_names"]]
    if vectors.ndim != 2 or int(vectors.shape[1]) != int(n_vars):
        raise ValueError(f"{path}: vectors shape={vectors.shape}, expected second dim {n_vars}")
    return {
        (context, pert): vectors[idx].astype(np.float32, copy=False)
        for idx, (context, pert) in enumerate(zip(contexts, pert_names))
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_lambdas(value: str) -> list[float]:
    lambdas = [float(item.strip()) for item in value.split(",") if item.strip()]
    for lambda_weight in lambdas:
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda must be in [0,1], got {lambda_weight}")
    return lambdas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--real-h5ad", type=Path, required=True)
    parser.add_argument("--center-table", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--lambdas", default="0,0.5,0.9,1")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--pred-clip-min", type=float, default=None)
    parser.add_argument("--real-clip-min", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambdas = parse_lambdas(args.lambdas)
    pred_means, pred_counts, n_vars = grouped_means(
        args.pred_h5ad,
        context_col=args.context_col,
        default_context=None,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.pred_clip_min,
    )
    real_means, real_counts, real_n_vars = grouped_means(
        args.real_h5ad,
        context_col=args.context_col,
        default_context=None,
        pert_col=args.pert_col,
        chunk_rows=int(args.chunk_rows),
        clip_min=args.real_clip_min,
    )
    if int(real_n_vars) != int(n_vars):
        raise ValueError(f"real vars={real_n_vars}, pred vars={n_vars}")
    pred_control, _ = split_control_means(pred_means, pred_counts, control_label=args.control_label)
    real_control, _ = split_control_means(real_means, real_counts, control_label=args.control_label)
    center_table = load_center_table(args.center_table, n_vars)

    summaries = {}
    rows_by_lambda = {}
    common_groups = sorted((set(pred_means) & set(real_means) & set(center_table)) - set())
    common_groups = [(c, p) for c, p in common_groups if p != args.control_label and c in pred_control and c in real_control]
    for lambda_weight in lambdas:
        rows = []
        for context, pert in common_groups:
            mu_model = pred_means[(context, pert)]
            prior_delta = center_table[(context, pert)]
            mu_pdc = mu_model + float(lambda_weight) * (pred_control[context] + prior_delta - mu_model)
            score = pearson(mu_pdc - pred_control[context], real_means[(context, pert)] - real_control[context])
            rows.append(
                {
                    "context": context,
                    "perturbation": pert,
                    "pearson_delta": score,
                    "pred_count": int(pred_counts[(context, pert)]),
                    "real_count": int(real_counts[(context, pert)]),
                }
            )
        values = np.asarray([row["pearson_delta"] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        key = f"{lambda_weight:g}"
        summaries[key] = {
            "lambda": float(lambda_weight),
            "num_scores": int(values.size),
            "num_finite": int(finite.size),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
        }
        rows_by_lambda[key] = rows

    payload = {
        "pred_h5ad": str(args.pred_h5ad),
        "real_h5ad_eval_only": str(args.real_h5ad),
        "center_table": str(args.center_table),
        "formula": "mu_pdc = mu_model + lambda * (pred_control + D_train - mu_model)",
        "control_label": args.control_label,
        "pred_clip_min": args.pred_clip_min,
        "real_clip_min": args.real_clip_min,
        "context_col": args.context_col,
        "pert_col": args.pert_col,
        "num_common_target_groups": len(common_groups),
        "summary_by_lambda": summaries,
        "rows_by_lambda": rows_by_lambda,
        "leakage_note": "real_h5ad is used only for scoring; center_table is precomputed train-only input.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_safe({k: v for k, v in payload.items() if k != "rows_by_lambda"}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
