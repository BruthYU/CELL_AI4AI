#!/usr/bin/env python3
"""Build full-cell Replogle model + other-cell train-delta memory predictions.

This reproduces the Replogle D_same_gene_other_cellline fusion used by the
existing compact artifact, but it writes full-cell AnnData files.  The output
keeps the model prediction h5ad row multiplicity: each perturbation retains its
predicted cells instead of being collapsed to one pseudobulk row.

For every target row with cell line c and perturbation p:

    delta_model_cell = model_pred_cell - mean_model_control(c)
    delta_memory = D_same_gene_other_cellline(p, c)
    pred_cell = target_control(c) + a * delta_model_cell + b * delta_memory

In workspace-memory mode, target_control(c) is the memory/prior h5ad control
mean, matching the June 22 provenance.  Control rows are shifted so their mean
equals target_control(c) while retaining full-cell row count.  The default a/b
values and paths match the June 22 Replogle artifact provenance:

    a = 0.1, b = 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_train_delta_memory_h5ad import (  # noqa: E402
    CELL_LINES,
    DEFAULT_REPLOGLE_ROOT,
    TrainDeltaMemory,
    _build_memory,
)


DEFAULT_MODEL_WORKSPACE = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "aivc-llama-jit-replogle-v3-statealign-resid-gdelta-neg0235-set512-dit36-2gpu-acc4-lr8e5-w6_ep129_none_normalgpu1_mem24g_scale0528_full_eval/"
    "predictions"
)
DEFAULT_MEMORY_WORKSPACE = (
    PROJECT_ROOT
    / "benchmark"
    / "workspace"
    / "replogle_reproduce_othercell_delta_20260622"
    / "predictions"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "workspace"
    / "replogle_othercell_model010_memory090_fullcell_repro"
)


def _as_dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _obs_strings(adata: ad.AnnData, column: str) -> pd.Series:
    if column not in adata.obs:
        raise KeyError(f"AnnData is missing obs column {column!r}; columns={list(adata.obs.columns)}")
    return adata.obs[column].astype(str)


def _row_mean(adata: ad.AnnData, row_indices: np.ndarray) -> np.ndarray:
    if row_indices.size == 0:
        raise ValueError("Cannot take mean over zero rows")
    return _as_dense_float32(adata.X[row_indices]).mean(axis=0).astype(np.float32)


def _group_means(path: Path, *, celltype_col: str, pert_col: str) -> dict[tuple[str, str], np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        cells = _obs_strings(adata, celltype_col).to_numpy()
        perts = _obs_strings(adata, pert_col).to_numpy()
        groups: dict[tuple[str, str], list[int]] = {}
        for idx, key in enumerate(zip(cells, perts, strict=True)):
            groups.setdefault((str(key[0]), str(key[1])), []).append(idx)
        return {
            key: _row_mean(adata, np.asarray(indices, dtype=np.int64))
            for key, indices in groups.items()
        }
    finally:
        adata.file.close()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _mean_nan_as_zero(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nan_to_num(arr, nan=0.0).mean())


def _summarize_pearsons(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return {
        "num_genes": int(arr.size),
        "finite_pearson_genes": int(np.count_nonzero(finite)),
        "nan_pearson_genes": int(arr.size - np.count_nonzero(finite)),
        "finite_pearson_fraction": float(np.count_nonzero(finite) / arr.size) if arr.size else 0.0,
        "pearson_delta_finite_only": _finite_mean(values),
        "pearson_delta_nan_as_zero": _mean_nan_as_zero(values),
        "_finite_pearson_sum": float(arr[finite].sum()) if np.any(finite) else 0.0,
        "_nan_as_zero_pearson_sum": float(np.nan_to_num(arr, nan=0.0).sum()) if arr.size else 0.0,
    }


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _score_workspace(
    *,
    workspace: Path,
    cell_lines: list[str],
    control_pert: str,
    celltype_col: str,
    pert_col: str,
    name: str,
) -> dict[str, Any]:
    cell_scores = []
    for cell_line in cell_lines:
        pred_path = workspace / f"replogle_pred_{cell_line}.h5ad"
        real_path = workspace / f"replogle_real_{cell_line}.h5ad"
        pred = _group_means(pred_path, celltype_col=celltype_col, pert_col=pert_col)
        real = _group_means(real_path, celltype_col=celltype_col, pert_col=pert_col)
        pred_control = pred[(cell_line, control_pert)]
        real_control = real[(cell_line, control_pert)]
        pearsons = []
        for pert in sorted({key[1] for key in pred} & {key[1] for key in real}):
            if pert == control_pert:
                continue
            pearsons.append(_pearson(pred[(cell_line, pert)] - pred_control, real[(cell_line, pert)] - real_control))
        row = _summarize_pearsons(pearsons)
        row["cell_line"] = cell_line
        row["pearson_delta"] = row["pearson_delta_finite_only"]
        cell_scores.append(row)

    finite_total = int(sum(int(row["finite_pearson_genes"]) for row in cell_scores))
    gene_total = int(sum(int(row["num_genes"]) for row in cell_scores))
    finite_sum = float(sum(float(row["_finite_pearson_sum"]) for row in cell_scores))
    nan_as_zero_sum = float(sum(float(row["_nan_as_zero_pearson_sum"]) for row in cell_scores))
    mean_finite = _finite_mean([float(row["pearson_delta_finite_only"]) for row in cell_scores])
    mean_nan_as_zero = _finite_mean([float(row["pearson_delta_nan_as_zero"]) for row in cell_scores])
    return {
        "name": name,
        "workspace": str(workspace),
        "mean_pearson_delta": mean_finite,
        "mean_pearson_delta_finite_only": mean_finite,
        "mean_pearson_delta_nan_as_zero": mean_nan_as_zero,
        "gene_weighted_pearson_delta_finite_only": float(finite_sum / finite_total) if finite_total else float("nan"),
        "gene_weighted_pearson_delta_nan_as_zero": float(nan_as_zero_sum / gene_total) if gene_total else float("nan"),
        "num_genes": gene_total,
        "finite_pearson_genes": finite_total,
        "nan_pearson_genes": int(gene_total - finite_total),
        "finite_pearson_fraction": float(finite_total / gene_total) if gene_total else 0.0,
        "cell_lines": [_strip_internal(row) for row in cell_scores],
    }


class MemoryDeltaResolver:
    def __init__(
        self,
        *,
        mode: str,
        memory_workspace: Path | None,
        memory: TrainDeltaMemory | None,
        cell_lines: list[str],
        control_pert: str,
        celltype_col: str,
        pert_col: str,
        average_mode: str,
        missing_action: str,
    ) -> None:
        self.mode = mode
        self.memory = memory
        self.control_pert = control_pert
        self.average_mode = average_mode
        self.missing_action = missing_action
        self.workspace_deltas: dict[tuple[str, str], np.ndarray] = {}
        self.workspace_controls: dict[str, np.ndarray] = {}

        if mode == "workspace":
            if memory_workspace is None:
                raise ValueError("memory_workspace is required in workspace mode")
            for cell_line in cell_lines:
                means = _group_means(
                    memory_workspace / f"replogle_pred_{cell_line}.h5ad",
                    celltype_col=celltype_col,
                    pert_col=pert_col,
                )
                control = means.get((cell_line, control_pert))
                if control is None:
                    raise KeyError(f"Missing memory control for {cell_line}")
                self.workspace_controls[cell_line] = control
                for key, value in means.items():
                    if key[0] != cell_line or key[1] == control_pert:
                        continue
                    self.workspace_deltas[key] = (value - control).astype(np.float32)

    def resolve(self, cell_line: str, pert: str, reference: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.mode == "workspace":
            delta = self.workspace_deltas.get((cell_line, pert))
            if delta is not None:
                return delta, {
                    "mode": "D_same_gene_other_cellline_workspace",
                    "support_context_count": None,
                    "support_cell_count": None,
                }
        else:
            if self.memory is None:
                raise ValueError("memory object is required in lmdb mode")
            delta, info = self.memory.same_pert_other_context(cell_line, pert, average_mode=self.average_mode)
            if delta is not None:
                return delta, {"mode": "D_same_gene_other_cellline_train_lmdb", **info}

        if self.missing_action == "error":
            raise KeyError(f"Missing D_same_gene_other_cellline for cell_line={cell_line!r}, pert={pert!r}")
        return np.zeros_like(reference, dtype=np.float32), {
            "mode": "zero_delta_missing",
            "support_context_count": 0,
            "support_cell_count": 0.0,
        }

    def anchor_control(self, cell_line: str, fallback: np.ndarray) -> np.ndarray:
        if self.mode == "workspace":
            control = self.workspace_controls.get(cell_line)
            if control is None:
                raise KeyError(f"Missing memory/prior control for cell_line={cell_line!r}")
            return control.astype(np.float32, copy=False)
        return fallback.astype(np.float32, copy=False)


def _load_lmdb_memory(args: argparse.Namespace, cell_lines: list[str]) -> tuple[TrainDeltaMemory, dict[str, Any]]:
    root = Path(args.replogle_root)
    global_keys_path = Path(args.global_keys_path.format(root=root))
    with global_keys_path.open("rb") as f:
        global_keys = pickle.load(f)
    memory, report = _build_memory(
        contexts=cell_lines,
        root=root,
        train_lmdb_template=args.train_lmdb_template,
        control_lmdb_template=args.control_lmdb_template,
        global_keys=global_keys,
        context_vocab_key=args.context_vocab_key,
        pert_vocab_key=args.pert_vocab_key,
        control_pert=args.control_pert,
        progress_every=int(args.progress_every),
    )
    report["global_keys_path"] = str(global_keys_path)
    return memory, report


def _target_count_report(
    adata: ad.AnnData,
    *,
    celltype_col: str,
    pert_col: str,
    control_pert: str,
    row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if row_mask is None:
        obs = adata.obs[[celltype_col, pert_col]].astype(str)
    else:
        obs = adata.obs.loc[row_mask, [celltype_col, pert_col]].astype(str)
    counts = obs.groupby([celltype_col, pert_col], observed=True).size()
    target_counts = counts[[key[1] != control_pert for key in counts.index]]
    one_cell = [
        {"celltype": key[0], "perturbation": key[1], "n_cells": int(value)}
        for key, value in target_counts.items()
        if int(value) <= 1
    ]
    return {
        "num_targets": int(target_counts.size),
        "min_cells_per_target": int(target_counts.min()) if target_counts.size else 0,
        "max_cells_per_target": int(target_counts.max()) if target_counts.size else 0,
        "one_cell_targets": one_cell[:50],
        "num_one_cell_targets": len(one_cell),
    }


def _write_cell_line(
    *,
    cell_line: str,
    model_workspace: Path,
    out_workspace: Path,
    resolver: MemoryDeltaResolver,
    model_weight: float,
    memory_weight: float,
    control_pert: str,
    celltype_col: str,
    pert_col: str,
    clip_min: float | None,
    allow_one_cell_targets: bool,
) -> dict[str, Any]:
    model_pred_path = model_workspace / f"replogle_pred_{cell_line}.h5ad"
    model_real_path = model_workspace / f"replogle_real_{cell_line}.h5ad"
    if not model_pred_path.exists():
        raise FileNotFoundError(model_pred_path)
    if not model_real_path.exists():
        raise FileNotFoundError(model_real_path)

    adata = ad.read_h5ad(model_pred_path, backed="r")
    try:
        cells = _obs_strings(adata, celltype_col).to_numpy()
        perts = _obs_strings(adata, pert_col).to_numpy()
        cell_mask = cells == cell_line
        if not np.any(cell_mask):
            raise RuntimeError(f"{cell_line}: no rows with {celltype_col}={cell_line!r}")

        count_report = _target_count_report(
            adata,
            celltype_col=celltype_col,
            pert_col=pert_col,
            control_pert=control_pert,
            row_mask=cell_mask,
        )
        if count_report["num_one_cell_targets"] and not allow_one_cell_targets:
            raise RuntimeError(
                f"{cell_line}: model prediction has one-cell perturbation groups; "
                f"first={count_report['one_cell_targets'][:5]}"
            )

        control_indices = np.flatnonzero((cells == cell_line) & (perts == control_pert))
        if control_indices.size == 0:
            raise RuntimeError(f"{cell_line}: missing model control rows")
        model_control_mean = _row_mean(adata, control_indices)
        anchor_control_mean = resolver.anchor_control(cell_line, model_control_mean)

        obs_out_parts: list[pd.DataFrame] = []
        x_out_parts: list[sparse.csr_matrix] = []
        support_counts: dict[str, int] = {}
        missing_memory = []
        written_rows = 0

        for pert in sorted(set(perts[cell_mask]), key=str):
            indices = np.flatnonzero(cell_mask & (perts == pert))
            if indices.size == 0:
                continue
            rows = _as_dense_float32(adata.X[indices])
            if pert == control_pert:
                pred_rows = rows + (anchor_control_mean - model_control_mean).reshape(1, -1)
                if clip_min is not None:
                    pred_rows = np.maximum(pred_rows, clip_min)
                mode = "control_shifted_to_anchor"
            else:
                memory_delta, info = resolver.resolve(cell_line, str(pert), model_control_mean)
                mode = str(info["mode"])
                if mode == "zero_delta_missing":
                    missing_memory.append(str(pert))
                pred_rows = anchor_control_mean.reshape(1, -1) + model_weight * (
                    rows - model_control_mean.reshape(1, -1)
                ) + memory_weight * memory_delta.reshape(1, -1)
                if clip_min is not None:
                    pred_rows = np.maximum(pred_rows, clip_min)

            support_counts[mode] = support_counts.get(mode, 0) + 1
            obs_out_parts.append(adata.obs.iloc[indices].copy())
            x_out_parts.append(sparse.csr_matrix(pred_rows.astype(np.float32, copy=False)))
            written_rows += int(indices.size)

        out_workspace.mkdir(parents=True, exist_ok=True)
        pred_out = out_workspace / f"replogle_pred_{cell_line}.h5ad"
        real_out = out_workspace / f"replogle_real_{cell_line}.h5ad"
        obs_out = pd.concat(obs_out_parts, axis=0)
        obs_out.index = [str(i) for i in range(written_rows)]
        pred_adata = ad.AnnData(X=sparse.vstack(x_out_parts), obs=obs_out, var=adata.var.copy())
        pred_adata.uns["delta_memory_fusion"] = {
            "formula": "pred_cell = target_control + model_weight*(model_pred_cell-mean_model_control) + memory_weight*D_same_gene_other_cellline",
            "model_weight": float(model_weight),
            "memory_weight": float(memory_weight),
            "control_pert": control_pert,
        }
        pred_adata.write_h5ad(pred_out)

        if real_out.exists() or real_out.is_symlink():
            real_out.unlink()
        os.symlink(model_real_path.resolve(), real_out)

        return {
            "cell_line": cell_line,
            "model_pred_path": str(model_pred_path),
            "model_real_path": str(model_real_path),
            "pred_path": str(pred_out),
            "real_path": str(real_out),
            "written_rows": int(written_rows),
            "target_count_report": count_report,
            "mode_counts": support_counts,
            "num_missing_memory_targets": len(missing_memory),
            "missing_memory_targets": missing_memory[:50],
        }
    finally:
        adata.file.close()


def _parse_cell_lines(text: str) -> list[str]:
    return [value.strip().lower() for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-workspace", default=str(DEFAULT_MODEL_WORKSPACE))
    parser.add_argument("--memory-mode", choices=("workspace", "lmdb"), default="workspace")
    parser.add_argument("--memory-workspace", default=str(DEFAULT_MEMORY_WORKSPACE))
    parser.add_argument("--cell-lines", default=",".join(CELL_LINES))
    parser.add_argument("--model-weight", type=float, default=0.1)
    parser.add_argument("--memory-weight", type=float, default=0.9)
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--celltype-col", default="celltype")
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--allow-one-cell-targets", action="store_true")
    parser.add_argument("--missing-memory-action", choices=("zero", "error"), default="zero")
    parser.add_argument("--score-name", default="model010_memory090_fullcell")
    parser.add_argument("--expected-rounded-mean-delta-pcc", type=float, default=0.75)
    parser.add_argument("--score-decimals", type=int, default=2)
    parser.add_argument("--min-mean-delta-pcc", type=float, default=None)

    parser.add_argument("--replogle-root", default=str(DEFAULT_REPLOGLE_ROOT))
    parser.add_argument("--global-keys-path", default="{root}/global_keys.pkl")
    parser.add_argument("--train-lmdb-template", default="{root}/few_shot/{context}/replogle_train_{context}")
    parser.add_argument("--control-lmdb-template", default="{root}/few_shot/{context}/replogle_control_{context}")
    parser.add_argument("--context-vocab-key", default="cell_line")
    parser.add_argument("--pert-vocab-key", default="gene")
    parser.add_argument("--average-mode", choices=("equal_context", "cell_weighted"), default="equal_context")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_lines = _parse_cell_lines(args.cell_lines)
    model_workspace = Path(args.model_workspace)
    memory_workspace = Path(args.memory_workspace) if args.memory_workspace else None
    out_root = Path(args.out_dir)
    out_workspace = out_root / "predictions"
    out_root.mkdir(parents=True, exist_ok=True)

    memory_report: dict[str, Any] | None = None
    memory_obj: TrainDeltaMemory | None = None
    if args.memory_mode == "lmdb":
        memory_obj, memory_report = _load_lmdb_memory(args, cell_lines)

    resolver = MemoryDeltaResolver(
        mode=args.memory_mode,
        memory_workspace=memory_workspace,
        memory=memory_obj,
        cell_lines=cell_lines,
        control_pert=args.control_pert,
        celltype_col=args.celltype_col,
        pert_col=args.pert_col,
        average_mode=args.average_mode,
        missing_action=args.missing_memory_action,
    )

    cell_summaries = []
    for cell_line in cell_lines:
        print(f"[replogle-fullcell] writing {cell_line}", flush=True)
        cell_summaries.append(
            _write_cell_line(
                cell_line=cell_line,
                model_workspace=model_workspace,
                out_workspace=out_workspace,
                resolver=resolver,
                model_weight=float(args.model_weight),
                memory_weight=float(args.memory_weight),
                control_pert=args.control_pert,
                celltype_col=args.celltype_col,
                pert_col=args.pert_col,
                clip_min=args.clip_min,
                allow_one_cell_targets=bool(args.allow_one_cell_targets),
            )
        )

    score = _score_workspace(
        workspace=out_workspace,
        cell_lines=cell_lines,
        control_pert=args.control_pert,
        celltype_col=args.celltype_col,
        pert_col=args.pert_col,
        name=args.score_name,
    )
    direct_score_payload = {"workspaces": [score]}
    (out_root / "direct_delta_pearson.json").write_text(json.dumps(direct_score_payload, indent=2, sort_keys=True) + "\n")

    mean_score = float(score["mean_pearson_delta_nan_as_zero"])
    if args.min_mean_delta_pcc is not None and mean_score < float(args.min_mean_delta_pcc):
        raise RuntimeError(
            f"mean_pearson_delta_nan_as_zero={mean_score:.10f} is below "
            f"--min-mean-delta-pcc={args.min_mean_delta_pcc}"
        )
    if args.expected_rounded_mean_delta_pcc is not None:
        rounded = round(mean_score, int(args.score_decimals))
        expected = round(float(args.expected_rounded_mean_delta_pcc), int(args.score_decimals))
        if rounded != expected:
            raise RuntimeError(
                f"Rounded mean delta PCC check failed: round({mean_score:.10f}, "
                f"{args.score_decimals})={rounded}, expected={expected}"
            )

    provenance = {
        "script": str(Path(__file__).resolve()),
        "format_reference": str(PROJECT_ROOT / "main_inference_replogle.py"),
        "formula": "pred_cell = target_control + model_weight*(model_pred_cell-mean_model_control) + memory_weight*D_same_gene_other_cellline",
        "canonical_branch": "model_plus_D_same_gene_other_cellline",
        "model_workspace": str(model_workspace),
        "memory_mode": args.memory_mode,
        "memory_workspace": str(memory_workspace) if memory_workspace is not None else None,
        "memory_source": memory_report,
        "model_weight": float(args.model_weight),
        "memory_weight": float(args.memory_weight),
        "clip_min": args.clip_min,
        "control_pert": args.control_pert,
        "celltype_col": args.celltype_col,
        "pert_col": args.pert_col,
        "cell_lines": cell_lines,
        "no_cell_averaging_in_output": True,
        "output_row_source": "model prediction rows; perturbation row counts are preserved",
        "expected_rounded_mean_delta_pcc": args.expected_rounded_mean_delta_pcc,
        "score_decimals": int(args.score_decimals),
        "observed_mean_delta_pcc": mean_score,
    }
    (out_root / "blend_formula_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    summary = {
        **provenance,
        "out_workspace": str(out_workspace),
        "cell_outputs": cell_summaries,
        "direct_delta_pearson_json": str(out_root / "direct_delta_pearson.json"),
        "direct_delta_pearson": score,
    }
    (out_root / "fullcell_model_memory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
