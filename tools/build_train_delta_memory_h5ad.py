#!/usr/bin/env python3
"""Build train-only delta-memory h5ad outputs for state cell-eval.

The default paths and output names target Replogle.  The AnnData layout matches
``main_inference_replogle.py``: cell-level rows, ``obs.celltype`` and
``obs.gene``, and files named ``{dataset}_pred_{context}.h5ad`` plus
``{dataset}_real_{context}.h5ad``.

For each eval target (perturbation p, context c), the memory-only prediction is:

    pred_cell = sampled_control_cell(c) + D_same_gene_other_cellline(p, c)

where ``D_same_gene_other_cellline`` is built from train split deltas only and
excludes the target context.  If no other-context source exists, the default
fallback is zero delta, so the prediction becomes the sampled control cell.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import lmdb
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")
DEFAULT_REPLOGLE_ROOT = PROJECT_ROOT / "preprocessing" / "arcinstitute" / "datasets" / "State_Replogle_Filtered"
DEFAULT_VAR_DIMS_PATH = PROJECT_ROOT / "preprocessing" / "arcinstitute" / "collections" / "ST_Replogle" / "var_dims.pkl"
DEFAULT_H5AD_PATH = DEFAULT_REPLOGLE_ROOT / "only_hvg" / "Replogle_only_hvg.h5ad"


def _open_lmdb(path: Path) -> lmdb.Environment:
    if not path.exists():
        raise FileNotFoundError(path)
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        max_readers=1024,
    )


def _cell_matrix(group_or_matrix: Any) -> Any:
    if isinstance(group_or_matrix, dict):
        return group_or_matrix["cell_matrix"]
    return group_or_matrix


def _matrix_mean(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def _matrix_nrows(matrix: Any) -> int:
    return int(matrix.shape[0])


def _to_csr(matrix: Any) -> sparse.csr_matrix:
    return sparse.csr_matrix(matrix)


def _to_dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _sample_rows(matrix: Any, n_rows: int | None, rng: np.random.Generator) -> Any:
    total = _matrix_nrows(matrix)
    if n_rows is None or n_rows <= 0:
        return matrix
    ids = rng.choice(total, size=int(n_rows), replace=True)
    return matrix[ids]


def _lookup_vocab_value(value: Any, values: list[Any]) -> str | None:
    value_str = str(value)
    for idx, candidate in enumerate(values):
        if value == candidate or value_str == str(candidate):
            return str(candidate)
        if value_str.lstrip("-").isdigit() and int(value_str) == idx:
            return str(candidate)
    return None


def _resolve_context_pert(
    cartesian_key: tuple[Any, ...],
    global_keys: dict[str, list[Any]],
    *,
    context_vocab_key: str,
    pert_vocab_key: str,
    context_hint: str | None,
) -> tuple[str, str]:
    context = None
    pert = None
    batch_values = global_keys.get("batch", [])
    context_values = global_keys[context_vocab_key]
    pert_values = global_keys[pert_vocab_key]

    for value in cartesian_key:
        if batch_values and _lookup_vocab_value(value, batch_values) is not None:
            continue
        if context is None:
            context = _lookup_vocab_value(value, context_values)
        if pert is None:
            pert = _lookup_vocab_value(value, pert_values)

    if context_hint is not None and pert is not None:
        return str(context_hint), pert
    if context is not None and pert is not None:
        return context, pert
    if len(cartesian_key) >= 3:
        return str(context_hint or cartesian_key[1]), str(cartesian_key[2])
    if len(cartesian_key) >= 2:
        return str(context_hint or cartesian_key[0]), str(cartesian_key[1])
    raise ValueError(f"Invalid cartesian_key={cartesian_key!r}")


def _control_group(control_txn: lmdb.Transaction, cartesian_key: tuple[Any, ...], control_pert: str) -> Any:
    candidates: list[tuple[Any, ...]] = []
    if len(cartesian_key) >= 2:
        candidates.append(tuple([*cartesian_key[:-1], control_pert]))
    if len(cartesian_key) >= 3:
        candidates.append((cartesian_key[0], cartesian_key[1], control_pert))
    if len(cartesian_key) >= 2:
        candidates.append((cartesian_key[0], control_pert))
        candidates.append((cartesian_key[1], control_pert))

    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        buf = control_txn.get(str(key).encode("utf-8"))
        if buf is not None:
            return pickle.loads(buf)
    raise KeyError(f"Missing control for cartesian_key={cartesian_key!r}; tried {candidates!r}")


def _iter_lmdb_groups(env: lmdb.Environment, *, label: str, progress_every: int) -> Any:
    with env.begin() as txn:
        n_groups = int(txn.get(b"__len__"))
        print(f"[delta-h5ad] reading {label}: {n_groups} groups", flush=True)
        for idx in range(n_groups):
            if progress_every > 0 and idx and idx % progress_every == 0:
                print(f"[delta-h5ad] reading {label}: {idx}/{n_groups}", flush=True)
            buf = txn.get(str(idx).encode("utf-8"))
            if buf is None:
                raise KeyError(f"Missing LMDB key {idx} in {label}")
            yield pickle.loads(buf)
        print(f"[delta-h5ad] finished {label}: {n_groups}/{n_groups}", flush=True)


def _prepare_var(var_dims_path: Path | None, var_names_h5ad: Path | None) -> pd.DataFrame:
    if var_dims_path is not None:
        with var_dims_path.open("rb") as f:
            var_dims = pickle.load(f)
        return pd.DataFrame(index=[str(x) for x in var_dims["gene_names"]])
    if var_names_h5ad is not None:
        adata = ad.read_h5ad(var_names_h5ad, backed="r")
        try:
            return pd.DataFrame(index=[str(x) for x in adata.var_names])
        finally:
            adata.file.close()
    raise ValueError("Provide --var-dims-path or --var-names-h5ad")


class TrainDeltaMemory:
    def __init__(self, cell_dim: int) -> None:
        self.cell_dim = int(cell_dim)
        self.zero = np.zeros(self.cell_dim, dtype=np.float32)
        self.combo_sum: dict[tuple[str, str], np.ndarray] = {}
        self.combo_weight: dict[tuple[str, str], float] = {}
        self.global_sum: np.ndarray | None = None
        self.global_weight = 0.0

    def add(self, context: str, pert: str, delta: np.ndarray, weight: float) -> None:
        key = (str(context), str(pert))
        weighted_delta = delta.astype(np.float32, copy=False) * float(weight)
        if key in self.combo_sum:
            self.combo_sum[key] += weighted_delta
            self.combo_weight[key] += float(weight)
        else:
            self.combo_sum[key] = weighted_delta.astype(np.float32, copy=True)
            self.combo_weight[key] = float(weight)
        self.global_sum = weighted_delta.copy() if self.global_sum is None else self.global_sum + weighted_delta
        self.global_weight += float(weight)

    def combo_delta(self, context: str, pert: str) -> tuple[np.ndarray | None, float]:
        key = (str(context), str(pert))
        weight = float(self.combo_weight.get(key, 0.0))
        if weight <= 0.0:
            return None, 0.0
        return (self.combo_sum[key] / weight).astype(np.float32), weight

    def same_pert_other_context(
        self,
        target_context: str,
        pert: str,
        *,
        average_mode: str,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        source_contexts: list[str] = []
        source_cell_count = 0.0
        if average_mode == "equal_context":
            deltas = []
            for context, source_pert in sorted(self.combo_sum):
                if source_pert != pert or context == target_context:
                    continue
                delta, weight = self.combo_delta(context, source_pert)
                if delta is None:
                    continue
                deltas.append(delta)
                source_contexts.append(context)
                source_cell_count += weight
            if not deltas:
                return None, {
                    "support_context_count": 0,
                    "support_cell_count": 0.0,
                    "source_contexts": [],
                }
            return np.mean(np.vstack(deltas), axis=0).astype(np.float32), {
                "support_context_count": len(source_contexts),
                "support_cell_count": float(source_cell_count),
                "source_contexts": source_contexts,
            }

        numerator = np.zeros(self.cell_dim, dtype=np.float32)
        denominator = 0.0
        for context, source_pert in sorted(self.combo_sum):
            if source_pert != pert or context == target_context:
                continue
            weight = float(self.combo_weight[(context, source_pert)])
            numerator += self.combo_sum[(context, source_pert)]
            denominator += weight
            source_contexts.append(context)
            source_cell_count += weight
        if denominator <= 0.0:
            return None, {
                "support_context_count": 0,
                "support_cell_count": 0.0,
                "source_contexts": [],
            }
        return (numerator / denominator).astype(np.float32), {
            "support_context_count": len(source_contexts),
            "support_cell_count": float(source_cell_count),
            "source_contexts": source_contexts,
        }

    def global_delta(self) -> tuple[np.ndarray | None, float]:
        if self.global_sum is None or self.global_weight <= 0.0:
            return None, 0.0
        return (self.global_sum / self.global_weight).astype(np.float32), float(self.global_weight)


def _parse_toml_list(text: str, name: str) -> list[str]:
    match = re.search(rf"^\s*{name}\s*=\s*(\[.*\])\s*$", text, flags=re.M)
    if not match:
        raise RuntimeError(f"Missing {name!r} list in TOML")
    return [str(value) for value in ast.literal_eval(match.group(1))]


def _load_split_sets(root: Path, contexts: list[str], split_toml_template: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    val: set[tuple[str, str]] = set()
    test: set[tuple[str, str]] = set()
    for context in contexts:
        path = Path(split_toml_template.format(root=root, context=context))
        text = path.read_text()
        val.update((context, pert) for pert in _parse_toml_list(text, "val"))
        test.update((context, pert) for pert in _parse_toml_list(text, "test"))
    return val, test


def _obs_frame_from_h5ad(h5ad_path: Path, *, context_col: str, pert_col: str, contexts: list[str]) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs[[context_col, pert_col]].copy()
    finally:
        adata.file.close()
    obs[context_col] = obs[context_col].astype(str).str.lower()
    obs[pert_col] = obs[pert_col].astype(str)
    obs.reset_index(drop=True, inplace=True)
    return obs


def _limit_train_combos(
    train_combos: set[tuple[str, str]],
    *,
    contexts: list[str],
    max_train_combos_per_context: int | None,
) -> set[tuple[str, str]]:
    if max_train_combos_per_context is None:
        return train_combos
    limited: set[tuple[str, str]] = set()
    for context in contexts:
        per_context = sorted(pert for cell, pert in train_combos if cell == context)
        limited.update((context, pert) for pert in per_context[: int(max_train_combos_per_context)])
    return limited


def _build_memory_from_h5ad(
    *,
    h5ad_path: Path,
    obs: pd.DataFrame,
    contexts: list[str],
    train_combos: set[tuple[str, str]],
    control_pert: str,
    source_context_col: str,
    source_pert_col: str,
    chunk_size: int,
) -> tuple[TrainDeltaMemory, dict[str, Any]]:
    context_values = obs[source_context_col].to_numpy()
    pert_values = obs[source_pert_col].to_numpy()
    control_sums: dict[str, np.ndarray] = {}
    control_counts: dict[str, int] = {}
    train_sums: dict[tuple[str, str], np.ndarray] = {}
    train_counts: dict[tuple[str, str], int] = {}

    with h5py.File(h5ad_path, "r") as handle:
        x = handle["X"]
        n_rows = int(x.shape[0])
        if n_rows != len(obs):
            raise ValueError(f"obs/X row mismatch: obs={len(obs)} X={n_rows}")
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            matrix = np.asarray(x[start:end], dtype=np.float32)
            chunk_contexts = context_values[start:end]
            chunk_perts = pert_values[start:end]
            for context in contexts:
                context_mask = chunk_contexts == context
                if not np.any(context_mask):
                    continue
                control_mask = context_mask & (chunk_perts == control_pert)
                if np.any(control_mask):
                    control_sums[context] = control_sums.get(context, 0.0) + matrix[control_mask].sum(axis=0)
                    control_counts[context] = control_counts.get(context, 0) + int(np.count_nonzero(control_mask))
                for pert in sorted(set(chunk_perts[context_mask]) - {control_pert}):
                    combo = (context, str(pert))
                    if combo not in train_combos:
                        continue
                    mask = context_mask & (chunk_perts == pert)
                    if not np.any(mask):
                        continue
                    train_sums[combo] = train_sums.get(combo, 0.0) + matrix[mask].sum(axis=0)
                    train_counts[combo] = train_counts.get(combo, 0) + int(np.count_nonzero(mask))

    control_means = {}
    for context in contexts:
        count = control_counts.get(context, 0)
        if count <= 0:
            raise RuntimeError(f"Missing control rows for context={context!r}")
        control_means[context] = (control_sums[context] / float(count)).astype(np.float32)

    memory: TrainDeltaMemory | None = None
    context_reports: dict[str, Any] = {
        context: {
            "used_train_groups": 0,
            "used_train_cells": 0,
            "control_cells": int(control_counts.get(context, 0)),
        }
        for context in contexts
    }
    for (context, pert), pert_sum in sorted(train_sums.items()):
        count = int(train_counts[(context, pert)])
        delta = (pert_sum / float(count)).astype(np.float32) - control_means[context]
        if memory is None:
            memory = TrainDeltaMemory(cell_dim=delta.shape[0])
        memory.add(context, pert, delta, float(count))
        context_reports[context]["used_train_groups"] += 1
        context_reports[context]["used_train_cells"] += count

    if memory is None:
        raise RuntimeError("No train deltas were collected from h5ad")
    report = {
        "source": "raw_h5ad_toml_train_combos",
        "h5ad_path": str(h5ad_path),
        "contexts": context_reports,
        "num_context_pert_pairs": len(memory.combo_weight),
        "global_train_cell_count": float(memory.global_weight),
        "control_counts": {key: int(value) for key, value in control_counts.items()},
    }
    return memory, report


def _read_h5ad_rows(adata: ad.AnnData, row_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(row_indices, dtype=np.int64)
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    matrix = adata.X[unique_indices]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)[inverse]


def _sample_h5ad_indices(indices: np.ndarray, n_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if n_rows is None or n_rows <= 0:
        return indices
    return rng.choice(indices, size=int(n_rows), replace=True)


def _write_context_h5ads_from_h5ad(
    *,
    context: str,
    h5ad_path: Path,
    obs: pd.DataFrame,
    test_combos: set[tuple[str, str]],
    source_context_col: str,
    source_pert_col: str,
    control_pert: str,
    memory: TrainDeltaMemory,
    average_mode: str,
    missing_fallback: str,
    cell_sentence_len: int | None,
    seed: int,
    clip_min: float | None,
    max_targets: int | None,
    var: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
    obs_context_col: str,
    obs_pert_col: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context_values = obs[source_context_col].to_numpy()
    pert_values = obs[source_pert_col].to_numpy()
    target_perts = sorted(pert for target_context, pert in test_combos if target_context == context and pert != control_pert)
    if max_targets is not None:
        target_perts = target_perts[: int(max_targets)]
    control_indices = np.flatnonzero((context_values == context) & (pert_values == control_pert))
    if len(control_indices) == 0:
        raise RuntimeError(f"Missing control rows for context={context!r}")

    rng = np.random.default_rng(seed)
    pred_cell_chunks: list[dict[str, Any]] = []
    real_cell_chunks: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    pred_rows = 0
    real_rows = 0
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        for pert in target_perts:
            pert_indices = np.flatnonzero((context_values == context) & (pert_values == pert))
            if len(pert_indices) == 0:
                continue
            selected_pert_indices = _sample_h5ad_indices(pert_indices, cell_sentence_len, rng)
            selected_control_indices = _sample_h5ad_indices(control_indices, len(selected_pert_indices), rng)
            pert_matrix = _read_h5ad_rows(adata, selected_pert_indices)
            ctrl_matrix = _read_h5ad_rows(adata, selected_control_indices)
            delta, info = _memory_delta(
                memory=memory,
                target_context=context,
                pert=pert,
                average_mode=average_mode,
                missing_fallback=missing_fallback,
            )
            pred_matrix = ctrl_matrix + delta.reshape(1, -1)
            if clip_min is not None:
                pred_matrix = np.maximum(pred_matrix, clip_min)

            pred_key = ("h5ad", context, pert)
            ctrl_key = ("h5ad", context, control_pert)
            pred_cell_chunks.append({"cartesian_key": pred_key, "cell_matrix": pred_matrix.astype(np.float32, copy=False)})
            real_cell_chunks.append({"cartesian_key": pred_key, "cell_matrix": pert_matrix.astype(np.float32, copy=False)})
            pred_cell_chunks.append({"cartesian_key": ctrl_key, "cell_matrix": ctrl_matrix.astype(np.float32, copy=False)})
            real_cell_chunks.append({"cartesian_key": ctrl_key, "cell_matrix": ctrl_matrix.astype(np.float32, copy=False)})

            pred_rows += len(selected_pert_indices) + len(selected_control_indices)
            real_rows += len(selected_pert_indices) + len(selected_control_indices)
            mode_counts[info["mode"]] = mode_counts.get(info["mode"], 0) + 1
            support_rows.append(
                {
                    "context": context,
                    "perturbation": pert,
                    "mode": info["mode"],
                    "support_context_count": int(info["support_context_count"]),
                    "support_cell_count": float(info["support_cell_count"]),
                    "source_contexts": ";".join(info["source_contexts"]),
                    "eval_rows": len(selected_pert_indices),
                    "control_rows": len(selected_control_indices),
                }
            )
    finally:
        adata.file.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{dataset_name}_pred_{context}.h5ad"
    real_path = out_dir / f"{dataset_name}_real_{context}.h5ad"
    print(f"[delta-h5ad] building AnnData for {context}", flush=True)
    pred_adata = _build_anndata(pred_cell_chunks, var, obs_context_col=obs_context_col, obs_pert_col=obs_pert_col)
    real_adata = _build_anndata(real_cell_chunks, var, obs_context_col=obs_context_col, obs_pert_col=obs_pert_col)
    print(f"[delta-h5ad] writing {pred_path}", flush=True)
    pred_adata.write_h5ad(pred_path)
    print(f"[delta-h5ad] writing {real_path}", flush=True)
    real_adata.write_h5ad(real_path)
    summary = {
        "context": context,
        "num_eval_targets": len(target_perts),
        "pred_rows": pred_rows,
        "real_rows": real_rows,
        "mode_counts": mode_counts,
        "num_missing_zero_delta": mode_counts.get("zero_delta_missing", 0),
        "pred_path": str(pred_path),
        "real_path": str(real_path),
    }
    return summary, support_rows


def _build_memory(
    *,
    contexts: list[str],
    root: Path,
    train_lmdb_template: str,
    control_lmdb_template: str,
    global_keys: dict[str, list[Any]],
    context_vocab_key: str,
    pert_vocab_key: str,
    control_pert: str,
    progress_every: int,
) -> tuple[TrainDeltaMemory, dict[str, Any]]:
    memory: TrainDeltaMemory | None = None
    report: dict[str, Any] = {"contexts": {}, "source_split": "train"}
    for context in contexts:
        train_path = Path(train_lmdb_template.format(root=root, context=context))
        control_path = Path(control_lmdb_template.format(root=root, context=context))
        train_env = _open_lmdb(train_path)
        control_env = _open_lmdb(control_path)
        used_groups = 0
        used_cells = 0
        combo_count_before = 0 if memory is None else len(memory.combo_weight)
        try:
            with control_env.begin() as control_txn:
                for group in _iter_lmdb_groups(train_env, label=f"{context}:train", progress_every=progress_every):
                    cartesian_key = tuple(group["cartesian_key"])
                    resolved_context, pert = _resolve_context_pert(
                        cartesian_key,
                        global_keys,
                        context_vocab_key=context_vocab_key,
                        pert_vocab_key=pert_vocab_key,
                        context_hint=context,
                    )
                    if pert == control_pert:
                        continue
                    pert_matrix = _cell_matrix(group)
                    ctrl_matrix = _cell_matrix(_control_group(control_txn, cartesian_key, control_pert))
                    pert_mean = _matrix_mean(pert_matrix)
                    ctrl_mean = _matrix_mean(ctrl_matrix)
                    weight = float(_matrix_nrows(pert_matrix))
                    if memory is None:
                        memory = TrainDeltaMemory(cell_dim=pert_mean.shape[0])
                    memory.add(resolved_context, pert, pert_mean - ctrl_mean, weight)
                    used_groups += 1
                    used_cells += int(weight)
        finally:
            train_env.close()
            control_env.close()
        report["contexts"][context] = {
            "train_lmdb": str(train_path),
            "control_lmdb": str(control_path),
            "used_train_groups": used_groups,
            "used_train_cells": used_cells,
            "new_context_pert_pairs": (0 if memory is None else len(memory.combo_weight)) - combo_count_before,
        }
    if memory is None:
        raise RuntimeError("No train deltas were collected")
    report["num_context_pert_pairs"] = len(memory.combo_weight)
    report["global_train_cell_count"] = float(memory.global_weight)
    return memory, report


def _memory_delta(
    *,
    memory: TrainDeltaMemory,
    target_context: str,
    pert: str,
    average_mode: str,
    missing_fallback: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta, info = memory.same_pert_other_context(target_context, pert, average_mode=average_mode)
    if delta is not None:
        return delta, {"mode": "D_same_gene_other_cellline", **info}

    if missing_fallback == "global":
        global_delta, global_weight = memory.global_delta()
        if global_delta is not None:
            return global_delta, {
                "mode": "D_global_train_delta_fallback",
                "support_context_count": 0,
                "support_cell_count": float(global_weight),
                "source_contexts": [],
            }

    return memory.zero, {
        "mode": "zero_delta_missing",
        "support_context_count": 0,
        "support_cell_count": 0.0,
        "source_contexts": [],
    }


def _build_anndata(cell_chunks: list[dict[str, Any]], var: pd.DataFrame, *, obs_context_col: str, obs_pert_col: str) -> ad.AnnData:
    obs = []
    xs = []
    for cell_chunk in cell_chunks:
        cartesian_key = tuple(cell_chunk["cartesian_key"])
        context = cartesian_key[-2]
        pert = cartesian_key[-1]
        matrix = _cell_matrix(cell_chunk)
        obs.extend([{obs_context_col: context, obs_pert_col: pert}] * _matrix_nrows(matrix))
        xs.append(_to_csr(matrix))
    return ad.AnnData(X=sparse.vstack(xs), obs=pd.DataFrame(obs), var=var)


def _write_context_h5ads(
    *,
    context: str,
    root: Path,
    eval_lmdb_template: str,
    control_lmdb_template: str,
    global_keys: dict[str, list[Any]],
    context_vocab_key: str,
    pert_vocab_key: str,
    control_pert: str,
    memory: TrainDeltaMemory,
    average_mode: str,
    missing_fallback: str,
    cell_sentence_len: int | None,
    seed: int,
    clip_min: float | None,
    max_targets: int | None,
    var: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
    obs_context_col: str,
    obs_pert_col: str,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_path = Path(eval_lmdb_template.format(root=root, context=context))
    control_path = Path(control_lmdb_template.format(root=root, context=context))
    eval_env = _open_lmdb(eval_path)
    control_env = _open_lmdb(control_path)
    rng = np.random.default_rng(seed)
    pred_cell_chunks: list[dict[str, Any]] = []
    real_cell_chunks: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    target_count = 0
    pred_rows = 0
    real_rows = 0
    try:
        with control_env.begin() as control_txn:
            for group in _iter_lmdb_groups(eval_env, label=f"{context}:eval", progress_every=progress_every):
                if max_targets is not None and target_count >= max_targets:
                    break
                cartesian_key = tuple(group["cartesian_key"])
                resolved_context, pert = _resolve_context_pert(
                    cartesian_key,
                    global_keys,
                    context_vocab_key=context_vocab_key,
                    pert_vocab_key=pert_vocab_key,
                    context_hint=context,
                )
                pert_matrix = _sample_rows(_cell_matrix(group), cell_sentence_len, rng)
                ctrl_group = _cell_matrix(_control_group(control_txn, cartesian_key, control_pert))
                ctrl_matrix = _sample_rows(ctrl_group, _matrix_nrows(pert_matrix), rng)
                delta, info = _memory_delta(
                    memory=memory,
                    target_context=resolved_context,
                    pert=pert,
                    average_mode=average_mode,
                    missing_fallback=missing_fallback,
                )
                pred_matrix = _to_dense_float32(ctrl_matrix) + delta.reshape(1, -1)
                if clip_min is not None:
                    pred_matrix = np.maximum(pred_matrix, clip_min)

                pred_key = (*cartesian_key[:-2], resolved_context, pert) if len(cartesian_key) >= 2 else (resolved_context, pert)
                ctrl_key = (*cartesian_key[:-2], resolved_context, control_pert) if len(cartesian_key) >= 2 else (resolved_context, control_pert)
                pred_cell_chunks.append({"cartesian_key": pred_key, "cell_matrix": pred_matrix.astype(np.float32, copy=False)})
                real_cell_chunks.append({"cartesian_key": pred_key, "cell_matrix": pert_matrix})
                pred_cell_chunks.append({"cartesian_key": ctrl_key, "cell_matrix": ctrl_matrix})
                real_cell_chunks.append({"cartesian_key": ctrl_key, "cell_matrix": ctrl_matrix})

                target_count += 1
                pred_rows += _matrix_nrows(pred_matrix) + _matrix_nrows(ctrl_matrix)
                real_rows += _matrix_nrows(pert_matrix) + _matrix_nrows(ctrl_matrix)
                mode_counts[info["mode"]] = mode_counts.get(info["mode"], 0) + 1
                support_rows.append(
                    {
                        "context": resolved_context,
                        "perturbation": pert,
                        "mode": info["mode"],
                        "support_context_count": int(info["support_context_count"]),
                        "support_cell_count": float(info["support_cell_count"]),
                        "source_contexts": ";".join(info["source_contexts"]),
                        "eval_rows": _matrix_nrows(pert_matrix),
                        "control_rows": _matrix_nrows(ctrl_matrix),
                    }
                )
    finally:
        eval_env.close()
        control_env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{dataset_name}_pred_{context}.h5ad"
    real_path = out_dir / f"{dataset_name}_real_{context}.h5ad"
    print(f"[delta-h5ad] building AnnData for {context}", flush=True)
    pred_adata = _build_anndata(pred_cell_chunks, var, obs_context_col=obs_context_col, obs_pert_col=obs_pert_col)
    real_adata = _build_anndata(real_cell_chunks, var, obs_context_col=obs_context_col, obs_pert_col=obs_pert_col)
    print(f"[delta-h5ad] writing {pred_path}", flush=True)
    pred_adata.write_h5ad(pred_path)
    print(f"[delta-h5ad] writing {real_path}", flush=True)
    real_adata.write_h5ad(real_path)

    summary = {
        "context": context,
        "eval_lmdb": str(eval_path),
        "control_lmdb": str(control_path),
        "num_eval_targets": target_count,
        "pred_rows": pred_rows,
        "real_rows": real_rows,
        "mode_counts": mode_counts,
        "num_missing_zero_delta": mode_counts.get("zero_delta_missing", 0),
        "pred_path": str(pred_path),
        "real_path": str(real_path),
    }
    return summary, support_rows


def _parse_contexts(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", choices=("h5ad", "lmdb"), default="h5ad")
    parser.add_argument("--root", default=str(DEFAULT_REPLOGLE_ROOT))
    parser.add_argument("--dataset-name", default="replogle")
    parser.add_argument("--contexts", "--cell-lines", dest="contexts", default=",".join(CELL_LINES))
    parser.add_argument("--h5ad-path", default=str(DEFAULT_H5AD_PATH))
    parser.add_argument("--source-context-col", default="cell_line")
    parser.add_argument("--source-pert-col", default="gene")
    parser.add_argument("--split-toml-template", default="{root}/few_shot/{context}/{context}.toml")
    parser.add_argument("--global-keys-path", default="{root}/global_keys.pkl")
    parser.add_argument("--train-lmdb-template", default="{root}/few_shot/{context}/replogle_train_{context}")
    parser.add_argument("--eval-lmdb-template", default="{root}/few_shot/{context}/replogle_test_{context}")
    parser.add_argument("--control-lmdb-template", default="{root}/few_shot/{context}/replogle_control_{context}")
    parser.add_argument("--var-dims-path", default=str(DEFAULT_VAR_DIMS_PATH))
    parser.add_argument("--var-names-h5ad", default=None)
    parser.add_argument("--context-vocab-key", default="cell_line")
    parser.add_argument("--pert-vocab-key", default="gene")
    parser.add_argument("--obs-context-col", default="celltype")
    parser.add_argument("--obs-pert-col", default="gene")
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--average-mode", choices=("equal_context", "cell_weighted"), default="equal_context")
    parser.add_argument("--missing-fallback", choices=("zero", "global"), default="zero")
    parser.add_argument("--cell-sentence-len", type=int, default=32)
    parser.add_argument("--use-all-cells", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--max-targets-per-context", type=int, default=None)
    parser.add_argument("--max-train-combos-per-context", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    contexts = _parse_contexts(args.contexts)
    var_dims_path = None if not args.var_dims_path else Path(args.var_dims_path)
    var_names_h5ad = None if not args.var_names_h5ad else Path(args.var_names_h5ad)
    var = _prepare_var(var_dims_path, var_names_h5ad)
    cell_sentence_len = None if args.use_all_cells else int(args.cell_sentence_len)
    clip_min = None if args.no_clip else float(args.clip_min)

    out_dir = Path(args.out_dir)
    summaries = []
    all_support_rows = []
    global_keys_path: Path | None = None
    state_split_report: dict[str, Any] | None = None

    if args.input_mode == "lmdb":
        global_keys_path = Path(args.global_keys_path.format(root=root))
        with global_keys_path.open("rb") as f:
            global_keys = pickle.load(f)
        memory, memory_report = _build_memory(
            contexts=contexts,
            root=root,
            train_lmdb_template=args.train_lmdb_template,
            control_lmdb_template=args.control_lmdb_template,
            global_keys=global_keys,
            context_vocab_key=args.context_vocab_key,
            pert_vocab_key=args.pert_vocab_key,
            control_pert=args.control_pert,
            progress_every=int(args.progress_every),
        )
        for idx, context in enumerate(contexts):
            summary, support_rows = _write_context_h5ads(
                context=context,
                root=root,
                eval_lmdb_template=args.eval_lmdb_template,
                control_lmdb_template=args.control_lmdb_template,
                global_keys=global_keys,
                context_vocab_key=args.context_vocab_key,
                pert_vocab_key=args.pert_vocab_key,
                control_pert=args.control_pert,
                memory=memory,
                average_mode=args.average_mode,
                missing_fallback=args.missing_fallback,
                cell_sentence_len=cell_sentence_len,
                seed=int(args.seed) + idx,
                clip_min=clip_min,
                max_targets=args.max_targets_per_context,
                var=var,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                obs_context_col=args.obs_context_col,
                obs_pert_col=args.obs_pert_col,
                progress_every=int(args.progress_every),
            )
            summaries.append(summary)
            all_support_rows.extend(support_rows)
    else:
        h5ad_path = Path(args.h5ad_path)
        obs = _obs_frame_from_h5ad(
            h5ad_path,
            context_col=args.source_context_col,
            pert_col=args.source_pert_col,
            contexts=contexts,
        )
        actual_combos = set(
            map(
                tuple,
                obs.loc[obs[args.source_context_col].isin(contexts), [args.source_context_col, args.source_pert_col]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        val_listed, test_listed = _load_split_sets(root, contexts, args.split_toml_template)
        val_combos = actual_combos & val_listed
        test_combos = actual_combos & test_listed
        train_combos_full = actual_combos - val_listed - test_listed
        train_combos = _limit_train_combos(
            train_combos_full,
            contexts=contexts,
            max_train_combos_per_context=args.max_train_combos_per_context,
        )
        state_split_report = {
            "actual_combos": len(actual_combos),
            "train_combos": len(train_combos),
            "train_combos_before_limit": len(train_combos_full),
            "val_combos": len(val_combos),
            "test_combos": len(test_combos),
            "train_test_overlap": len(train_combos & test_combos),
            "max_train_combos_per_context": args.max_train_combos_per_context,
        }
        if train_combos & test_combos:
            raise RuntimeError(f"Train/test combo overlap is nonzero: {sorted(train_combos & test_combos)[:10]}")
        memory, memory_report = _build_memory_from_h5ad(
            h5ad_path=h5ad_path,
            obs=obs,
            contexts=contexts,
            train_combos=train_combos,
            control_pert=args.control_pert,
            source_context_col=args.source_context_col,
            source_pert_col=args.source_pert_col,
            chunk_size=int(args.chunk_size),
        )
        for idx, context in enumerate(contexts):
            summary, support_rows = _write_context_h5ads_from_h5ad(
                context=context,
                h5ad_path=h5ad_path,
                obs=obs,
                test_combos=test_combos,
                source_context_col=args.source_context_col,
                source_pert_col=args.source_pert_col,
                control_pert=args.control_pert,
                memory=memory,
                average_mode=args.average_mode,
                missing_fallback=args.missing_fallback,
                cell_sentence_len=cell_sentence_len,
                seed=int(args.seed) + idx,
                clip_min=clip_min,
                max_targets=args.max_targets_per_context,
                var=var,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                obs_context_col=args.obs_context_col,
                obs_pert_col=args.obs_pert_col,
            )
            summaries.append(summary)
            all_support_rows.extend(support_rows)

    support_csv = out_dir / "train_delta_memory_support.csv"
    with support_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "context",
                "perturbation",
                "mode",
                "support_context_count",
                "support_cell_count",
                "source_contexts",
                "eval_rows",
                "control_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(all_support_rows)

    payload = {
        "method": "train_only_delta_memory_h5ad",
        "canonical_branch": "D_same_gene_other_cellline",
        "formula": "pred_cell = sampled_control_cell + D_same_gene_other_cellline",
        "format_reference": str(PROJECT_ROOT / "main_inference_replogle.py"),
        "strict_train_only_valid": True,
        "input_mode": args.input_mode,
        "root": str(root),
        "global_keys_path": None if global_keys_path is None else str(global_keys_path),
        "h5ad_path": args.h5ad_path if args.input_mode == "h5ad" else None,
        "state_split": state_split_report,
        "contexts": contexts,
        "dataset_name": args.dataset_name,
        "average_mode": args.average_mode,
        "missing_fallback": args.missing_fallback,
        "control_pert": args.control_pert,
        "cell_sentence_len": cell_sentence_len,
        "clip_min": clip_min,
        "obs_context_col": args.obs_context_col,
        "obs_pert_col": args.obs_pert_col,
        "memory_source": memory_report,
        "cell_outputs": summaries,
        "support_csv": str(support_csv),
    }
    summary_path = out_dir / "train_delta_memory_h5ad_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
