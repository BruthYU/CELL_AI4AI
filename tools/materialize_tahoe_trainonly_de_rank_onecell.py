#!/usr/bin/env python3
"""Materialize one Tahoe-100M cellline with train-only DE-rank correction.

This is a quick validation script for a DE-rank-aware extension of the strict
train-only delta memory branch.  It builds gene weights only from the train
plate h5ad, then emits a one-cellline Tahoe h5ad pair for full cell-eval.

Formula per target context c, perturbation p, gene j:

    r[j] = score_train[j] * (delta_train_prior[j] - delta_model[j])

    mu_out[j] = ctrl_mu[j]
              + (1 - lambda_mu * score_train[j]) * delta_model[j]
              + lambda_mu * score_train[j] * delta_train_prior[j]
              + lambda_module * project_to_train_delta_modules(r)[j]

    out_i[j] = mu_out[j] + sigma_scale[j] * (pred_i[j] - mu_model[j])

No validation/test real expression or DE is read while constructing the prior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp


BASE = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow")
WORKSPACE = BASE / "benchmark/workspace/20260615_055828_JIT_FP_pdc_trainonly_20260624"
DEFAULT_INPUT_DIR = WORKSPACE / "lambda02"
DEFAULT_OUT_ROOT = WORKSPACE / "de_rank_trainonly_two_stage_onecell_20260625"
DEFAULT_TRAIN_DIR = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/datasets/by_plate/hvg")
DEFAULT_CENTER_TABLE = WORKSPACE / "tahoe_pdc_trainonly_prior_delta.npz"
DEFAULT_SUPPORT_CSV = WORKSPACE / "tahoe_pdc_trainonly_prior_delta_support.csv"
DEFAULT_TARGET_CONTEXT = "plate10_CVCL_0480"
DEFAULT_CONTROL = "[('DMSO_TF', 0.0, 'uM')]"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    if hasattr(dataset, "asstr"):
        return [str(value) for value in dataset.asstr()[:]]
    values = dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_categorical(group: h5py.Group) -> tuple[list[str], np.ndarray]:
    return read_string_dataset(group["categories"]), np.asarray(group["codes"][:])


def read_var_names(handle: h5py.File) -> list[str]:
    var = handle["var"]
    index_name = var.attrs.get("_index", "_index")
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    dataset = var[str(index_name)] if str(index_name) in var else var["_index"]
    return read_string_dataset(dataset)


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def subset_obs_group(src_obs: h5py.Group, dst: h5py.Group, row_indices: np.ndarray) -> None:
    obs = dst.create_group("obs")
    copy_attrs(src_obs.attrs, obs.attrs)
    for key in src_obs.keys():
        obj = src_obs[key]
        if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
            group = obs.create_group(key)
            copy_attrs(obj.attrs, group.attrs)
            obj.copy("categories", group)
            codes = obj["codes"][row_indices]
            dset = group.create_dataset("codes", data=codes, dtype=obj["codes"].dtype)
            copy_attrs(obj["codes"].attrs, dset.attrs)
        elif isinstance(obj, h5py.Dataset) and obj.shape and int(obj.shape[0]) >= int(row_indices.max()) + 1:
            data = obj[row_indices]
            dset = obs.create_dataset(key, data=data, dtype=obj.dtype)
            copy_attrs(obj.attrs, dset.attrs)
        else:
            src_obs.copy(key, obs)


def create_subset_h5ad_shell(
    *,
    src_path: Path,
    dst_path: Path,
    row_indices: np.ndarray,
    n_vars: int,
    dtype: np.dtype,
    chunk_rows: int,
) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        x = dst.create_dataset(
            "X",
            shape=(int(row_indices.size), int(n_vars)),
            dtype=dtype,
            chunks=(min(int(chunk_rows), max(1, int(row_indices.size))), int(n_vars)),
        )
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"
        subset_obs_group(src["obs"], dst, row_indices)
        for key in ["var", "layers", "obsm", "obsp", "uns", "varm", "varp"]:
            src.copy(key, dst)


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int, int]]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, indices.size]
    return [(int(starts[i]), int(indices[starts[i]]), int(indices[stops[i] - 1])) for i in range(len(starts))]


def read_dense_rows(handle: h5py.File, row_indices: np.ndarray) -> np.ndarray:
    x = handle["X"]
    if isinstance(x, h5py.Dataset):
        out = np.empty((int(row_indices.size), int(x.shape[1])), dtype=x.dtype)
        out_pos = 0
        for _, start, end in contiguous_runs(row_indices):
            block = x[start : end + 1, :]
            out[out_pos : out_pos + block.shape[0], :] = block
            out_pos += block.shape[0]
        return out

    encoding = x.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding != "csr_matrix":
        raise TypeError(f"Unsupported X encoding: {encoding!r}")
    shape = tuple(int(v) for v in x.attrs["shape"])
    out = np.zeros((int(row_indices.size), shape[1]), dtype=x["data"].dtype)
    indptr = x["indptr"]
    data = x["data"]
    indices = x["indices"]
    for out_pos, row_idx in enumerate(row_indices):
        start = int(indptr[row_idx])
        stop = int(indptr[row_idx + 1])
        out[out_pos, indices[start:stop]] = data[start:stop]
    return out


def csr_chunk_from_h5(x: h5py.Group, start: int, stop: int) -> sp.csr_matrix:
    shape = tuple(int(v) for v in x.attrs["shape"])
    indptr_ds = x["indptr"]
    data_ds = x["data"]
    indices_ds = x["indices"]
    old_indptr = indptr_ds[start : stop + 1]
    span_start = int(old_indptr[0])
    span_stop = int(old_indptr[-1])
    data = data_ds[span_start:span_stop]
    indices = indices_ds[span_start:span_stop]
    indptr = old_indptr - span_start
    return sp.csr_matrix((data, indices, indptr), shape=(stop - start, shape[1]))


def load_support_for_context(path: Path, target_context: str) -> tuple[list[str], dict[str, list[str]]]:
    pert_to_sources: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["context"] != target_context:
                continue
            sources = [item for item in row["source_contexts"].split(";") if item]
            pert_to_sources[row["perturbation"]] = sources
    if not pert_to_sources:
        raise RuntimeError(f"No support rows for context {target_context!r} in {path}")
    return sorted(pert_to_sources), pert_to_sources


def load_center_table(path: Path, target_context: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        contexts = [str(v) for v in data["contexts"]]
        perts = [str(v) for v in data["pert_names"]]
        for idx, (context, pert) in enumerate(zip(contexts, perts, strict=True)):
            if context == target_context:
                out[pert] = vectors[idx].astype(np.float32, copy=True)
    if not out:
        raise RuntimeError(f"No center table rows for {target_context!r}")
    return out


def infer_train_h5ad(train_dir: Path, target_context: str) -> Path:
    plate = target_context.split("_", 1)[0]
    path = train_dir / f"{plate}_hvg.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_train_de_rank_scores(
    *,
    train_h5ad: Path,
    target_context: str,
    pert_names: list[str],
    pert_to_sources: dict[str, list[str]],
    control_label: str,
    n_vars: int,
    chunk_rows: int,
    rank_scale: float,
    module_k: int,
    module_basis_cache: Path | None,
    eps: float,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, dict[str, Any]]:
    plate = target_context.split("_", 1)[0]
    raw_source_contexts = sorted({ctx for sources in pert_to_sources.values() for ctx in sources})
    source_contexts = [ctx for ctx in raw_source_contexts if ctx.startswith(f"{plate}_")]
    if not source_contexts:
        raise RuntimeError(f"No same-plate train source contexts for {target_context!r}")
    source_cell_lines = [ctx.split("_", 1)[1] for ctx in source_contexts]
    source_context_to_local = {ctx: idx for idx, ctx in enumerate(source_contexts)}
    needed_perts = [control_label, *pert_names]

    with h5py.File(train_h5ad, "r") as handle:
        genes = read_var_names(handle)
        if len(genes) != n_vars:
            raise ValueError(f"train n_vars={len(genes)} does not match pred n_vars={n_vars}")
        cell_line_categories, cell_line_codes = read_categorical(handle["obs"]["cell_line"])
        pert_categories, pert_codes = read_categorical(handle["obs"]["drugname_drugconc"])
        cell_line_to_local = {
            cell_line_categories.index(cell_line): idx
            for idx, cell_line in enumerate(source_cell_lines)
            if cell_line in cell_line_categories
        }
        pert_to_local = {
            pert_categories.index(pert): idx
            for idx, pert in enumerate(needed_perts)
            if pert in pert_categories
        }
        missing_perts = [pert for pert in needed_perts if pert not in pert_categories]
        missing_contexts = [
            f"{plate}_{cell_line}" for cell_line in source_cell_lines if cell_line not in cell_line_categories
        ]
        if missing_perts:
            raise RuntimeError(f"Missing train perturbations: {missing_perts[:10]}")
        if missing_contexts:
            raise RuntimeError(f"Missing train source contexts: {missing_contexts[:10]}")

        ctx_local = np.full(cell_line_codes.shape, -1, dtype=np.int16)
        for code, local in cell_line_to_local.items():
            ctx_local[cell_line_codes == code] = int(local)
        pert_local = np.full(pert_codes.shape, -1, dtype=np.int16)
        for code, local in pert_to_local.items():
            pert_local[pert_codes == code] = int(local)

        n_contexts = len(source_cell_lines)
        n_perts = len(needed_perts)
        n_groups = n_contexts * n_perts
        sums = np.zeros((n_groups, n_vars), dtype=np.float64)
        counts = np.zeros(n_groups, dtype=np.int64)

        x = handle["X"]
        n_obs = int(x.attrs["shape"][0]) if isinstance(x, h5py.Group) else int(x.shape[0])
        for start in range(0, n_obs, int(chunk_rows)):
            stop = min(n_obs, start + int(chunk_rows))
            ctx_block = ctx_local[start:stop]
            pert_block = pert_local[start:stop]
            keep = (ctx_block >= 0) & (pert_block >= 0)
            if not np.any(keep):
                continue
            if isinstance(x, h5py.Dataset):
                dense = np.asarray(x[start:stop, :], dtype=np.float32)[keep]
            else:
                dense = csr_chunk_from_h5(x, start, stop)[keep].toarray().astype(np.float32, copy=False)
            gids = (ctx_block[keep].astype(np.int64) * n_perts + pert_block[keep].astype(np.int64))
            bincount = np.bincount(gids, minlength=n_groups)
            counts += bincount
            for gid in np.flatnonzero(bincount):
                sums[int(gid)] += dense[gids == gid].sum(axis=0, dtype=np.float64)
            if start and start % (int(chunk_rows) * 100) == 0:
                print(f"[train-score] scanned rows={start}/{n_obs}", flush=True)

    means = np.zeros((n_contexts, n_perts, n_vars), dtype=np.float32)
    sums3 = sums.reshape(n_contexts, n_perts, n_vars)
    counts2 = counts.reshape(n_contexts, n_perts)
    valid = counts2 > 0
    means[valid] = (sums3[valid] / counts2[valid, None]).astype(np.float32)
    control_means = means[:, 0, :]

    scores: dict[str, np.ndarray] = {}
    support_counts: dict[str, int] = {}
    score_summaries: dict[str, dict[str, float]] = {}
    module_delta_blocks: list[np.ndarray] = []
    for pert_idx, pert in enumerate(pert_names, start=1):
        source_locals = [
            source_context_to_local[ctx]
            for ctx in pert_to_sources[pert]
            if ctx in source_context_to_local
        ]
        source_locals = [
            idx for idx in source_locals if counts2[idx, 0] > 0 and counts2[idx, pert_idx] > 0
        ]
        if not source_locals:
            scores[pert] = np.zeros(n_vars, dtype=np.float32)
            support_counts[pert] = 0
            continue
        delta = means[source_locals, pert_idx, :] - control_means[source_locals, :]
        if module_k > 0:
            module_delta_blocks.append(delta.astype(np.float32, copy=False))
        delta_mean = delta.mean(axis=0)
        delta_std = delta.std(axis=0)
        abs_mean = np.abs(delta_mean)
        order = np.argsort(abs_mean)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        rank_weight = 1.0 / (1.0 + ranks.astype(np.float32) / float(rank_scale))
        sign_consistency = np.abs(np.sign(delta).mean(axis=0)).astype(np.float32)
        effect_denom = float(np.quantile(abs_mean, 0.95)) + float(eps)
        effect_size = np.clip(abs_mean / effect_denom, 0.0, 1.0).astype(np.float32)
        uncertainty = 1.0 + (delta_std / (abs_mean + float(eps)))
        raw_score = rank_weight * sign_consistency * effect_size / uncertainty.astype(np.float32)
        norm = float(np.quantile(raw_score, 0.99)) + float(eps)
        score = np.clip(raw_score / norm, 0.0, 1.0).astype(np.float32)
        scores[pert] = score
        support_counts[pert] = len(source_locals)
        score_summaries[pert] = {
            "score_mean": float(score.mean()),
            "score_p95": float(np.quantile(score, 0.95)),
            "score_max": float(score.max()),
            "support_context_count": float(len(source_locals)),
        }

    module_basis: np.ndarray | None = None
    module_stats: dict[str, Any] = {
        "module_k_requested": int(module_k),
        "module_basis_cache": str(module_basis_cache) if module_basis_cache is not None else None,
    }
    if module_k > 0:
        if module_basis_cache is not None and module_basis_cache.exists():
            with np.load(module_basis_cache, allow_pickle=True) as data:
                cached_basis = np.asarray(data["basis"], dtype=np.float32)
                cached_n_vars = int(data["n_vars"])
                cached_k = int(data["module_k"])
                cached_train_delta_rows = int(data["train_delta_rows"]) if "train_delta_rows" in data else None
            if cached_n_vars != n_vars or cached_basis.shape[0] != n_vars or cached_basis.shape[1] < module_k:
                raise RuntimeError(
                    f"Module basis cache {module_basis_cache} is incompatible: "
                    f"cached_n_vars={cached_n_vars}, cached_k={cached_k}, requested_k={module_k}, n_vars={n_vars}"
                )
            module_basis = cached_basis[:, : int(module_k)].astype(np.float32, copy=True)
            module_stats.update(
                {
                    "module_basis_source": "cache",
                    "module_k": int(module_basis.shape[1]),
                    "module_train_delta_rows": cached_train_delta_rows,
                }
            )
        else:
            if not module_delta_blocks:
                raise RuntimeError("No train delta rows available to compute module basis")
            train_delta_matrix = np.concatenate(module_delta_blocks, axis=0).astype(np.float32, copy=False)
            train_delta_matrix -= train_delta_matrix.mean(axis=0, keepdims=True)
            train_delta_rows = int(train_delta_matrix.shape[0])
            use_k = min(int(module_k), int(train_delta_matrix.shape[0]), int(train_delta_matrix.shape[1]))
            print(
                f"[train-module] computing SVD rows={train_delta_matrix.shape[0]} "
                f"genes={train_delta_matrix.shape[1]} k={use_k}",
                flush=True,
            )
            _, singular_values, vt = np.linalg.svd(train_delta_matrix, full_matrices=False)
            module_basis = vt[:use_k, :].T.astype(np.float32, copy=True)
            total_var = float(np.square(singular_values).sum())
            explained = np.square(singular_values[:use_k])
            module_stats.update(
                {
                    "module_basis_source": "computed",
                    "module_k": int(use_k),
                    "module_train_delta_rows": train_delta_rows,
                    "module_singular_value_first5": [float(v) for v in singular_values[:5]],
                    "module_explained_variance_ratio": float(explained.sum() / total_var) if total_var > 0 else 0.0,
                }
            )
            if module_basis_cache is not None:
                module_basis_cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    module_basis_cache,
                    basis=module_basis,
                    n_vars=np.array(n_vars, dtype=np.int64),
                    module_k=np.array(use_k, dtype=np.int64),
                    train_delta_rows=np.array(train_delta_rows, dtype=np.int64),
                    target_context=np.array(target_context),
                    train_h5ad=np.array(str(train_h5ad)),
                )

    stats = {
        "train_h5ad": str(train_h5ad),
        "source_context_filter": f"{plate}_*",
        "num_raw_source_contexts": len(raw_source_contexts),
        "num_source_contexts": len(source_contexts),
        "num_perturbations": len(pert_names),
        "control_label": control_label,
        "min_support_contexts": int(min(support_counts.values())),
        "max_support_contexts": int(max(support_counts.values())),
        "mean_support_contexts": float(sum(support_counts.values()) / len(support_counts)),
        "score_summary_first3": {k: score_summaries[k] for k in sorted(score_summaries)[:3]},
        "module_stats": module_stats,
    }
    return scores, module_basis, stats


def get_target_rows(handle: h5py.File, target_context: str, context_col: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    categories, codes = read_categorical(handle["obs"][context_col])
    code = categories.index(target_context)
    rows = np.flatnonzero(codes == code).astype(np.int64)
    return rows, categories, codes


def compute_pred_group_means(
    pred_h5ad: Path,
    rows: np.ndarray,
    pert_codes: np.ndarray,
    pert_categories: list[str],
    chunk_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with h5py.File(pred_h5ad, "r") as handle:
        n_vars = int(handle["X"].shape[1])
        for start in range(0, int(rows.size), int(chunk_rows)):
            sub = rows[start : start + int(chunk_rows)]
            block = read_dense_rows(handle, sub).astype(np.float64, copy=False)
            block_codes = pert_codes[sub]
            for code in np.unique(block_codes):
                label = pert_categories[int(code)]
                mask = block_codes == code
                sums.setdefault(label, np.zeros(n_vars, dtype=np.float64))
                sums[label] += block[mask].sum(axis=0)
                counts[label] = counts.get(label, 0) + int(mask.sum())
    means = {label: (value / counts[label]).astype(np.float32) for label, value in sums.items()}
    return means, counts


def materialize_subset_real(src_real: Path, dst_real: Path, rows: np.ndarray, chunk_rows: int) -> None:
    with h5py.File(src_real, "r") as src:
        n_vars = int(src["X"].attrs["shape"][1]) if isinstance(src["X"], h5py.Group) else int(src["X"].shape[1])
        dtype = src["X"]["data"].dtype if isinstance(src["X"], h5py.Group) else src["X"].dtype
    create_subset_h5ad_shell(src_path=src_real, dst_path=dst_real, row_indices=rows, n_vars=n_vars, dtype=dtype, chunk_rows=chunk_rows)
    with h5py.File(src_real, "r") as src, h5py.File(dst_real, "r+") as dst:
        x_out = dst["X"]
        for start in range(0, int(rows.size), int(chunk_rows)):
            sub = rows[start : start + int(chunk_rows)]
            x_out[start : start + int(sub.size), :] = read_dense_rows(src, sub)


def materialize_corrected_pred(
    *,
    src_pred: Path,
    dst_pred: Path,
    rows: np.ndarray,
    pert_codes: np.ndarray,
    pert_categories: list[str],
    pert_names: list[str],
    control_label: str,
    pred_means: dict[str, np.ndarray],
    center_vectors: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    module_basis: np.ndarray | None,
    lambda_mu: float,
    lambda_module: float,
    sigma_mode: str,
    sigma_global: float,
    sigma_gamma: float,
    sigma_clip_min: float,
    sigma_clip_max: float | None,
    clip_min: float,
    chunk_rows: int,
) -> dict[str, Any]:
    with h5py.File(src_pred, "r") as src:
        n_vars = int(src["X"].shape[1])
        dtype = src["X"].dtype
    create_subset_h5ad_shell(src_path=src_pred, dst_path=dst_pred, row_indices=rows, n_vars=n_vars, dtype=dtype, chunk_rows=chunk_rows)
    ctrl_mu = pred_means[control_label]

    corrected_params: dict[str, dict[str, float]] = {}
    per_pert_mu: dict[str, np.ndarray] = {}
    per_pert_sigma: dict[str, np.ndarray] = {}
    for pert in pert_names:
        score = scores[pert]
        mix = np.clip(float(lambda_mu) * score, 0.0, 1.0).astype(np.float32)
        if sigma_mode == "global":
            sigma = np.full_like(score, float(sigma_global), dtype=np.float32)
        elif sigma_mode == "score-gamma":
            sigma = (1.0 + float(sigma_gamma) * score).astype(np.float32)
        else:
            raise ValueError(f"Unsupported sigma_mode: {sigma_mode!r}")
        sigma = np.maximum(sigma, float(sigma_clip_min))
        if sigma_clip_max is not None:
            sigma = np.minimum(sigma, float(sigma_clip_max))
        delta_model = pred_means[pert] - ctrl_mu
        delta_prior = center_vectors[pert]
        prior_residual = score * (delta_prior - delta_model)
        if module_basis is not None and float(lambda_module) != 0.0:
            module_residual = (prior_residual @ module_basis) @ module_basis.T
        else:
            module_residual = np.zeros_like(delta_model, dtype=np.float32)
        mu_out = ctrl_mu + (1.0 - mix) * delta_model + mix * delta_prior + float(lambda_module) * module_residual
        per_pert_mu[pert] = mu_out.astype(np.float32)
        per_pert_sigma[pert] = sigma.astype(np.float32)
        corrected_params[pert] = {
            "score_mean": float(score.mean()),
            "score_p95": float(np.quantile(score, 0.95)),
            "mix_mean": float(mix.mean()),
            "mix_p95": float(np.quantile(mix, 0.95)),
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "sigma_max": float(sigma.max()),
            "delta_model_abs_mean": float(np.abs(delta_model).mean()),
            "prior_abs_mean": float(np.abs(delta_prior).mean()),
            "module_residual_abs_mean": float(np.abs(module_residual).mean()),
            "module_residual_abs_p95": float(np.quantile(np.abs(module_residual), 0.95)),
        }

    post_clip_sums: dict[str, np.ndarray] = {
        pert: np.zeros(n_vars, dtype=np.float64) for pert in pert_names
    }
    post_clip_counts: dict[str, int] = {pert: 0 for pert in pert_names}
    with h5py.File(src_pred, "r") as src, h5py.File(dst_pred, "r+") as dst:
        x_out = dst["X"]
        for start in range(0, int(rows.size), int(chunk_rows)):
            sub = rows[start : start + int(chunk_rows)]
            block = read_dense_rows(src, sub).astype(np.float32, copy=False)
            out = np.empty_like(block)
            block_codes = pert_codes[sub]
            for code in np.unique(block_codes):
                label = pert_categories[int(code)]
                mask = block_codes == code
                if label == control_label or label not in per_pert_mu:
                    out[mask] = block[mask]
                    continue
                residual = block[mask] - pred_means[label]
                out[mask] = per_pert_mu[label] + per_pert_sigma[label] * residual
            if clip_min is not None:
                out = np.maximum(out, float(clip_min))
            for code in np.unique(block_codes):
                label = pert_categories[int(code)]
                if label in post_clip_sums:
                    mask = block_codes == code
                    post_clip_sums[label] += out[mask].sum(axis=0, dtype=np.float64)
                    post_clip_counts[label] += int(mask.sum())
            x_out[start : start + int(sub.size), :] = out

    drift_summaries: dict[str, dict[str, float]] = {}
    drift_abs_all: list[np.ndarray] = []
    for pert in pert_names:
        if post_clip_counts[pert] == 0:
            continue
        post_mu = (post_clip_sums[pert] / post_clip_counts[pert]).astype(np.float32)
        drift_abs = np.abs(post_mu - per_pert_mu[pert])
        drift_abs_all.append(drift_abs)
        drift_summaries[pert] = {
            "post_clip_mu_drift_abs_mean": float(drift_abs.mean()),
            "post_clip_mu_drift_abs_p95": float(np.quantile(drift_abs, 0.95)),
            "post_clip_mu_drift_abs_max": float(drift_abs.max()),
        }
    if drift_abs_all:
        drift_concat = np.concatenate(drift_abs_all)
        drift_summary_all = {
            "post_clip_mu_drift_abs_mean": float(drift_concat.mean()),
            "post_clip_mu_drift_abs_p95": float(np.quantile(drift_concat, 0.95)),
            "post_clip_mu_drift_abs_max": float(drift_concat.max()),
        }
    else:
        drift_summary_all = {}
    return {
        "num_corrected_perturbations": len(per_pert_mu),
        "corrected_param_first3": {k: corrected_params[k] for k in sorted(corrected_params)[:3]},
        "post_clip_mu_drift_all": drift_summary_all,
        "post_clip_mu_drift_first3": {k: drift_summaries[k] for k in sorted(drift_summaries)[:3]},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-context", default=DEFAULT_TARGET_CONTEXT)
    parser.add_argument("--base-pred-h5ad", type=Path, default=DEFAULT_INPUT_DIR / "tahoe100m_pred.h5ad")
    parser.add_argument("--real-h5ad", type=Path, default=DEFAULT_INPUT_DIR / "tahoe100m_real.h5ad")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--center-table", type=Path, default=DEFAULT_CENTER_TABLE)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--reuse-real-h5ad", type=Path, default=None)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--train-chunk-rows", type=int, default=16384)
    parser.add_argument("--rank-scale", type=float, default=200.0)
    parser.add_argument("--lambda-mu", type=float, default=0.0)
    parser.add_argument("--lambda-module", type=float, default=0.0)
    parser.add_argument("--module-k", type=int, default=0)
    parser.add_argument("--module-basis-cache", type=Path, default=None)
    parser.add_argument("--sigma-mode", choices=["global", "score-gamma"], default="global")
    parser.add_argument("--sigma-global", type=float, default=1.0)
    parser.add_argument("--sigma-gamma", type=float, default=0.0)
    parser.add_argument("--sigma-clip-min", type=float, default=0.0)
    parser.add_argument("--sigma-clip-max", type=float, default=None)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_context = str(args.target_context)
    run_id = args.run_id or target_context
    out_dir = args.out_root / run_id
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred = out_dir / "tahoe100m_pred.h5ad"
    out_real = out_dir / "tahoe100m_real.h5ad"
    manifest_path = out_dir / "de_rank_trainonly_manifest.json"
    if (out_pred.exists() or out_real.exists()) and not args.force:
        raise FileExistsError(f"{out_dir} already contains h5ad outputs; use --force")

    pert_names, pert_to_sources = load_support_for_context(args.support_csv, target_context)
    center_vectors = load_center_table(args.center_table, target_context)
    pert_names = [pert for pert in pert_names if pert in center_vectors]
    if not pert_names:
        raise RuntimeError("No perturbations with center vectors")

    with h5py.File(args.base_pred_h5ad, "r") as pred:
        rows, _, _ = get_target_rows(pred, target_context, args.context_col)
        pert_categories, pert_codes = read_categorical(pred["obs"][args.pert_col])
        genes = read_var_names(pred)
        n_vars = len(genes)
    print(f"[target] {target_context}: rows={rows.size}, perts={len(pert_names)}", flush=True)

    train_h5ad = infer_train_h5ad(args.train_dir, target_context)
    scores, module_basis, train_stats = build_train_de_rank_scores(
        train_h5ad=train_h5ad,
        target_context=target_context,
        pert_names=pert_names,
        pert_to_sources=pert_to_sources,
        control_label=args.control_label,
        n_vars=n_vars,
        chunk_rows=int(args.train_chunk_rows),
        rank_scale=float(args.rank_scale),
        module_k=int(args.module_k),
        module_basis_cache=args.module_basis_cache,
        eps=float(args.eps),
    )

    print("[pred] computing group means", flush=True)
    pred_means, pred_counts = compute_pred_group_means(
        args.base_pred_h5ad,
        rows,
        pert_codes,
        pert_categories,
        chunk_rows=int(args.chunk_rows),
    )
    missing = [pert for pert in [args.control_label, *pert_names] if pert not in pred_means]
    if missing:
        raise RuntimeError(f"Missing target perturbations in pred h5ad: {missing[:10]}")

    if args.reuse_real_h5ad is not None:
        print("[real] reusing one-cellline real h5ad", flush=True)
        reuse_real = args.reuse_real_h5ad.resolve()
        if not reuse_real.exists():
            raise FileNotFoundError(reuse_real)
        os.symlink(str(reuse_real), out_real)
        with h5py.File(out_real, "r") as real:
            real_rows = np.arange(int(real["X"].shape[0]), dtype=np.int64)
    else:
        print("[real] writing one-cellline real h5ad", flush=True)
        with h5py.File(args.real_h5ad, "r") as real:
            real_rows, _, _ = get_target_rows(real, target_context, args.context_col)
        materialize_subset_real(args.real_h5ad, out_real, real_rows, chunk_rows=int(args.chunk_rows))

    print("[pred] writing corrected one-cellline pred h5ad", flush=True)
    correction_stats = materialize_corrected_pred(
        src_pred=args.base_pred_h5ad,
        dst_pred=out_pred,
        rows=rows,
        pert_codes=pert_codes,
        pert_categories=pert_categories,
        pert_names=pert_names,
        control_label=args.control_label,
        pred_means=pred_means,
        center_vectors=center_vectors,
        scores=scores,
        module_basis=module_basis,
        lambda_mu=float(args.lambda_mu),
        lambda_module=float(args.lambda_module),
        sigma_mode=str(args.sigma_mode),
        sigma_global=float(args.sigma_global),
        sigma_gamma=float(args.sigma_gamma),
        sigma_clip_min=float(args.sigma_clip_min),
        sigma_clip_max=args.sigma_clip_max,
        clip_min=float(args.clip_min),
        chunk_rows=int(args.chunk_rows),
    )

    eval_command = (
        f"bash {BASE / 'submit_evaluate_tahoe_rjob.sh'} {out_dir} "
        f"tahoe100m-de-rank-onecell-{target_context}"
    )
    manifest = {
        "method": "train-only DE-rank gene-wise mean correction plus residual scale",
        "target_context": target_context,
        "train_only": True,
        "no_val_or_test_real_used_for_prior": True,
        "base_pred_h5ad": str(args.base_pred_h5ad),
        "real_h5ad": str(args.real_h5ad),
        "out_dir": str(out_dir),
        "out_pred_h5ad": str(out_pred),
        "out_real_h5ad": str(out_real),
        "reuse_real_h5ad": str(args.reuse_real_h5ad) if args.reuse_real_h5ad is not None else None,
        "support_csv": str(args.support_csv),
        "center_table": str(args.center_table),
        "formula": "r = score_train * (delta_train_prior - delta_model); mu_out = ctrl_mu + (1 - lambda_mu * score_train) * delta_model + lambda_mu * score_train * delta_train_prior + lambda_module * project_train_delta_modules(r); out_i = mu_out + sigma_scale * residual_model",
        "params": {
            "rank_scale": float(args.rank_scale),
            "lambda_mu": float(args.lambda_mu),
            "lambda_module": float(args.lambda_module),
            "module_k": int(args.module_k),
            "module_basis_cache": str(args.module_basis_cache) if args.module_basis_cache is not None else None,
            "sigma_mode": str(args.sigma_mode),
            "sigma_global": float(args.sigma_global),
            "sigma_gamma": float(args.sigma_gamma),
            "sigma_clip_min": float(args.sigma_clip_min),
            "sigma_clip_max": args.sigma_clip_max,
            "clip_min": float(args.clip_min),
        },
        "target_rows_pred": int(rows.size),
        "target_rows_real": int(real_rows.size),
        "pred_counts": {k: int(v) for k, v in pred_counts.items() if k == args.control_label or k in pert_names[:5]},
        "train_score_stats": train_stats,
        "correction_stats": correction_stats,
        "suggested_eval_command": eval_command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
