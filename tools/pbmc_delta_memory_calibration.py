#!/usr/bin/env python3
"""Build and evaluate a PBMC train/val perturbation-effect memory prior.

This mirrors the Replogle delta-memory workflow, but keys the memory by
``(celltype, cytokine)`` for PBMC:

    final = control_test_mean
          + model_weight * (model_pred_mean - control_test_mean)
          + prior_weight * trainval_delta(celltype, cytokine)

The ``calibrate`` command never uses test perturbed expression to build the
prior. The ``donor-proxy`` command is only a diagnostic upper-bound style proxy:
it uses held-out-file real deltas from other donors and must not be reported as
the official test result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import lmdb
import numpy as np
from scipy import sparse


DEFAULT_ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow")
DEFAULT_SPLIT_ROOT = (
    DEFAULT_ROOT
    / "preprocessing/arcinstitute/datasets/State_Parse_Filtered/few_shot/split_4"
)
DEFAULT_MODEL_WORKSPACE = (
    DEFAULT_ROOT / "benchmark/workspace/20260610_095214_PRIOR_LLM"
)
DEFAULT_RAW_H5AD = (
    DEFAULT_ROOT
    / "preprocessing/arcinstitute/datasets/State_Parse_Filtered/only_hvg/PBMC_only_hvg.h5ad"
)
DEFAULT_SPLIT_TOML = (
    DEFAULT_ROOT
    / "preprocessing/arcinstitute/datasets/State_Parse_Filtered/few_shot/split_4.toml"
)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.astype(bytes).decode()
    return str(value)


def _obs_codes(file: h5py.File, column: str) -> tuple[np.ndarray, list[str]]:
    obj = file["obs"][column]
    if isinstance(obj, h5py.Group):
        codes = obj["codes"][:].astype(np.int64, copy=False)
        categories = [_decode(x) for x in obj["categories"][:]]
        return codes, categories

    values = [_decode(x) for x in obj[:]]
    categories = list(dict.fromkeys(values))
    lookup = {value: idx for idx, value in enumerate(categories)}
    codes = np.asarray([lookup[value] for value in values], dtype=np.int64)
    return codes, categories


def _x_shape(file: h5py.File) -> tuple[int, int]:
    x = file["X"]
    if isinstance(x, h5py.Group):
        return tuple(int(v) for v in x.attrs["shape"])  # type: ignore[return-value]
    return int(x.shape[0]), int(x.shape[1])


def _var_names(h5ad_path: Path) -> list[str]:
    with h5py.File(h5ad_path, "r") as f:
        if "_index" in f["var"]:
            return [_decode(x) for x in f["var"]["_index"][:]]
    with h5py.File(h5ad_path, "r") as f:
        return [f"gene_{i}" for i in range(_x_shape(f)[1])]


def _group_means(
    h5ad_path: Path,
    key_cols: list[str],
    *,
    label: str | None = None,
    progress_every: int = 655_360,
) -> dict[tuple[str, ...], np.ndarray]:
    """Compute grouped means from dense or CSR h5ad without densifying X."""
    start_time = time.time()
    with h5py.File(h5ad_path, "r") as f:
        n_rows, n_vars = _x_shape(f)
        codes_by_col: list[np.ndarray] = []
        categories_by_col: list[list[str]] = []
        for col in key_cols:
            codes, categories = _obs_codes(f, col)
            if len(codes) != n_rows:
                raise ValueError(f"{h5ad_path}: obs/{col} length != X rows")
            codes_by_col.append(codes)
            categories_by_col.append(categories)

        lookup: dict[tuple[str, ...], int] = {}
        keys: list[tuple[str, ...]] = []
        group_ids = np.empty(n_rows, dtype=np.int32)
        for row_idx in range(n_rows):
            key = tuple(
                categories_by_col[col_idx][int(codes_by_col[col_idx][row_idx])]
                for col_idx in range(len(key_cols))
            )
            group_id = lookup.get(key)
            if group_id is None:
                group_id = len(keys)
                lookup[key] = group_id
                keys.append(key)
            group_ids[row_idx] = group_id

        counts = np.bincount(group_ids, minlength=len(keys)).astype(np.float64)
        sums = np.zeros((len(keys), n_vars), dtype=np.float64)

        x = f["X"]
        if isinstance(x, h5py.Group) and x.attrs.get("encoding-type") == "csr_matrix":
            data = x["data"]
            indices = x["indices"]
            indptr = x["indptr"][:]
            for row_idx in range(n_rows):
                start = int(indptr[row_idx])
                end = int(indptr[row_idx + 1])
                if end > start:
                    sums[int(group_ids[row_idx]), indices[start:end]] += data[start:end]
                if label and (row_idx + 1 == n_rows or (row_idx + 1) % progress_every == 0):
                    print(
                        f"[means] {label}: {row_idx + 1:,}/{n_rows:,} rows, "
                        f"elapsed {time.time() - start_time:.1f}s",
                        flush=True,
                    )
        else:
            chunk_size = 8192
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                block = np.asarray(x[start:end], dtype=np.float64)
                for local_idx, row in enumerate(block):
                    sums[int(group_ids[start + local_idx])] += row
                if label and (end == n_rows or end % progress_every == 0):
                    print(
                        f"[means] {label}: {end:,}/{n_rows:,} rows, "
                        f"elapsed {time.time() - start_time:.1f}s",
                        flush=True,
                    )

    means = sums / np.maximum(counts[:, None], 1.0)
    return {key: means[idx].astype(np.float32) for idx, key in enumerate(keys)}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _parse_pbmc_split_toml(path: Path) -> dict[str, dict[str, set[str]]]:
    """Parse the small PBMC fewshot TOML without requiring Python 3.11 tomllib."""
    text = path.read_text()
    out: dict[str, dict[str, set[str]]] = {}
    pattern = re.compile(r'^\[fewshot\."parse\.([^"]+)"\]\s*$', re.M)
    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        raw_celltype = match.group(1)
        celltype = raw_celltype.replace("_", " ")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        entry: dict[str, set[str]] = {}
        for split in ("val", "test"):
            split_match = re.search(rf"^{split}\s*=\s*(\[.*\])", block, re.M)
            entry[split] = set(ast.literal_eval(split_match.group(1))) if split_match else set()
        out[celltype] = entry
    return out


def _split_for_group(celltype: str, cytokine: str, fewshot: dict[str, dict[str, set[str]]]) -> str:
    entry = fewshot.get(celltype)
    if entry is None:
        return "train"
    # Match the original notebook: val branch is checked before test.
    if cytokine in entry.get("val", set()):
        return "val"
    if cytokine in entry.get("test", set()):
        return "test"
    return "train"


def _distance_scores(pred_deltas: np.ndarray, real_deltas: np.ndarray) -> dict[str, np.ndarray]:
    scores: dict[str, list[float]] = {"l1": [], "l2": [], "cosine": []}
    real_norms = np.linalg.norm(real_deltas, axis=1)
    for idx, pred in enumerate(pred_deltas):
        l1 = np.abs(real_deltas - pred).sum(axis=1)
        l2 = np.linalg.norm(real_deltas - pred, axis=1)
        pred_norm = np.linalg.norm(pred)
        denom = np.maximum(real_norms * pred_norm, 1e-12)
        cosine = 1.0 - (real_deltas @ pred) / denom
        for name, distances in (("l1", l1), ("l2", l2), ("cosine", cosine)):
            correct = distances[idx]
            scores[name].append(float(np.mean(distances >= correct - 1e-12)))
    return {key: np.asarray(value, dtype=np.float64) for key, value in scores.items()}


def _write_metric_tables(rows: list[dict[str, Any]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "celltype",
        "cytokine",
        "pearson_delta",
        "mse",
        "mae",
        "mse_delta",
        "mae_delta",
        "discrimination_score_l1",
        "discrimination_score_l2",
        "discrimination_score_cosine",
    ]
    with (outdir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    metric_names = fieldnames[2:]
    stat_rows: list[dict[str, Any]] = []
    for stat in ("count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        out: dict[str, Any] = {"statistic": stat}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if stat == "count":
                out[metric] = float(len(finite))
            elif stat == "null_count":
                out[metric] = float(len(values) - len(finite))
            elif len(finite) == 0:
                out[metric] = float("nan")
            elif stat == "mean":
                out[metric] = float(np.mean(finite))
            elif stat == "std":
                out[metric] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            elif stat == "min":
                out[metric] = float(np.min(finite))
            elif stat == "25%":
                out[metric] = float(np.quantile(finite, 0.25))
            elif stat == "50%":
                out[metric] = float(np.quantile(finite, 0.50))
            elif stat == "75%":
                out[metric] = float(np.quantile(finite, 0.75))
            elif stat == "max":
                out[metric] = float(np.max(finite))
        stat_rows.append(out)

    with (outdir / "agg_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["statistic", *metric_names])
        writer.writeheader()
        writer.writerows(stat_rows)


def score_h5ads(
    *,
    pred_h5ad: Path,
    real_h5ad: Path,
    outdir: Path,
    control_pert: str,
    celltype_col: str,
    pert_col: str,
) -> dict[str, Any]:
    pred = _group_means(pred_h5ad, [celltype_col, pert_col], label="pred-score")
    real = _group_means(real_h5ad, [celltype_col, pert_col], label="real-score")

    rows: list[dict[str, Any]] = []
    celltypes = sorted({key[0] for key in pred} & {key[0] for key in real})
    for celltype in celltypes:
        pred_control = pred.get((celltype, control_pert))
        real_control = real.get((celltype, control_pert))
        if pred_control is None or real_control is None:
            continue
        perts = sorted(
            {
                pert
                for ct, pert in pred
                if ct == celltype and pert != control_pert and (celltype, pert) in real
            }
        )
        if not perts:
            continue
        pred_deltas = np.vstack([pred[(celltype, pert)] - pred_control for pert in perts])
        real_deltas = np.vstack([real[(celltype, pert)] - real_control for pert in perts])
        discr = _distance_scores(pred_deltas, real_deltas)
        for idx, pert in enumerate(perts):
            pred_mean = pred[(celltype, pert)]
            real_mean = real[(celltype, pert)]
            pred_delta = pred_deltas[idx]
            real_delta = real_deltas[idx]
            rows.append(
                {
                    "celltype": celltype,
                    "cytokine": pert,
                    "pearson_delta": _pearson(pred_delta, real_delta),
                    "mse": float(np.mean((pred_mean - real_mean) ** 2)),
                    "mae": float(np.mean(np.abs(pred_mean - real_mean))),
                    "mse_delta": float(np.mean((pred_delta - real_delta) ** 2)),
                    "mae_delta": float(np.mean(np.abs(pred_delta - real_delta))),
                    "discrimination_score_l1": float(discr["l1"][idx]),
                    "discrimination_score_l2": float(discr["l2"][idx]),
                    "discrimination_score_cosine": float(discr["cosine"][idx]),
                }
            )

    _write_metric_tables(rows, outdir)
    summary = {
        "pred_h5ad": str(pred_h5ad),
        "real_h5ad": str(real_h5ad),
        "outdir": str(outdir),
        "control_pert": control_pert,
        "celltype_col": celltype_col,
        "pert_col": pert_col,
        "n_perturbations": len(rows),
        "celltypes": sorted({row["celltype"] for row in rows}),
        "mean_pearson_delta": float(np.nanmean([row["pearson_delta"] for row in rows])) if rows else float("nan"),
    }
    (outdir / "strict_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )


def _cell_matrix(group_or_matrix: Any) -> Any:
    if isinstance(group_or_matrix, dict):
        return group_or_matrix["cell_matrix"]
    return group_or_matrix


def _matrix_mean(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)


def _matrix_nrows(matrix: Any) -> int:
    return int(matrix.shape[0])


def _load_group(txn: lmdb.Transaction, key: Any) -> Any:
    buf = txn.get(str(key).encode())
    if buf is None:
        raise KeyError(key)
    return pickle.loads(buf)


def _control_group(control_txn: lmdb.Transaction, donor: str, celltype: str, control_pert: str) -> Any:
    candidates = (
        (donor, celltype, control_pert),
        (str(donor), str(celltype), control_pert),
    )
    for key in candidates:
        buf = control_txn.get(str(key).encode())
        if buf is not None:
            return pickle.loads(buf)
    raise KeyError(f"Missing control for donor={donor!r}, celltype={celltype!r}")


def build_trainval_prior(
    *,
    source_lmdbs: list[Path],
    control_lmdb: Path,
    control_pert: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int]]:
    sum_delta: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    control_env = _open_lmdb(control_lmdb)
    source_envs = [(path, _open_lmdb(path)) for path in source_lmdbs]
    try:
        with control_env.begin() as control_txn:
            for path, env in source_envs:
                with env.begin() as txn:
                    n_raw = txn.get(b"__len__")
                    if n_raw is None:
                        raise KeyError(f"{path} missing __len__")
                    n = int(n_raw)
                    print(f"[prior] reading {path}: {n:,} groups", flush=True)
                    for idx in range(n):
                        if idx and idx % 100 == 0:
                            print(f"[prior] {path.name}: {idx:,}/{n:,}", flush=True)
                        group = _load_group(txn, idx)
                        donor, celltype, cytokine = tuple(group["cartesian_key"])
                        if cytokine == control_pert:
                            continue
                        pert_matrix = _cell_matrix(group)
                        ctrl_matrix = _cell_matrix(_control_group(control_txn, donor, celltype, control_pert))
                        nrows = _matrix_nrows(pert_matrix)
                        delta = (_matrix_mean(pert_matrix) - _matrix_mean(ctrl_matrix)) * nrows
                        key = (str(celltype), str(cytokine))
                        if key in sum_delta:
                            sum_delta[key] += delta
                            counts[key] += nrows
                        else:
                            sum_delta[key] = delta.astype(np.float64, copy=True)
                            counts[key] = nrows
                    print(f"[prior] finished {path}: {n:,}/{n:,}", flush=True)
    finally:
        for _, env in source_envs:
            env.close()
        control_env.close()

    prior = {key: (total / counts[key]).astype(np.float32) for key, total in sum_delta.items()}
    return prior, counts


def _write_string_array(group: h5py.Group, name: str, values: Iterable[str]) -> h5py.Dataset:
    dtype = h5py.string_dtype(encoding="utf-8")
    dataset = group.create_dataset(name, data=np.asarray(list(values), dtype=object), dtype=dtype)
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"
    return dataset


def _write_compact_h5ad(
    path: Path,
    *,
    rows: list[np.ndarray],
    obs_rows: list[dict[str, str]],
    var_names: list[str],
) -> None:
    matrix = sparse.csr_matrix(np.vstack(rows).astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
        x.create_dataset("data", data=matrix.data)
        x.create_dataset("indices", data=matrix.indices)
        x.create_dataset("indptr", data=matrix.indptr)

        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["_index"] = "_index"
        columns = ["celltype", "cell_type", "cytokine", "donor"]
        obs.attrs["column-order"] = np.asarray([col.encode() for col in columns])
        _write_string_array(obs, "_index", [str(i) for i in range(matrix.shape[0])])
        for col in columns:
            _write_string_array(obs, col, [row[col] for row in obs_rows])

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["_index"] = "_index"
        var.attrs["column-order"] = np.asarray([], dtype="S")
        _write_string_array(var, "_index", var_names)

        for name in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            group = f.create_group(name)
            group.attrs["encoding-type"] = "dict"
            group.attrs["encoding-version"] = "0.1.0"


def _save_prior_npz(path: Path, prior: dict[tuple[str, str], np.ndarray], counts: dict[tuple[str, str], int]) -> None:
    keys = sorted(prior)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        celltype=np.asarray([key[0] for key in keys], dtype=object),
        cytokine=np.asarray([key[1] for key in keys], dtype=object),
        delta=np.vstack([prior[key] for key in keys]).astype(np.float32) if keys else np.empty((0, 0), dtype=np.float32),
        counts=np.asarray([counts[key] for key in keys], dtype=np.int64),
    )


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    source_lmdbs = [Path(x) for x in args.source_lmdb]
    prior, counts = build_trainval_prior(
        source_lmdbs=source_lmdbs,
        control_lmdb=Path(args.control_lmdb),
        control_pert=args.control_pert,
    )

    model_pred = Path(args.model_pred)
    model_real = Path(args.model_real)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_prior_npz(out_dir / "pbmc_trainval_delta_prior.npz", prior, counts)

    model_means = _group_means(
        model_pred,
        [args.celltype_col, args.pert_col],
        label="model-calibrate",
    )
    var_names = _var_names(model_pred)
    celltypes = sorted({key[0] for key in model_means})

    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, str]] = []
    missing_prior: list[tuple[str, str]] = []
    calibrated_pairs = 0
    for celltype in celltypes:
        control_mean = model_means.get((celltype, args.control_pert))
        if control_mean is None:
            print(f"[calibrate] skip {celltype}: missing control {args.control_pert}", file=sys.stderr)
            continue
        rows.append(control_mean.astype(np.float32))
        obs_rows.append(
            {
                "celltype": celltype,
                "cell_type": celltype,
                "cytokine": args.control_pert,
                "donor": "memory_prior",
            }
        )
        perts = sorted(pert for ct, pert in model_means if ct == celltype and pert != args.control_pert)
        for pert in perts:
            model_delta = model_means[(celltype, pert)] - control_mean
            prior_delta = prior.get((celltype, pert))
            if prior_delta is None:
                missing_prior.append((celltype, pert))
                if args.missing_prior_action == "skip":
                    continue
                if args.missing_prior_action == "zero":
                    final_delta = args.model_weight * model_delta
                else:
                    final_delta = model_delta
            else:
                calibrated_pairs += 1
                final_delta = args.model_weight * model_delta + args.prior_weight * prior_delta
            pred = control_mean + final_delta
            if args.clip_min is not None:
                pred = np.maximum(pred, args.clip_min)
            rows.append(pred.astype(np.float32))
            obs_rows.append(
                {
                    "celltype": celltype,
                    "cell_type": celltype,
                    "cytokine": pert,
                    "donor": "memory_prior",
                }
            )

    pred_out = out_dir / "pbmc_pred_memory_m04_p06.h5ad"
    real_out = out_dir / "pbmc_real_memory_reference.h5ad"
    _write_compact_h5ad(pred_out, rows=rows, obs_rows=obs_rows, var_names=var_names)
    if real_out.exists() or real_out.is_symlink():
        real_out.unlink()
    os.symlink(model_real.resolve(), real_out)

    summary = {
        "model_pred": str(model_pred),
        "model_real": str(model_real),
        "pred_out": str(pred_out),
        "real_out": str(real_out),
        "source_lmdbs": [str(path) for path in source_lmdbs],
        "control_lmdb": str(args.control_lmdb),
        "control_pert": args.control_pert,
        "model_weight": args.model_weight,
        "prior_weight": args.prior_weight,
        "clip_min": args.clip_min,
        "missing_prior_action": args.missing_prior_action,
        "num_prior_pairs": len(prior),
        "num_calibrated_pairs": calibrated_pairs,
        "num_missing_prior_pairs": len(missing_prior),
        "missing_prior_pairs": missing_prior[:50],
    }
    (out_dir / "delta_memory_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    if args.score:
        summary["base_score"] = score_h5ads(
            pred_h5ad=model_pred,
            real_h5ad=model_real,
            outdir=out_dir / "base_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
        summary["calibrated_score"] = score_h5ads(
            pred_h5ad=pred_out,
            real_h5ad=model_real,
            outdir=out_dir / "calibrated_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
        (out_dir / "delta_memory_calibration_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def calibrate_from_prior_h5ad(args: argparse.Namespace) -> dict[str, Any]:
    """Blend model means with a prior computed from another h5ad.

    This is useful when a validation/reference h5ad already exists. If the
    prior h5ad overlaps the test real h5ad, the result is a leaky diagnostic.
    """
    model_pred = Path(args.model_pred)
    model_real = Path(args.model_real)
    prior_h5ad = Path(args.prior_h5ad)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_means = _group_means(
        model_pred,
        [args.celltype_col, args.pert_col],
        label="model-h5prior-calibrate",
    )
    prior_means = _group_means(
        prior_h5ad,
        [args.celltype_col, args.pert_col],
        label="prior-h5ad-calibrate",
    )
    var_names = _var_names(model_pred)
    celltypes = sorted({key[0] for key in model_means})

    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, str]] = []
    missing_prior: list[tuple[str, str]] = []
    calibrated_pairs = 0
    for celltype in celltypes:
        model_control = model_means.get((celltype, args.control_pert))
        prior_control = prior_means.get((celltype, args.control_pert))
        if model_control is None:
            print(f"[h5prior] skip {celltype}: missing model control", file=sys.stderr)
            continue
        rows.append(model_control.astype(np.float32))
        obs_rows.append(
            {
                "celltype": celltype,
                "cell_type": celltype,
                "cytokine": args.control_pert,
                "donor": "h5ad_memory_prior",
            }
        )
        perts = sorted(pert for ct, pert in model_means if ct == celltype and pert != args.control_pert)
        for pert in perts:
            model_delta = model_means[(celltype, pert)] - model_control
            if prior_control is None or (celltype, pert) not in prior_means:
                missing_prior.append((celltype, pert))
                final_delta = model_delta
            else:
                calibrated_pairs += 1
                prior_delta = prior_means[(celltype, pert)] - prior_control
                final_delta = args.model_weight * model_delta + args.prior_weight * prior_delta
            pred = model_control + final_delta
            if args.clip_min is not None:
                pred = np.maximum(pred, args.clip_min)
            rows.append(pred.astype(np.float32))
            obs_rows.append(
                {
                    "celltype": celltype,
                    "cell_type": celltype,
                    "cytokine": pert,
                    "donor": "h5ad_memory_prior",
                }
            )

    pred_out = out_dir / "pbmc_pred_memory_h5prior_m04_p06.h5ad"
    real_out = out_dir / "pbmc_real_memory_reference.h5ad"
    _write_compact_h5ad(pred_out, rows=rows, obs_rows=obs_rows, var_names=var_names)
    if real_out.exists() or real_out.is_symlink():
        real_out.unlink()
    os.symlink(model_real.resolve(), real_out)

    summary = {
        "note": args.note,
        "model_pred": str(model_pred),
        "model_real": str(model_real),
        "prior_h5ad": str(prior_h5ad),
        "pred_out": str(pred_out),
        "real_out": str(real_out),
        "control_pert": args.control_pert,
        "model_weight": args.model_weight,
        "prior_weight": args.prior_weight,
        "clip_min": args.clip_min,
        "num_calibrated_pairs": calibrated_pairs,
        "num_missing_prior_pairs": len(missing_prior),
        "missing_prior_pairs": missing_prior[:50],
    }
    if args.score:
        summary["base_score"] = score_h5ads(
            pred_h5ad=model_pred,
            real_h5ad=model_real,
            outdir=out_dir / "base_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
        summary["calibrated_score"] = score_h5ads(
            pred_h5ad=pred_out,
            real_h5ad=model_real,
            outdir=out_dir / "calibrated_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
    (out_dir / "h5prior_delta_memory_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_raw_split_prior(
    *,
    raw_h5ad: Path,
    split_toml: Path,
    source_splits: set[str],
    control_pert: str,
    shard_size: int,
    padding_seed: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int], dict[str, Any]]:
    """Rebuild PBMC train/val memory deltas from raw h5ad using split notebook rules."""
    fewshot = _parse_pbmc_split_toml(split_toml)
    start_time = time.time()
    with h5py.File(raw_h5ad, "r") as f:
        n_rows, n_vars = _x_shape(f)
        donor_codes, donors = _obs_codes(f, "donor")
        cell_codes, celltypes = _obs_codes(f, "cell_type")
        pert_codes, perts = _obs_codes(f, "cytokine")
        if not (len(donor_codes) == len(cell_codes) == len(pert_codes) == n_rows):
            raise ValueError("raw h5ad obs length mismatch")

        group_lookup: dict[tuple[str, str, str], int] = {}
        group_keys: list[tuple[str, str, str]] = []
        group_ids = np.empty(n_rows, dtype=np.int32)
        counts_list: list[int] = []
        print(f"[raw-prior] pass 1 obs grouping: {n_rows:,} rows", flush=True)
        for row_idx in range(n_rows):
            key = (
                donors[int(donor_codes[row_idx])],
                celltypes[int(cell_codes[row_idx])],
                perts[int(pert_codes[row_idx])],
            )
            group_id = group_lookup.get(key)
            if group_id is None:
                group_id = len(group_keys)
                group_lookup[key] = group_id
                group_keys.append(key)
                counts_list.append(0)
            group_ids[row_idx] = group_id
            counts_list[group_id] += 1
            if row_idx and row_idx % 2_000_000 == 0:
                print(
                    f"[raw-prior] obs grouping: {row_idx:,}/{n_rows:,}, "
                    f"groups={len(group_keys):,}, elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

        raw_counts = np.asarray(counts_list, dtype=np.int64)
        processed = np.ones(len(group_keys), dtype=bool)
        for gid, (_, _, cytokine) in enumerate(group_keys):
            if cytokine != control_pert and raw_counts[gid] <= 3:
                processed[gid] = False

        source_group_ids: set[int] = set()
        control_group_ids: set[int] = set()
        split_counts = {"train": 0, "val": 0, "test": 0, "control": 0, "dropped_lt4": 0}
        for gid, (_, celltype, cytokine) in enumerate(group_keys):
            if not processed[gid]:
                split_counts["dropped_lt4"] += 1
                continue
            if cytokine == control_pert:
                control_group_ids.add(gid)
                split_counts["control"] += 1
                continue
            split = _split_for_group(celltype, cytokine, fewshot)
            split_counts[split] = split_counts.get(split, 0) + 1
            if split in source_splits:
                source_group_ids.add(gid)

        selected_group_ids = sorted(source_group_ids | control_group_ids)
        selected_index = {gid: idx for idx, gid in enumerate(selected_group_ids)}
        sums = np.zeros((len(selected_group_ids), n_vars), dtype=np.float64)
        effective_counts = raw_counts.copy()

        rng = np.random.default_rng(padding_seed)
        pad_counts_by_group: dict[int, np.ndarray] = {}
        print(
            f"[raw-prior] processed groups={int(processed.sum()):,}, "
            f"source groups={len(source_group_ids):,}, controls={len(control_group_ids):,}, "
            f"selected={len(selected_group_ids):,}",
            flush=True,
        )
        for gid, (_, _, cytokine) in enumerate(group_keys):
            if not processed[gid] or cytokine == control_pert:
                continue
            n = int(raw_counts[gid])
            pad_n = (-n) % shard_size
            if pad_n:
                pad_idx = rng.integers(0, n, size=pad_n)
                effective_counts[gid] = n + pad_n
                if gid in source_group_ids:
                    pad_counts_by_group[gid] = np.bincount(pad_idx, minlength=n).astype(np.int16)
            else:
                effective_counts[gid] = n

        seen_offsets = np.zeros(len(group_keys), dtype=np.int64)
        x = f["X"]
        if not isinstance(x, h5py.Group) or x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError(f"{raw_h5ad} must store X as csr_matrix")
        data = x["data"]
        indices = x["indices"]
        indptr = x["indptr"][:]
        print(f"[raw-prior] pass 2 matrix sums: {n_rows:,} rows", flush=True)
        for row_idx in range(n_rows):
            gid = int(group_ids[row_idx])
            offset = int(seen_offsets[gid])
            seen_offsets[gid] += 1
            sum_idx = selected_index.get(gid)
            if sum_idx is None:
                continue
            mult = 1
            pads = pad_counts_by_group.get(gid)
            if pads is not None:
                mult += int(pads[offset])
            row_start = int(indptr[row_idx])
            row_end = int(indptr[row_idx + 1])
            if row_end > row_start:
                sums[sum_idx, indices[row_start:row_end]] += data[row_start:row_end] * mult
            if row_idx and row_idx % 1_000_000 == 0:
                print(
                    f"[raw-prior] matrix sums: {row_idx:,}/{n_rows:,}, "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

    control_means: dict[tuple[str, str], np.ndarray] = {}
    for gid in control_group_ids:
        donor, celltype, _ = group_keys[gid]
        sum_idx = selected_index[gid]
        control_means[(donor, celltype)] = (sums[sum_idx] / max(raw_counts[gid], 1)).astype(np.float32)

    sum_delta: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    missing_control = 0
    for gid in source_group_ids:
        donor, celltype, cytokine = group_keys[gid]
        control_mean = control_means.get((donor, celltype))
        if control_mean is None:
            missing_control += 1
            continue
        sum_idx = selected_index[gid]
        n_eff = int(effective_counts[gid])
        pert_mean = sums[sum_idx] / max(n_eff, 1)
        delta_weighted = (pert_mean - control_mean) * n_eff
        key = (celltype, cytokine)
        if key in sum_delta:
            sum_delta[key] += delta_weighted
            counts[key] += n_eff
        else:
            sum_delta[key] = delta_weighted.astype(np.float64, copy=True)
            counts[key] = n_eff

    prior = {key: (value / counts[key]).astype(np.float32) for key, value in sum_delta.items()}
    meta = {
        "raw_h5ad": str(raw_h5ad),
        "split_toml": str(split_toml),
        "source_splits": sorted(source_splits),
        "control_pert": control_pert,
        "shard_size": shard_size,
        "padding_seed": padding_seed,
        "n_rows": int(n_rows),
        "n_vars": int(n_vars),
        "n_groups": len(group_keys),
        "split_group_counts": split_counts,
        "num_prior_pairs": len(prior),
        "missing_control_groups": missing_control,
        "elapsed_seconds": time.time() - start_time,
    }
    return prior, counts, meta


def calibrate_from_raw_split(args: argparse.Namespace) -> dict[str, Any]:
    source_splits = {x.strip() for x in args.source_splits.split(",") if x.strip()}
    prior, counts, prior_meta = build_raw_split_prior(
        raw_h5ad=Path(args.raw_h5ad),
        split_toml=Path(args.split_toml),
        source_splits=source_splits,
        control_pert=args.control_pert,
        shard_size=args.shard_size,
        padding_seed=args.padding_seed,
    )

    model_pred = Path(args.model_pred)
    model_real = Path(args.model_real)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_prior_npz(out_dir / "pbmc_raw_split_delta_prior.npz", prior, counts)

    model_means = _group_means(
        model_pred,
        [args.celltype_col, args.pert_col],
        label="model-rawsplit-calibrate",
    )
    var_names = _var_names(model_pred)
    celltypes = sorted({key[0] for key in model_means})

    fewshot = _parse_pbmc_split_toml(Path(args.split_toml))
    target_celltype = args.target_celltype
    source_celltypes = sorted({ct for ct, _ in prior if ct != target_celltype})
    selected_source_celltypes: list[str] | None = None
    val_similarity: list[dict[str, Any]] = []
    if args.memory_strategy == "val_topk_celltypes":
        if target_celltype is None:
            raise ValueError("--target-celltype is required for val_topk_celltypes")
        val_perts = sorted(fewshot.get(target_celltype, {}).get("val", set()))
        scored: list[tuple[str, float, int]] = []
        for source_celltype in source_celltypes:
            scores = []
            for pert in val_perts:
                source_delta = prior.get((source_celltype, pert))
                target_delta = prior.get((target_celltype, pert))
                if source_delta is not None and target_delta is not None:
                    scores.append(_pearson(source_delta, target_delta))
            if scores:
                scored.append((source_celltype, float(np.nanmean(scores)), len(scores)))
        scored.sort(key=lambda row: row[1], reverse=True)
        selected_source_celltypes = [row[0] for row in scored[: args.val_top_k]]
        val_similarity = [
            {
                "celltype": celltype,
                "mean_val_pearson_to_target": score,
                "n_val": n_val,
            }
            for celltype, score, n_val in scored
        ]

    def resolve_memory_delta(celltype: str, pert: str) -> np.ndarray | None:
        if args.memory_strategy == "same_celltype":
            return prior.get((celltype, pert))

        if args.memory_strategy == "global_same_cytokine":
            allowed_celltypes = [ct for ct in source_celltypes if (ct, pert) in prior]
        elif args.memory_strategy == "val_topk_celltypes":
            allowed_celltypes = [
                ct for ct in (selected_source_celltypes or []) if (ct, pert) in prior
            ]
        else:
            raise ValueError(f"Unknown memory strategy: {args.memory_strategy}")

        if not allowed_celltypes:
            return None
        deltas = np.vstack([prior[(ct, pert)] for ct in allowed_celltypes]).astype(np.float64)
        if args.memory_celltype_weighting == "count":
            weights = np.asarray([counts[(ct, pert)] for ct in allowed_celltypes], dtype=np.float64)
            return np.average(deltas, axis=0, weights=weights).astype(np.float32)
        return np.mean(deltas, axis=0).astype(np.float32)

    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, str]] = []
    missing_prior: list[tuple[str, str]] = []
    calibrated_pairs = 0
    for celltype in celltypes:
        model_control = model_means.get((celltype, args.control_pert))
        if model_control is None:
            continue
        rows.append(model_control.astype(np.float32))
        obs_rows.append(
            {
                "celltype": celltype,
                "cell_type": celltype,
                "cytokine": args.control_pert,
                "donor": "raw_split_memory_prior",
            }
        )
        perts = sorted(pert for ct, pert in model_means if ct == celltype and pert != args.control_pert)
        for pert in perts:
            model_delta = model_means[(celltype, pert)] - model_control
            memory_delta = resolve_memory_delta(celltype, pert)
            if memory_delta is None:
                missing_prior.append((celltype, pert))
                if args.missing_prior_action == "zero":
                    final_delta = args.model_weight * model_delta
                else:
                    final_delta = model_delta
            else:
                calibrated_pairs += 1
                final_delta = args.model_weight * model_delta + args.prior_weight * memory_delta
            pred = model_control + final_delta
            if args.clip_min is not None:
                pred = np.maximum(pred, args.clip_min)
            rows.append(pred.astype(np.float32))
            obs_rows.append(
                {
                    "celltype": celltype,
                    "cell_type": celltype,
                    "cytokine": pert,
                    "donor": "raw_split_memory_prior",
                }
            )

    pred_out = out_dir / "pbmc_pred_memory_rawsplit_m04_p06.h5ad"
    real_out = out_dir / "pbmc_real_memory_reference.h5ad"
    _write_compact_h5ad(pred_out, rows=rows, obs_rows=obs_rows, var_names=var_names)
    if real_out.exists() or real_out.is_symlink():
        real_out.unlink()
    os.symlink(model_real.resolve(), real_out)

    summary = {
        "note": "PBMC Replogle-style memory delta rebuilt from raw h5ad and split TOML.",
        "model_pred": str(model_pred),
        "model_real": str(model_real),
        "pred_out": str(pred_out),
        "real_out": str(real_out),
        "prior_meta": prior_meta,
        "control_pert": args.control_pert,
        "model_weight": args.model_weight,
        "memory_weight": args.prior_weight,
        "clip_min": args.clip_min,
        "memory_strategy": args.memory_strategy,
        "memory_celltype_weighting": args.memory_celltype_weighting,
        "target_celltype": target_celltype,
        "val_top_k": args.val_top_k if args.memory_strategy == "val_topk_celltypes" else None,
        "selected_source_celltypes": selected_source_celltypes,
        "val_similarity": val_similarity[:20],
        "missing_prior_action": args.missing_prior_action,
        "num_calibrated_pairs": calibrated_pairs,
        "num_missing_prior_pairs": len(missing_prior),
        "missing_prior_pairs": missing_prior[:50],
    }
    if args.score:
        summary["base_score"] = score_h5ads(
            pred_h5ad=model_pred,
            real_h5ad=model_real,
            outdir=out_dir / "base_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
        summary["calibrated_score"] = score_h5ads(
            pred_h5ad=pred_out,
            real_h5ad=model_real,
            outdir=out_dir / "calibrated_results",
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
    (out_dir / "rawsplit_delta_memory_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def donor_proxy(args: argparse.Namespace) -> dict[str, Any]:
    pred = _group_means(Path(args.model_pred), [args.celltype_col, "donor", args.pert_col], label="pred-proxy")
    real = _group_means(Path(args.model_real), [args.celltype_col, "donor", args.pert_col], label="real-proxy")
    celltypes = sorted({key[0] for key in pred} & {key[0] for key in real})
    labels = [
        ("model", 1.0, 0.0),
        ("leave_one_donor_prior", 0.0, 1.0),
        ("blend_m05_p05", 0.5, 0.5),
        ("blend_m04_p06", 0.4, 0.6),
        ("blend_m025_p075", 0.25, 0.75),
    ]
    summary_rows = []
    for label, model_weight, prior_weight in labels:
        scores = []
        donor_scores: dict[str, list[float]] = {}
        for celltype in celltypes:
            donors = sorted({key[1] for key in real if key[0] == celltype})
            perts = sorted(
                {
                    key[2]
                    for key in real
                    if key[0] == celltype and key[2] != args.control_pert
                }
            )
            for donor in donors:
                pred_control = pred.get((celltype, donor, args.control_pert))
                real_control = real.get((celltype, donor, args.control_pert))
                if pred_control is None or real_control is None:
                    continue
                for pert in perts:
                    pred_key = (celltype, donor, pert)
                    real_key = (celltype, donor, pert)
                    if pred_key not in pred or real_key not in real:
                        continue
                    prior_deltas = []
                    for other in donors:
                        if other == donor:
                            continue
                        other_key = (celltype, other, pert)
                        other_control_key = (celltype, other, args.control_pert)
                        if other_key in real and other_control_key in real:
                            prior_deltas.append(real[other_key] - real[other_control_key])
                    if not prior_deltas:
                        continue
                    model_delta = pred[pred_key] - pred_control
                    real_delta = real[real_key] - real_control
                    prior_delta = np.mean(np.vstack(prior_deltas), axis=0)
                    final_delta = model_weight * model_delta + prior_weight * prior_delta
                    score = _pearson(final_delta, real_delta)
                    scores.append(score)
                    donor_scores.setdefault(donor, []).append(score)
        summary_rows.append(
            {
                "label": label,
                "mean_donor_cytokine_pearson_delta": float(np.nanmean(scores)) if scores else float("nan"),
                "donor_means": {
                    donor: float(np.nanmean(values)) for donor, values in sorted(donor_scores.items())
                },
            }
        )

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "note": "Proxy only: uses current heldout-file real deltas from other donors, not train-only memory.",
        "model_pred": str(args.model_pred),
        "model_real": str(args.model_real),
        "control_pert": args.control_pert,
        "summary": summary_rows,
    }
    (outdir / "leave_one_donor_memory_proxy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="State-style pseudobulk score for PBMC h5ads.")
    score_parser.add_argument("--pred-h5ad", required=True)
    score_parser.add_argument("--real-h5ad", required=True)
    score_parser.add_argument("--out-dir", required=True)
    score_parser.add_argument("--control-pert", default="PBS")
    score_parser.add_argument("--celltype-col", default="celltype")
    score_parser.add_argument("--pert-col", default="cytokine")

    cal_parser = subparsers.add_parser("calibrate", help="Build train/val memory prior and calibrated PBMC prediction.")
    cal_parser.add_argument("--model-pred", default=str(DEFAULT_MODEL_WORKSPACE / "pbmc_pred_split_4.h5ad"))
    cal_parser.add_argument("--model-real", default=str(DEFAULT_MODEL_WORKSPACE / "pbmc_real_split_4.h5ad"))
    cal_parser.add_argument("--source-lmdb", action="append", default=None)
    cal_parser.add_argument("--control-lmdb", default=str(DEFAULT_SPLIT_ROOT / "pbmc_control_split_4"))
    cal_parser.add_argument("--out-dir", required=True)
    cal_parser.add_argument("--control-pert", default="PBS")
    cal_parser.add_argument("--celltype-col", default="celltype")
    cal_parser.add_argument("--pert-col", default="cytokine")
    cal_parser.add_argument("--model-weight", type=float, default=0.4)
    cal_parser.add_argument("--prior-weight", type=float, default=0.6)
    cal_parser.add_argument("--clip-min", type=float, default=0.0)
    cal_parser.add_argument("--no-clip", action="store_true")
    cal_parser.add_argument("--missing-prior-action", choices=("model", "zero", "skip"), default="model")
    cal_parser.add_argument("--score", action="store_true")

    h5prior_parser = subparsers.add_parser(
        "calibrate-from-prior-h5ad",
        help="Build calibrated prediction from an existing validation/reference h5ad prior.",
    )
    h5prior_parser.add_argument("--model-pred", required=True)
    h5prior_parser.add_argument("--model-real", required=True)
    h5prior_parser.add_argument("--prior-h5ad", required=True)
    h5prior_parser.add_argument("--out-dir", required=True)
    h5prior_parser.add_argument("--control-pert", default="PBS")
    h5prior_parser.add_argument("--celltype-col", default="celltype")
    h5prior_parser.add_argument("--pert-col", default="cytokine")
    h5prior_parser.add_argument("--model-weight", type=float, default=0.4)
    h5prior_parser.add_argument("--prior-weight", type=float, default=0.6)
    h5prior_parser.add_argument("--clip-min", type=float, default=0.0)
    h5prior_parser.add_argument("--no-clip", action="store_true")
    h5prior_parser.add_argument(
        "--note",
        default=(
            "H5AD-prior calibration. If prior_h5ad overlaps model_real, "
            "this is a leaky diagnostic, not an official test result."
        ),
    )
    h5prior_parser.add_argument("--score", action="store_true")

    rawsplit_parser = subparsers.add_parser(
        "calibrate-from-raw-split",
        help="Rebuild PBMC train/val memory delta from raw h5ad and split TOML.",
    )
    rawsplit_parser.add_argument("--model-pred", required=True)
    rawsplit_parser.add_argument("--model-real", required=True)
    rawsplit_parser.add_argument("--raw-h5ad", default=str(DEFAULT_RAW_H5AD))
    rawsplit_parser.add_argument("--split-toml", default=str(DEFAULT_SPLIT_TOML))
    rawsplit_parser.add_argument("--source-splits", default="train,val")
    rawsplit_parser.add_argument("--out-dir", required=True)
    rawsplit_parser.add_argument("--control-pert", default="PBS")
    rawsplit_parser.add_argument("--celltype-col", default="celltype")
    rawsplit_parser.add_argument("--pert-col", default="cytokine")
    rawsplit_parser.add_argument("--model-weight", type=float, default=0.4)
    rawsplit_parser.add_argument("--prior-weight", type=float, default=0.6)
    rawsplit_parser.add_argument(
        "--memory-weight",
        dest="prior_weight",
        type=float,
        default=argparse.SUPPRESS,
    )
    rawsplit_parser.add_argument("--clip-min", type=float, default=0.0)
    rawsplit_parser.add_argument("--no-clip", action="store_true")
    rawsplit_parser.add_argument("--missing-prior-action", choices=("model", "zero"), default="model")
    rawsplit_parser.add_argument(
        "--memory-strategy",
        choices=("same_celltype", "global_same_cytokine", "val_topk_celltypes"),
        default="same_celltype",
    )
    rawsplit_parser.add_argument(
        "--memory-celltype-weighting",
        choices=("equal", "count"),
        default="equal",
    )
    rawsplit_parser.add_argument("--target-celltype", default=None)
    rawsplit_parser.add_argument("--val-top-k", type=int, default=1)
    rawsplit_parser.add_argument("--shard-size", type=int, default=1024)
    rawsplit_parser.add_argument("--padding-seed", type=int, default=0)
    rawsplit_parser.add_argument("--score", action="store_true")

    proxy_parser = subparsers.add_parser("donor-proxy", help="Diagnostic leave-one-donor memory proxy.")
    proxy_parser.add_argument("--model-pred", required=True)
    proxy_parser.add_argument("--model-real", required=True)
    proxy_parser.add_argument("--out-dir", required=True)
    proxy_parser.add_argument("--control-pert", default="PBS")
    proxy_parser.add_argument("--celltype-col", default="celltype")
    proxy_parser.add_argument("--pert-col", default="cytokine")

    args = parser.parse_args()
    if getattr(args, "no_clip", False):
        args.clip_min = None
    if args.command == "score":
        summary = score_h5ads(
            pred_h5ad=Path(args.pred_h5ad),
            real_h5ad=Path(args.real_h5ad),
            outdir=Path(args.out_dir),
            control_pert=args.control_pert,
            celltype_col=args.celltype_col,
            pert_col=args.pert_col,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "calibrate":
        if args.source_lmdb is None:
            args.source_lmdb = [
                str(DEFAULT_SPLIT_ROOT / "pbmc_train_split_4"),
                str(DEFAULT_SPLIT_ROOT / "pbmc_val_split_4"),
            ]
        calibrate(args)
    elif args.command == "calibrate-from-prior-h5ad":
        calibrate_from_prior_h5ad(args)
    elif args.command == "calibrate-from-raw-split":
        calibrate_from_raw_split(args)
    elif args.command == "donor-proxy":
        donor_proxy(args)


if __name__ == "__main__":
    main()
