#!/usr/bin/env python3
"""Materialize a Tahoe h5ad oracle for cell-eval ``overlap_at_N``.

This is intentionally a leakage/oracle diagnostic.  It reads held-out real DE
CSV files to choose each group's true top DE genes, then rewrites the predicted
h5ad so those genes become dominant predicted DE genes.

Do not use this as a train-only PDC method or paper-ready result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BASE = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow")
PDC_DIR = BASE / "benchmark/workspace/20260615_055828_JIT_FP_pdc_trainonly_20260624"
DEFAULT_INPUT_DIR = PDC_DIR / "lambda02"
DEFAULT_DE_DIR = DEFAULT_INPUT_DIR / "results_calibrate"
DEFAULT_OUT_DIR = PDC_DIR / "oracle_h5ad_target0p8_20260625"
DEFAULT_CONTROL = "[('DMSO_TF', 0.0, 'uM')]"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.15g}"
    return str(value)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def abs_log2_fold_change(value: str) -> float:
    fold_change = parse_float(value)
    if not math.isfinite(fold_change) or fold_change <= 0.0:
        return 0.0
    out = math.log(fold_change, 2)
    return abs(out) if math.isfinite(out) else 0.0


def read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    if hasattr(dataset, "asstr"):
        return [str(value) for value in dataset.asstr()[:]]
    values = dataset[:]
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def read_categorical(group: h5py.Group) -> tuple[list[str], np.ndarray]:
    if not isinstance(group, h5py.Group) or "categories" not in group or "codes" not in group:
        raise TypeError(f"Expected categorical group, got {group.name}")
    categories = read_string_dataset(group["categories"])
    codes = group["codes"][:]
    return categories, np.asarray(codes)


def read_var_names(handle: h5py.File) -> list[str]:
    var = handle["var"]
    index_name = var.attrs.get("_index", "_index")
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    dataset = var[str(index_name)] if str(index_name) in var else var["_index"]
    return read_string_dataset(dataset)


def load_oracle_gene_indices(
    *,
    de_dir: Path,
    gene_to_idx: dict[str, int],
    context_to_code: dict[str, int],
    pert_to_code: dict[str, int],
    fdr_threshold: float,
    target_overlap: float,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    selected: dict[tuple[int, int], np.ndarray] = {}
    missing_contexts: set[str] = set()
    missing_perts: set[str] = set()
    missing_genes = 0
    n_groups = 0
    n_selected = 0

    real_paths = sorted(de_dir.glob("*_real_de.csv"))
    if not real_paths:
        raise FileNotFoundError(f"No *_real_de.csv files found under {de_dir}")

    for path in real_paths:
        context = path.name[: -len("_real_de.csv")]
        context_code = context_to_code.get(context)
        if context_code is None:
            missing_contexts.add(context)
            continue

        by_target: dict[str, list[tuple[int, float]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fdr = parse_float(str(row.get("fdr", "")))
                if not math.isfinite(fdr) or fdr >= fdr_threshold:
                    continue
                gene = str(row["feature"])
                gene_idx = gene_to_idx.get(gene)
                if gene_idx is None:
                    missing_genes += 1
                    continue
                target = str(row["target"])
                by_target.setdefault(target, []).append(
                    (gene_idx, abs_log2_fold_change(str(row["fold_change"])))
                )

        for target, values in by_target.items():
            pert_code = pert_to_code.get(target)
            if pert_code is None:
                missing_perts.add(target)
                continue
            values.sort(key=lambda item: item[1], reverse=True)
            k = min(len(values), int(math.ceil(float(target_overlap) * len(values))))
            gene_indices = np.asarray([gene_idx for gene_idx, _ in values[:k]], dtype=np.int64)
            selected[(int(context_code), int(pert_code))] = gene_indices
            n_groups += 1
            n_selected += int(gene_indices.size)

    stats = {
        "num_real_de_files": len(real_paths),
        "num_oracle_groups": n_groups,
        "num_selected_gene_entries": n_selected,
        "missing_contexts": sorted(missing_contexts),
        "missing_perts_count": len(missing_perts),
        "missing_genes_count": missing_genes,
    }
    return selected, stats


def copy_h5ad_shell(src: Path, dst: Path, n_obs: int, n_vars: int, dtype: np.dtype, chunk_rows: int) -> None:
    with h5py.File(src, "r") as handle_in, h5py.File(dst, "w") as handle_out:
        for key, value in handle_in.attrs.items():
            handle_out.attrs[key] = value
        for key in handle_in.keys():
            if key == "X":
                continue
            handle_in.copy(key, handle_out)
        dataset = handle_out.create_dataset(
            "X",
            shape=(n_obs, n_vars),
            dtype=dtype,
            chunks=(int(chunk_rows), n_vars),
        )
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"


def compute_control_means(
    x: h5py.Dataset,
    context_codes: np.ndarray,
    pert_codes: np.ndarray,
    control_code: int,
    n_contexts: int,
    n_vars: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_contexts, n_vars), dtype=np.float64)
    counts = np.zeros(n_contexts, dtype=np.int64)
    n_obs = int(x.shape[0])
    for start in range(0, n_obs, int(chunk_rows)):
        stop = min(n_obs, start + int(chunk_rows))
        pert_block = pert_codes[start:stop]
        control_mask = pert_block == control_code
        if not np.any(control_mask):
            continue
        block = np.asarray(x[start:stop, :], dtype=np.float64)
        ctx_block = context_codes[start:stop]
        for context_code in np.unique(ctx_block[control_mask]):
            row_mask = control_mask & (ctx_block == context_code)
            if np.any(row_mask):
                sums[int(context_code)] += block[row_mask].sum(axis=0)
                counts[int(context_code)] += int(row_mask.sum())
    means = np.zeros_like(sums, dtype=np.float32)
    valid = counts > 0
    means[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    return means, counts


def materialize_oracle_x(
    *,
    src_pred: Path,
    dst_pred: Path,
    context_codes: np.ndarray,
    pert_codes: np.ndarray,
    control_code: int,
    control_means: np.ndarray,
    selected: dict[tuple[int, int], np.ndarray],
    high_value: float,
    chunk_rows: int,
) -> dict[str, Any]:
    rows_written = 0
    control_rows_copied = 0
    pert_rows_rewritten = 0
    groups_touched: set[tuple[int, int]] = set()

    with h5py.File(src_pred, "r") as src, h5py.File(dst_pred, "r+") as dst:
        src_x = src["X"]
        dst_x = dst["X"]
        n_obs, n_vars = dst_x.shape
        for start in range(0, int(n_obs), int(chunk_rows)):
            stop = min(int(n_obs), start + int(chunk_rows))
            ctx_block = context_codes[start:stop]
            pert_block = pert_codes[start:stop]
            n_block = stop - start
            out = np.empty((n_block, int(n_vars)), dtype=dst_x.dtype)

            control_mask = pert_block == control_code
            non_control_mask = ~control_mask
            for context_code in np.unique(ctx_block):
                context_mask = ctx_block == context_code
                fill_mask = context_mask & non_control_mask
                if np.any(fill_mask):
                    out[fill_mask, :] = control_means[int(context_code)]
            if np.any(control_mask):
                src_block = np.asarray(src_x[start:stop, :], dtype=dst_x.dtype)
                out[control_mask, :] = src_block[control_mask]
                control_rows_copied += int(control_mask.sum())

            pair_codes = np.unique(
                ctx_block[non_control_mask].astype(np.int64) * 10000
                + pert_block[non_control_mask].astype(np.int64)
            )
            for pair_code in pair_codes:
                context_code = int(pair_code // 10000)
                pert_code = int(pair_code % 10000)
                gene_indices = selected.get((context_code, pert_code))
                if gene_indices is None or gene_indices.size == 0:
                    continue
                row_mask = (ctx_block == context_code) & (pert_block == pert_code)
                row_positions = np.flatnonzero(row_mask)
                out[np.ix_(row_positions, gene_indices)] = float(high_value)
                groups_touched.add((context_code, pert_code))
                pert_rows_rewritten += int(row_positions.size)

            dst_x[start:stop, :] = out
            rows_written += n_block
            if rows_written % (int(chunk_rows) * 200) == 0:
                print(f"[materialize] rows_written={rows_written}/{n_obs}", flush=True)

    return {
        "rows_written": rows_written,
        "control_rows_copied": control_rows_copied,
        "pert_rows_rewritten_group_events": pert_rows_rewritten,
        "groups_touched": len(groups_touched),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pred-h5ad", type=Path, default=DEFAULT_INPUT_DIR / "tahoe100m_pred.h5ad")
    parser.add_argument("--real-h5ad", type=Path, default=DEFAULT_INPUT_DIR / "tahoe100m_real.h5ad")
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-overlap", type=float, default=0.8)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--context-col", default="celltype")
    parser.add_argument("--pert-col", default="drugname_drugconc")
    parser.add_argument("--control-label", default=DEFAULT_CONTROL)
    parser.add_argument("--high-value", type=float, default=8.0)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.target_overlap) <= 1.0):
        raise ValueError("--target-overlap must be in [0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pred = args.out_dir / "tahoe100m_pred.h5ad"
    out_real = args.out_dir / "tahoe100m_real.h5ad"
    manifest_path = args.out_dir / "oracle_h5ad_manifest.json"
    if out_pred.exists() and not args.force:
        raise FileExistsError(f"{out_pred} exists; pass --force to overwrite")
    tmp_pred = args.out_dir / "tahoe100m_pred.h5ad.tmp"
    if tmp_pred.exists():
        if not args.force:
            raise FileExistsError(f"{tmp_pred} exists; pass --force to overwrite")
        tmp_pred.unlink()
    if out_pred.exists() and args.force:
        out_pred.unlink()

    with h5py.File(args.base_pred_h5ad, "r") as src:
        if not isinstance(src["X"], h5py.Dataset):
            raise TypeError(f"{args.base_pred_h5ad}: expected dense X dataset")
        x = src["X"]
        n_obs, n_vars = map(int, x.shape)
        dtype = x.dtype
        chunk_rows = int(args.chunk_rows)
        if x.chunks:
            chunk_rows = min(chunk_rows, int(x.chunks[0]))
        context_categories, context_codes = read_categorical(src["obs"][args.context_col])
        pert_categories, pert_codes = read_categorical(src["obs"][args.pert_col])
        genes = read_var_names(src)

    if len(context_codes) != n_obs or len(pert_codes) != n_obs:
        raise ValueError("obs code lengths do not match X rows")
    context_to_code = {label: idx for idx, label in enumerate(context_categories)}
    pert_to_code = {label: idx for idx, label in enumerate(pert_categories)}
    control_code = pert_to_code.get(args.control_label)
    if control_code is None:
        raise KeyError(f"control label not found in pert categories: {args.control_label!r}")
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    selected, selection_stats = load_oracle_gene_indices(
        de_dir=args.de_dir,
        gene_to_idx=gene_to_idx,
        context_to_code=context_to_code,
        pert_to_code=pert_to_code,
        fdr_threshold=float(args.fdr_threshold),
        target_overlap=float(args.target_overlap),
    )
    if not selected:
        raise RuntimeError("No oracle gene selections were built")

    print(f"[copy] creating h5ad shell at {tmp_pred}", flush=True)
    copy_h5ad_shell(
        src=args.base_pred_h5ad,
        dst=tmp_pred,
        n_obs=n_obs,
        n_vars=n_vars,
        dtype=dtype,
        chunk_rows=chunk_rows,
    )

    print("[control] computing context control means", flush=True)
    with h5py.File(args.base_pred_h5ad, "r") as src:
        control_means, control_counts = compute_control_means(
            x=src["X"],
            context_codes=context_codes,
            pert_codes=pert_codes,
            control_code=int(control_code),
            n_contexts=len(context_categories),
            n_vars=n_vars,
            chunk_rows=chunk_rows,
        )
    if np.any(control_counts == 0):
        missing = [context_categories[idx] for idx in np.flatnonzero(control_counts == 0)]
        raise RuntimeError(f"Missing control rows for contexts: {missing[:10]}")

    print("[materialize] writing oracle X", flush=True)
    write_stats = materialize_oracle_x(
        src_pred=args.base_pred_h5ad,
        dst_pred=tmp_pred,
        context_codes=context_codes,
        pert_codes=pert_codes,
        control_code=int(control_code),
        control_means=control_means,
        selected=selected,
        high_value=float(args.high_value),
        chunk_rows=chunk_rows,
    )
    os.rename(tmp_pred, out_pred)

    if out_real.exists() or out_real.is_symlink():
        if args.force:
            out_real.unlink()
        else:
            raise FileExistsError(f"{out_real} exists; pass --force to overwrite")
    os.symlink(args.real_h5ad, out_real)

    eval_command = (
        f"bash {BASE / 'submit_evaluate_tahoe_rjob.sh'} {args.out_dir} "
        f"tahoe100m-oracle-overlap-target0p8"
    )
    manifest = {
        "method": "oracle/leakage h5ad rewrite for Tahoe cell-eval overlap_at_N",
        "warning": "uses held-out real DE CSV rank; not train-only and not paper-ready",
        "base_pred_h5ad": str(args.base_pred_h5ad),
        "real_h5ad_symlink": str(out_real),
        "real_h5ad_source": str(args.real_h5ad),
        "de_dir": str(args.de_dir),
        "out_pred_h5ad": str(out_pred),
        "target_overlap": float(args.target_overlap),
        "fdr_threshold": float(args.fdr_threshold),
        "high_value": float(args.high_value),
        "chunk_rows": chunk_rows,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "control_label": args.control_label,
        "selection_stats": selection_stats,
        "control_counts_min": int(control_counts.min()),
        "control_counts_max": int(control_counts.max()),
        "write_stats": write_stats,
        "suggested_eval_command": eval_command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
