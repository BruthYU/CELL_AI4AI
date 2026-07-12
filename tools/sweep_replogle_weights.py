#!/usr/bin/env python3
"""Sweep Replogle model-vs-memory weights using full-cell means."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")
DEFAULT_PRIMARY_MEMORY_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "replogle_trainonly_h5ad_othercell_delta_baseline_local_20260617/predictions"
)
DEFAULT_FALLBACK_MEMORY_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "replogle_trainonly_delta_baseline_percell_best_local_20260617/predictions"
)


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if "shape" in x.attrs:
        shape = x.attrs["shape"]
        return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])


def obs_column(handle: h5py.File, name: str) -> np.ndarray:
    obj = handle["obs"][name]
    if isinstance(obj, h5py.Group) and {"codes", "categories"} <= set(obj.keys()):
        codes = obj["codes"][:]
        categories = obj["categories"].asstr()[:]
        return np.asarray([categories[int(code)] for code in codes], dtype=object)
    if hasattr(obj, "asstr"):
        return obj.asstr()[:]
    return obj[:].astype(str)


def read_rows(x: h5py.Group, start: int, end: int, n_vars: int, *, clip_min: float | None) -> np.ndarray:
    indptr = x["indptr"][start : end + 1].astype(np.int64, copy=False)
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    indptr = indptr - data_start
    matrix = sp.csr_matrix(
        (x["data"][data_start:data_end], x["indices"][data_start:data_end], indptr),
        shape=(end - start, n_vars),
    )
    arr = matrix.toarray().astype(np.float64, copy=False)
    if clip_min is not None:
        np.maximum(arr, clip_min, out=arr)
    return arr


def read_compact_csr(handle: h5py.File) -> sp.csr_matrix:
    x = handle["X"]
    if not isinstance(x, h5py.Group) or not {"data", "indices", "indptr"} <= set(x.keys()):
        raise ValueError("Expected compact h5ad X to be a CSR group")
    return sp.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=h5ad_shape(handle))


def compact_deltas(path: Path, *, control_pert: str, zero_atol: float) -> tuple[dict[str, np.ndarray], set[str]]:
    with h5py.File(path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_compact_csr(handle)

    control_mask = genes == control_pert
    if not control_mask.any():
        raise ValueError(f"{path} has no {control_pert!r} control row")
    control_mean = np.asarray(matrix[control_mask].mean(axis=0)).ravel().astype(np.float64)
    deltas: dict[str, np.ndarray] = {}
    zero_genes: set[str] = set()
    for gene in np.unique(genes[~control_mask]):
        mask = genes == gene
        delta = np.asarray(matrix[mask].mean(axis=0)).ravel().astype(np.float64) - control_mean
        gene = str(gene)
        deltas[gene] = delta
        if float(np.linalg.norm(delta)) <= float(zero_atol):
            zero_genes.add(gene)
    return deltas, zero_genes


def memory_delta_for_gene(
    gene: str,
    *,
    n_vars: int,
    primary: dict[str, np.ndarray],
    primary_zero: set[str],
    fallback: dict[str, np.ndarray],
) -> np.ndarray:
    if gene in primary and gene not in primary_zero:
        return primary[gene]
    if gene in fallback:
        return fallback[gene]
    return np.zeros(n_vars, dtype=np.float64)


def contiguous_runs(values: np.ndarray):
    start = 0
    n = int(values.shape[0])
    while start < n:
        end = start + 1
        while end < n and values[end] == values[start]:
            end += 1
        yield str(values[start]), start, end
        start = end


def full_real_means(path: Path, *, control_pert: str, clip_min: float | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    control_sum = None
    control_count = 0
    with h5py.File(path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        x = handle["X"]
        n_vars = h5ad_shape(handle)[1]
        for gene, start, end in contiguous_runs(genes):
            row_sum = read_rows(x, start, end, n_vars, clip_min=clip_min).sum(axis=0)
            n_rows = end - start
            if gene == control_pert:
                control_sum = row_sum if control_sum is None else control_sum + row_sum
                control_count += n_rows
            else:
                sums[gene] = sums.get(gene, 0) + row_sum
                counts[gene] = counts.get(gene, 0) + n_rows
    if control_sum is None or control_count == 0:
        raise ValueError(f"{path} has no control rows")
    return control_sum / control_count, {gene: sums[gene] / counts[gene] for gene in sums}


def pred_components(
    raw_pred_path: Path,
    *,
    primary_memory_path: Path,
    fallback_memory_path: Path,
    control_pert: str,
    zero_atol: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    primary, primary_zero = compact_deltas(primary_memory_path, control_pert=control_pert, zero_atol=zero_atol)
    fallback, _ = compact_deltas(fallback_memory_path, control_pert=control_pert, zero_atol=zero_atol)

    target_ctrl_sum: dict[str, np.ndarray] = {}
    raw_delta_sum: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    memory: dict[str, np.ndarray] = {}
    control_sum = None
    control_count = 0

    with h5py.File(raw_pred_path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        runs = list(contiguous_runs(genes))
        x = handle["X"]
        n_vars = h5ad_shape(handle)[1]
        idx = 0
        while idx < len(runs):
            gene, target_start, target_end = runs[idx]
            if gene == control_pert:
                raise ValueError(f"{raw_pred_path} has an unexpected control run at {idx}")
            if idx + 1 >= len(runs):
                raise ValueError(f"{raw_pred_path} target run {gene} has no paired control")
            control_gene, control_start, control_end = runs[idx + 1]
            if control_gene != control_pert:
                raise ValueError(f"{raw_pred_path} target run {gene} is followed by {control_gene}")
            if (target_end - target_start) != (control_end - control_start):
                raise ValueError(f"{raw_pred_path} target/control row count mismatch for {gene}")

            pred = read_rows(x, target_start, target_end, n_vars, clip_min=None)
            ctrl = read_rows(x, control_start, control_end, n_vars, clip_min=None)
            target_ctrl_sum[gene] = target_ctrl_sum.get(gene, 0) + ctrl.sum(axis=0)
            raw_delta_sum[gene] = raw_delta_sum.get(gene, 0) + (pred - ctrl).sum(axis=0)
            counts[gene] = counts.get(gene, 0) + (target_end - target_start)
            memory[gene] = memory_delta_for_gene(
                gene,
                n_vars=n_vars,
                primary=primary,
                primary_zero=primary_zero,
                fallback=fallback,
            )
            control_sum = ctrl.sum(axis=0) if control_sum is None else control_sum + ctrl.sum(axis=0)
            control_count += control_end - control_start
            idx += 2

    if control_sum is None or control_count == 0:
        raise ValueError(f"{raw_pred_path} has no paired controls")
    return control_sum / control_count, target_ctrl_sum, raw_delta_sum, memory, counts


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def score_cell(
    *,
    weight: float,
    pred_control: np.ndarray,
    target_ctrl_sum: dict[str, np.ndarray],
    raw_delta_sum: dict[str, np.ndarray],
    memory: dict[str, np.ndarray],
    counts: dict[str, int],
    real_control: np.ndarray,
    real_means: dict[str, np.ndarray],
    clip_min: float | None,
) -> tuple[float, float, int]:
    scores = []
    for gene in sorted(set(counts) & set(real_means)):
        pred_mean = target_ctrl_sum[gene] / counts[gene]
        pred_mean = pred_mean + weight * (raw_delta_sum[gene] / counts[gene])
        pred_mean = pred_mean + (1.0 - weight) * memory[gene]
        if clip_min is not None:
            pred_mean = np.maximum(pred_mean, clip_min)
        scores.append(pearson(pred_mean - pred_control, real_means[gene] - real_control))
    arr = np.asarray(scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()), float(np.median(finite)), int(finite.size)


def parse_weights(value: str) -> list[float]:
    return [float(x) for x in value.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--raw-step-prefix", default="replogle_remote_ep129_raw_")
    parser.add_argument("--primary-memory-dir", type=Path, default=DEFAULT_PRIMARY_MEMORY_DIR)
    parser.add_argument("--fallback-memory-dir", type=Path, default=DEFAULT_FALLBACK_MEMORY_DIR)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--weights", default="0 0.01 0.02 0.05 0.1 0.2")
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--zero-atol", type=float, default=0.0)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    clip_min = None if np.isnan(args.clip_min) else float(args.clip_min)
    weights = parse_weights(args.weights)
    cell_cache = {}
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        raw_dir = args.workspace / f"{args.raw_step_prefix}{cell_line}"
        print(f"load means: {cell_line}", flush=True)
        pred_control, target_ctrl_sum, raw_delta_sum, memory, counts = pred_components(
            raw_dir / f"replogle_pred_{cell_line}.h5ad",
            primary_memory_path=args.primary_memory_dir / f"replogle_pred_{cell_line}.h5ad",
            fallback_memory_path=args.fallback_memory_dir / f"replogle_pred_{cell_line}.h5ad",
            control_pert=args.control_pert,
            zero_atol=float(args.zero_atol),
        )
        real_control, real_means = full_real_means(
            raw_dir / f"replogle_real_{cell_line}.h5ad",
            control_pert=args.control_pert,
            clip_min=clip_min,
        )
        cell_cache[cell_line] = (
            pred_control,
            target_ctrl_sum,
            raw_delta_sum,
            memory,
            counts,
            real_control,
            real_means,
        )

    rows = []
    for weight in weights:
        means = []
        medians = []
        for cell_line, data in cell_cache.items():
            mean, median, finite = score_cell(
                weight=weight,
                pred_control=data[0],
                target_ctrl_sum=data[1],
                raw_delta_sum=data[2],
                memory=data[3],
                counts=data[4],
                real_control=data[5],
                real_means=data[6],
                clip_min=clip_min,
            )
            means.append(mean)
            medians.append(median)
            rows.append(
                {
                    "model_weight": weight,
                    "memory_weight": 1.0 - weight,
                    "cell_line": cell_line,
                    "direct_delta_pcc_mean": mean,
                    "direct_delta_pcc_median": median,
                    "finite": finite,
                }
            )
        rows.append(
            {
                "model_weight": weight,
                "memory_weight": 1.0 - weight,
                "cell_line": "macro",
                "direct_delta_pcc_mean": float(np.mean(means)),
                "direct_delta_pcc_median": float(np.mean(medians)),
                "finite": sum(int(r["finite"]) for r in rows if r["model_weight"] == weight and r["cell_line"] != "macro"),
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as handle:
        json.dump(
            {
                "formula": "pred = control + model_weight * delta_model + memory_weight * delta_memory",
                "delta_memory": "D_same_gene_other_cellline with fallback to D_same_cell_other_pert",
                "rows": rows,
            },
            handle,
            indent=2,
        )
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["cell_line"] == "macro":
            print(
                f"A={row['model_weight']:.3f} B={row['memory_weight']:.3f} "
                f"macro={row['direct_delta_pcc_mean']:.6f}",
                flush=True,
            )
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
