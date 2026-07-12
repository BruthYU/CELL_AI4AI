#!/usr/bin/env python3
"""Build full-cell Replogle h5ads for model_plus_D_same_gene_other_cellline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")
DEFAULT_MEMORY_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/"
    "replogle_trainonly_h5ad_othercell_delta_baseline_local_20260617/predictions"
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


def read_csr(handle: h5py.File) -> sp.csr_matrix:
    x = handle["X"]
    if not isinstance(x, h5py.Group) or not {"data", "indices", "indptr"} <= set(x.keys()):
        raise ValueError("Expected h5ad X to be a CSR group")
    return sp.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=h5ad_shape(handle))


def copy_non_x_groups(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        if key == "X":
            continue
        src.copy(key, dst)


def append_1d(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if values.size == 0:
        return
    old_size = int(dataset.shape[0])
    dataset.resize((old_size + int(values.size),))
    dataset[old_size:] = values


def load_memory_delta(memory_pred_path: Path, *, control_pert: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with h5py.File(memory_pred_path, "r") as handle:
        genes = obs_column(handle, "gene").astype(str)
        matrix = read_csr(handle)

    control_mask = genes == control_pert
    if not control_mask.any():
        raise ValueError(f"{memory_pred_path} has no {control_pert!r} row")
    control_mean = np.asarray(matrix[control_mask].mean(axis=0)).ravel().astype(np.float32)
    delta_by_gene: dict[str, np.ndarray] = {}
    count_by_gene: dict[str, int] = {}
    for gene in np.unique(genes[~control_mask]):
        mask = genes == gene
        mean = np.asarray(matrix[mask].mean(axis=0)).ravel().astype(np.float32)
        delta_by_gene[str(gene)] = mean - control_mean
        count_by_gene[str(gene)] = int(mask.sum())
    return delta_by_gene, count_by_gene


def contiguous_runs(values: np.ndarray) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    start = 0
    n = int(values.shape[0])
    while start < n:
        end = start + 1
        while end < n and values[end] == values[start]:
            end += 1
        runs.append((str(values[start]), start, end))
        start = end
    return runs


def create_output_x(dst: h5py.File, shape: tuple[int, int]) -> tuple[h5py.Group, h5py.Dataset, h5py.Dataset, np.ndarray]:
    out_x = dst.create_group("X")
    out_x.attrs["encoding-type"] = "csr_matrix"
    out_x.attrs["encoding-version"] = "0.1.0"
    out_x.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    data_ds = out_x.create_dataset(
        "data",
        shape=(0,),
        maxshape=(None,),
        chunks=(1_000_000,),
        dtype=np.float32,
    )
    indices_ds = out_x.create_dataset(
        "indices",
        shape=(0,),
        maxshape=(None,),
        chunks=(1_000_000,),
        dtype=np.int32,
    )
    out_indptr = np.empty(shape[0] + 1, dtype=np.int64)
    out_indptr[0] = 0
    return out_x, data_ds, indices_ds, out_indptr


def build_one_cell_line(
    *,
    raw_pred_path: Path,
    raw_real_path: Path,
    memory_pred_path: Path,
    out_pred_path: Path,
    out_real_path: Path,
    control_pert: str,
    model_weight: float,
    memory_weight: float,
    clip_min: float | None,
) -> dict[str, object]:
    delta_by_gene, memory_counts = load_memory_delta(memory_pred_path, control_pert=control_pert)

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    if out_pred_path.exists() or out_pred_path.is_symlink():
        out_pred_path.unlink()
    if out_real_path.exists() or out_real_path.is_symlink():
        out_real_path.unlink()

    with h5py.File(raw_pred_path, "r") as src, h5py.File(out_pred_path, "w") as dst:
        n_obs, n_vars = h5ad_shape(src)
        genes = obs_column(src, "gene").astype(str)
        runs = contiguous_runs(genes)
        raw_matrix = read_csr(src)
        copy_non_x_groups(src, dst)
        out_x, data_ds, indices_ds, out_indptr = create_output_x(dst, (n_obs, n_vars))

        target_rows = 0
        control_rows = 0
        missing_memory_rows = 0
        nnz_so_far = 0
        run_idx = 0
        while run_idx < len(runs):
            gene, target_start, target_end = runs[run_idx]
            if gene == control_pert:
                raise ValueError(
                    f"{raw_pred_path} has an unexpected control run at position {run_idx}; "
                    "expected target/control pairs"
                )
            if run_idx + 1 >= len(runs):
                raise ValueError(f"{raw_pred_path} target run {gene} has no paired control run")
            control_gene, control_start, control_end = runs[run_idx + 1]
            if control_gene != control_pert:
                raise ValueError(
                    f"{raw_pred_path} target run {gene} is followed by {control_gene}, "
                    f"not {control_pert}"
                )
            if (target_end - target_start) != (control_end - control_start):
                raise ValueError(f"{raw_pred_path} target/control row count mismatch for {gene}")

            pred_cells = raw_matrix[target_start:target_end].toarray().astype(np.float32, copy=False)
            ctrl_cells = raw_matrix[control_start:control_end].toarray().astype(np.float32, copy=False)
            memory_delta = delta_by_gene.get(gene)
            if memory_delta is None:
                memory_delta = np.zeros(n_vars, dtype=np.float32)
                missing_memory_rows += int(target_end - target_start)

            target_out = ctrl_cells + model_weight * (pred_cells - ctrl_cells) + memory_weight * memory_delta
            control_out = ctrl_cells
            if clip_min is not None:
                np.maximum(target_out, clip_min, out=target_out)
                np.maximum(control_out, clip_min, out=control_out)

            for start, block in ((target_start, target_out), (control_start, control_out)):
                csr = sp.csr_matrix(block)
                append_1d(data_ds, csr.data.astype(np.float32, copy=False))
                append_1d(indices_ds, csr.indices.astype(np.int32, copy=False))
                out_indptr[start + 1 : start + 1 + csr.shape[0]] = csr.indptr[1:] + nnz_so_far
                nnz_so_far += int(csr.nnz)

            target_rows += int(target_end - target_start)
            control_rows += int(control_end - control_start)
            run_idx += 2

        if run_idx != len(runs):
            raise ValueError(f"{raw_pred_path} did not consume all runs")
        indptr_dtype = np.int32 if nnz_so_far <= np.iinfo(np.int32).max else np.int64
        out_x.create_dataset(
            "indptr",
            data=out_indptr.astype(indptr_dtype, copy=False),
            chunks=(min(n_obs + 1, 65536),),
        )

    os.symlink(raw_real_path.resolve(), out_real_path)
    return {
        "raw_pred_path": str(raw_pred_path),
        "raw_real_path": str(raw_real_path.resolve()),
        "memory_pred_path": str(memory_pred_path),
        "out_pred_path": str(out_pred_path),
        "out_real_path": str(out_real_path),
        "canonical_method": "model_plus_D_same_gene_other_cellline",
        "formula": "pred = control + model_weight * delta_model + memory_weight * D_same_gene_other_cellline",
        "delta_model": "raw_model_pred_cell - paired_test_control_cell",
        "delta_memory": "D_same_gene_other_cellline from train-only same-gene other-cellline compact h5ad",
        "model_weight": float(model_weight),
        "memory_weight": float(memory_weight),
        "control_pert": control_pert,
        "clip_min": clip_min,
        "rows": int(n_obs),
        "vars": int(n_vars),
        "target_rows": int(target_rows),
        "control_rows": int(control_rows),
        "missing_memory_rows": int(missing_memory_rows),
        "num_memory_genes": int(len(delta_by_gene)),
        "num_memory_rows_by_gene_min": int(min(memory_counts.values())) if memory_counts else 0,
        "num_memory_rows_by_gene_max": int(max(memory_counts.values())) if memory_counts else 0,
        "nnz": int(nnz_so_far),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--raw-step-prefix", default="replogle_remote_ep129_raw_")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--model-weight", type=float, default=0.1)
    parser.add_argument("--memory-weight", type=float, default=0.9)
    parser.add_argument(
        "--clip-min",
        type=float,
        default=float("nan"),
        help="Optional output clipping. Default nan preserves un-clipped h5ad values.",
    )
    args = parser.parse_args()

    clip_min = None if np.isnan(args.clip_min) else float(args.clip_min)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        raw_dir = args.workspace / f"{args.raw_step_prefix}{cell_line}"
        raw_pred_path = raw_dir / f"replogle_pred_{cell_line}.h5ad"
        raw_real_path = raw_dir / f"replogle_real_{cell_line}.h5ad"
        memory_pred_path = args.memory_dir / f"replogle_pred_{cell_line}.h5ad"
        out_pred_path = args.out_dir / f"replogle_pred_{cell_line}.h5ad"
        out_real_path = args.out_dir / f"replogle_real_{cell_line}.h5ad"
        for path in (raw_pred_path, raw_real_path, memory_pred_path):
            if not path.exists():
                raise FileNotFoundError(path)

        print(
            f"build {cell_line}: control + {args.model_weight} * delta_model + "
            f"{args.memory_weight} * D_same_gene_other_cellline",
            flush=True,
        )
        summary = build_one_cell_line(
            raw_pred_path=raw_pred_path,
            raw_real_path=raw_real_path,
            memory_pred_path=memory_pred_path,
            out_pred_path=out_pred_path,
            out_real_path=out_real_path,
            control_pert=args.control_pert,
            model_weight=float(args.model_weight),
            memory_weight=float(args.memory_weight),
            clip_min=clip_min,
        )
        summaries.append(summary)
        print(
            f"wrote {out_pred_path} rows={summary['rows']} target_rows={summary['target_rows']} "
            f"control_rows={summary['control_rows']} missing_memory_rows={summary['missing_memory_rows']} "
            f"nnz={summary['nnz']}",
            flush=True,
        )

    predictions_dir = args.out_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for cell_line in [str(x).lower() for x in args.cell_lines]:
        for kind in ("pred", "real"):
            link = predictions_dir / f"replogle_{kind}_{cell_line}.h5ad"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(Path("..") / f"replogle_{kind}_{cell_line}.h5ad", link)

    manifest_path = args.out_dir / "model_plus_D_same_gene_other_cellline_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
