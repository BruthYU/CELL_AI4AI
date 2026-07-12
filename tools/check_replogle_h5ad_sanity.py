#!/usr/bin/env python3
"""Check Replogle h5ad artifacts before running cell_eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "benchmark" / "workspace"
DEFAULT_CELL_LINES = ("rpe1", "hepg2", "jurkat", "k562")


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


def summarize(path: Path, *, control_pert: str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        missing = [col for col in ("celltype", "gene") if col not in obs]
        if missing:
            raise ValueError(f"{path} missing obs columns: {missing}")

        n_obs, n_vars = h5ad_shape(handle)
        genes = obs_column(handle, "gene")
        celltypes = obs_column(handle, "celltype")
        unique_genes, counts = np.unique(genes.astype(str), return_counts=True)
        unique_celltypes = np.unique(celltypes.astype(str))

    control_mask = unique_genes == control_pert
    target_counts = counts[~control_mask]
    if target_counts.size == 0:
        raise ValueError(f"{path} has no non-control perturbations")

    return {
        "path": str(path),
        "rows": int(n_obs),
        "vars": int(n_vars),
        "celltypes": [str(x) for x in unique_celltypes.tolist()],
        "num_targets": int(target_counts.size),
        "control_count": int(counts[control_mask][0]) if control_mask.any() else 0,
        "target_min": int(target_counts.min()),
        "target_median": float(np.median(target_counts)),
        "target_max": int(target_counts.max()),
    }


def resolve_step_path(workspace: Path, step_dir: str) -> Path:
    step_path = Path(step_dir)
    if not step_path.is_absolute():
        step_path = workspace / step_path
    return step_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--step-dir", required=True, help="Workspace step dir or absolute artifact dir.")
    parser.add_argument("--cell-lines", nargs="+", default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--min-target-cells", type=int, default=2)
    args = parser.parse_args()

    step_path = resolve_step_path(args.workspace, args.step_dir)
    failures: list[str] = []
    for cell_line in args.cell_lines:
        for kind in ("pred", "real"):
            path = step_path / f"replogle_{kind}_{cell_line}.h5ad"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            summary = summarize(path, control_pert=args.control_pert)
            print(
                f"{kind:4s} {cell_line:6s} rows={summary['rows']} vars={summary['vars']} "
                f"targets={summary['num_targets']} control={summary['control_count']} "
                f"target_min={summary['target_min']} median={summary['target_median']:.1f} "
                f"max={summary['target_max']} celltypes={','.join(summary['celltypes'])}"
            )
            if int(summary["target_min"]) < args.min_target_cells:
                failures.append(
                    f"{path} has target_min={summary['target_min']} < {args.min_target_cells}"
                )
            if int(summary["control_count"]) <= 0:
                failures.append(f"{path} has no {args.control_pert!r} control rows")
            if int(summary["vars"]) != 2000:
                failures.append(f"{path} has vars={summary['vars']} instead of 2000")

    if failures:
        print("FAILED sanity check:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASSED sanity check")


if __name__ == "__main__":
    main()
