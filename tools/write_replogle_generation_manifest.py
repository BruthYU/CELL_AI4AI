#!/usr/bin/env python3
"""Write provenance for a protected Replogle main_inference generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_FILES = [
    PROJECT_ROOT / "main_inference_replogle.py",
    PROJECT_ROOT / "benchmark" / "evaluate_replogle.py",
    PROJECT_ROOT / "submit_evaluate_replogle_rjob.sh",
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": sha256(path),
    }


def avg_delta_info(path: Path) -> dict[str, object]:
    info = file_info(path)
    if not path.exists():
        return info
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    source = payload.get("source", {}) if isinstance(payload, dict) else {}
    delta = payload.get("delta") if isinstance(payload, dict) else None
    pert_to_idx = payload.get("pert_to_idx") if isinstance(payload, dict) else None
    info.update(
        {
            "delta_shape": list(getattr(delta, "shape", [])),
            "num_perturbations": len(pert_to_idx) if isinstance(pert_to_idx, dict) else None,
            "source": source if isinstance(source, dict) else {},
        }
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--avg-delta", type=Path, required=True)
    parser.add_argument("--active-avg-delta", type=Path, required=True)
    parser.add_argument("--active-avg-delta-backup", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--expected-delta-pcc", type=float, default=0.75)
    parser.add_argument("--delta-pcc-tolerance", type=float, default=0.08)
    args = parser.parse_args()

    cell_line = args.cell_line.lower()
    pred_path = args.out_dir / f"replogle_pred_{cell_line}.h5ad"
    real_path = args.out_dir / f"replogle_real_{cell_line}.h5ad"

    payload = {
        "created_at": datetime.now().isoformat(),
        "status": args.status,
        "project_root": str(PROJECT_ROOT),
        "cell_line": cell_line,
        "entrypoint": str(PROJECT_ROOT / "main_inference_replogle.py"),
        "command": [
            "/usr/bin/python",
            "-u",
            "main_inference_replogle.py",
            "--config",
            str(args.config),
            "--cell-line",
            cell_line,
            "--out-dir",
            str(args.out_dir),
            "--timestamp",
            "",
        ],
        "config": file_info(args.config),
        "correction_avg_delta": avg_delta_info(args.avg_delta),
        "active_avg_delta_path": str(args.active_avg_delta),
        "active_avg_delta_at_manifest_time": avg_delta_info(args.active_avg_delta),
        "active_avg_delta_backup": None if args.active_avg_delta_backup is None else file_info(args.active_avg_delta_backup),
        "output_dir": str(args.out_dir),
        "log_file": None if args.log_file is None else str(args.log_file),
        "expected_delta_pcc": float(args.expected_delta_pcc),
        "delta_pcc_tolerance": float(args.delta_pcc_tolerance),
        "output_h5ads": {
            "pred": file_info(pred_path),
            "real": file_info(real_path),
        },
        "protected_files": {path.name: file_info(path) for path in PROTECTED_FILES},
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "DISTRIBUTED_JOB": os.environ.get("DISTRIBUTED_JOB", ""),
            "REPLOGLE_CELL_LINES": os.environ.get("REPLOGLE_CELL_LINES", ""),
            "REPLOGLE_OUT_PREFIX": os.environ.get("REPLOGLE_OUT_PREFIX", ""),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Replogle generation manifest: {args.out}")


if __name__ == "__main__":
    main()
