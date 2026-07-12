#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
INPUT_DIR="${1:?Usage: $0 <input-dir> <run-name>}"
RUN_NAME="${2:?Usage: $0 <input-dir> <run-name>}"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${INPUT_DIR}/logs/evaluate_rjob"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/evaluate_tahoe_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1
trap 'echo "RJOB runner exiting with status $? at $(date -Is)"' EXIT

echo "=== Starting Tahoe evaluation RJOB ==="
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Input dir: ${INPUT_DIR}"
echo "Run name: ${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Master log: ${MASTER_LOG}"
echo "Out subfolder: ${TAHOE_OUT_SUBFOLDER:-results_calibrate}"
echo "Num threads: ${TAHOE_NUM_THREADS:-32}"
echo "Skip metrics: ${TAHOE_SKIP_METRICS-pearson_edistance,clustering_agreement}"
echo "Only celltypes: ${TAHOE_ONLY_CELLTYPES:-}"
echo "Only celltype contains: ${TAHOE_ONLY_CELLTYPE_CONTAINS:-}"

extra_args=()
if [ -n "${TAHOE_ONLY_CELLTYPES:-}" ]; then
  extra_args+=(--only-celltypes "${TAHOE_ONLY_CELLTYPES}")
fi
if [ -n "${TAHOE_ONLY_CELLTYPE_CONTAINS:-}" ]; then
  extra_args+=(--only-celltype-contains "${TAHOE_ONLY_CELLTYPE_CONTAINS}")
fi

/usr/bin/python -u benchmark/evaluate_tahoe.py \
  --input-dir "$INPUT_DIR" \
  --out-subfolder "${TAHOE_OUT_SUBFOLDER:-results_calibrate}" \
  --num-threads "${TAHOE_NUM_THREADS:-32}" \
  --skip-metrics "${TAHOE_SKIP_METRICS-pearson_edistance,clustering_agreement}" \
  "${extra_args[@]}"

echo "Tahoe evaluation completed successfully"
