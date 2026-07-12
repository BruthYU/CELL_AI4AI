#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
INPUT_DIR="${1:?Usage: $0 <input-dir> <job-name>}"
JOB_NAME="${2:?Usage: $0 <input-dir> <job-name>}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

INPUT_DIR="$(realpath "$INPUT_DIR")"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PBMC_OUT_SUBFOLDER="${PBMC_OUT_SUBFOLDER:-results_calibrate}"
PBMC_CELLTYPE_COL="${PBMC_CELLTYPE_COL:-celltype}"
PBMC_NUM_THREADS="${PBMC_NUM_THREADS:-32}"
PBMC_SKIP_METRICS="${PBMC_SKIP_METRICS-pearson_edistance,clustering_agreement}"

LOG_DIR="${INPUT_DIR}/logs/evaluate_rjob"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/evaluate_pbmc_${JOB_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
trap 'echo "RJOB runner exiting with status $? at $(date -Is)"' EXIT

echo "=== Starting PBMC evaluation RJOB ==="
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Input dir: ${INPUT_DIR}"
echo "Job name: ${JOB_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Master log: ${MASTER_LOG}"
echo "Out subfolder: ${PBMC_OUT_SUBFOLDER}"
echo "Celltype col: ${PBMC_CELLTYPE_COL}"
echo "Num threads: ${PBMC_NUM_THREADS}"
echo "Skip metrics: ${PBMC_SKIP_METRICS}"

/usr/bin/python -u benchmark/evaluate_pbmc.py \
  --input-dir "$INPUT_DIR" \
  --out-subfolder "$PBMC_OUT_SUBFOLDER" \
  --celltype-col "$PBMC_CELLTYPE_COL" \
  --num-threads "$PBMC_NUM_THREADS" \
  --skip-metrics "$PBMC_SKIP_METRICS"

echo "PBMC evaluation completed successfully"
