#!/bin/bash
set -euo pipefail

cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INPUT_DIR="${1:?Usage: $0 <input-dir> <job-name>}"
JOB_NAME="${2:?Usage: $0 <input-dir> <job-name>}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

INPUT_DIR="$(realpath "$INPUT_DIR")"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

REPLOGLE_OUT_SUBFOLDER="${REPLOGLE_OUT_SUBFOLDER:-results_calibrate}"
REPLOGLE_NUM_THREADS="${REPLOGLE_NUM_THREADS:-32}"
REPLOGLE_BATCH_SIZE="${REPLOGLE_BATCH_SIZE:-100}"
REPLOGLE_SKIP_METRICS="${REPLOGLE_SKIP_METRICS-pearson_edistance,clustering_agreement}"
REPLOGLE_DE_SOURCE_SUBFOLDER="${REPLOGLE_DE_SOURCE_SUBFOLDER:-}"

LOG_DIR="${INPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/evaluate_replogle_rjob_${JOB_NAME}_${TIMESTAMP}.log"

echo "=== Starting Replogle evaluation RJOB ==="
echo "Input dir: ${INPUT_DIR}"
echo "Job name: ${JOB_NAME}"
echo "Cell line: auto"
echo "Log file: ${LOG_FILE}"
echo "Out subfolder: ${REPLOGLE_OUT_SUBFOLDER}"
echo "Batch size: ${REPLOGLE_BATCH_SIZE}"
echo "Skip metrics: ${REPLOGLE_SKIP_METRICS}"
echo "DE source subfolder: ${REPLOGLE_DE_SOURCE_SUBFOLDER:-<none>}"

/usr/bin/python -u benchmark/evaluate_replogle.py \
  --step-dir "$INPUT_DIR" \
  --job-name "$JOB_NAME" \
  --out-subfolder "$REPLOGLE_OUT_SUBFOLDER" \
  --num-threads "$REPLOGLE_NUM_THREADS" \
  --batch-size "$REPLOGLE_BATCH_SIZE" \
  --skip-metrics "$REPLOGLE_SKIP_METRICS" \
  --de-source-subfolder "$REPLOGLE_DE_SOURCE_SUBFOLDER" \
  2>&1 | tee "$LOG_FILE"

echo "Replogle evaluation completed successfully"
