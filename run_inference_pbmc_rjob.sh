#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
CONFIG_PATH="${1:?Usage: $0 <config-path> <job-name>}"
JOB_NAME="${2:?Usage: $0 <config-path> <job-name>}"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi
CONFIG_PATH="$(realpath "$CONFIG_PATH")"
JOB_NAME_SAFE="${JOB_NAME//[^A-Za-z0-9_.-]/_}"

LOG_DIR="./rjob_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/inference_pbmc_${JOB_NAME_SAFE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "PBMC inference RJOB runner exiting with status $? at $(date -Is)"' EXIT

echo "=== Starting PBMC inference RJOB ==="
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Job name: ${JOB_NAME}"
echo "Config path: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Log file: ${LOG_FILE}"

python -u ./main_inference_pbmc.py --config "$CONFIG_PATH"

echo "PBMC inference completed successfully"
