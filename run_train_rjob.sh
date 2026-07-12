#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CONFIG_PATH="${1:?Usage: $0 <config-path>}"

LOG_DIR="./rjob_logs"
mkdir -p "${LOG_DIR}"

echo "=== NeMo Training Configuration ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "CONFIG_PATH: ${CONFIG_PATH}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "=== Starting Training ==="

# 直接运行 python，NeMo 会自动处理分布式
if ! python ./main_train.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"; then
    echo "Training failed"
    exit 1
fi

echo "Training completed successfully"
