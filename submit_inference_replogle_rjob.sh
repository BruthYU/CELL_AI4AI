#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
RUN_SCRIPT="${PROJECT_DIR}/run_inference_replogle_rjob.sh"
JOB_NAME="${1:?Usage: $0 <job-name> <config-path>}"
CONFIG_PATH="${2:?Usage: $0 <job-name> <config-path>}"

if [[ ! -f "$CONFIG_PATH" && -f "${PROJECT_DIR}/${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi
CONFIG_PATH="$(realpath "$CONFIG_PATH")"
printf -v CONFIG_PATH_Q "%q" "$CONFIG_PATH"
printf -v JOB_NAME_Q "%q" "$JOB_NAME"

chmod +x "$RUN_SCRIPT"

rjob submit \
  --name="$JOB_NAME" \
  --priority=9 \
  --enable-sshd \
  --gpu=1 \
  --memory=300000 \
  --cpu=16 \
  --charged-group=beam_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/yulang:/mnt/shared-storage-user/yulang \
  --mount=gpfs://gpfs1/beam:/mnt/shared-storage-user/beam \
  --mount=gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02/ \
  --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/yulang:scale_codex_0602 \
  -P 1 \
  --host-network=false \
  -e DISTRIBUTED_JOB=true \
  -- bash -exc "$RUN_SCRIPT $CONFIG_PATH_Q $JOB_NAME_Q"
