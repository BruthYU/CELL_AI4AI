#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
RUN_SCRIPT="${PROJECT_DIR}/run_train_rjob.sh"
JOB_NAME="${1:?Usage: $0 <job-name> <config-path>}"
CONFIG_PATH="${2:?Usage: $0 <job-name> <config-path>}"
printf -v CONFIG_PATH_Q "%q" "$CONFIG_PATH"

chmod +x "$RUN_SCRIPT"

# rjob submit --name="${JOB_NAME}" --priority=9 --enable-sshd --gpu=8 --memory=200000 --cpu=32 --charged-group=beam_gpu --private-machine=group --mount=gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02/ --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/yulang:scale_codex_0602 -P 1 --host-network=false -e DISTRIBUTED_JOB=true -- bash -exc "$RUN_SCRIPT $CONFIG_PATH_Q"


rjob submit --name="${JOB_NAME}" --priority=9 --enable-sshd --gpu=6 --memory=200000 --cpu=32 --charged-group=ma4agismall_gpu --private-machine=group --mount=gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02/ --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/yulang:scale_codex_0602 -P 1 --host-network=false -e DISTRIBUTED_JOB=true -- bash -exc "$RUN_SCRIPT $CONFIG_PATH_Q"