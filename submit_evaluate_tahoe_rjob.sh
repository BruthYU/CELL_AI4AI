#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
RUN_SCRIPT="${PROJECT_DIR}/run_evaluate_tahoe_rjob.sh"
INPUT_DIR="${1:?Usage: $0 <input-dir> <job-name>}"
JOB_NAME="${2:?Usage: $0 <input-dir> <job-name>}"
printf -v INPUT_DIR_Q "%q" "$INPUT_DIR"
printf -v JOB_NAME_Q "%q" "$JOB_NAME"

chmod +x "$RUN_SCRIPT"

rjob submit \
  --name="$JOB_NAME" \
  --enable-sshd \
  --gpu=0 \
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
  -e TAHOE_OUT_SUBFOLDER=results_calibrate \
  -e TAHOE_NUM_THREADS=32 \
  -e TAHOE_SKIP_METRICS=pearson_edistance,clustering_agreement \
  -- bash -exc "$RUN_SCRIPT $INPUT_DIR_Q $JOB_NAME_Q"
