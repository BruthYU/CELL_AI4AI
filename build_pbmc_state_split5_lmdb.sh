#!/bin/bash
set -euo pipefail

cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SPLIT_TOML="${PBMC_SPLIT_TOML:-preprocessing/arcinstitute/datasets/State_Parse_Filtered/few_shot/split_5_state.toml}"
RAW_H5AD="${PBMC_RAW_H5AD:-preprocessing/arcinstitute/datasets/State_Parse_Filtered/only_hvg/PBMC_only_hvg.h5ad}"
SPLIT_ROOT="${PBMC_SPLIT_ROOT:-preprocessing/arcinstitute/datasets/State_Parse_Filtered/few_shot/split_5_state}"
SPLIT_NAME="${PBMC_SPLIT_NAME:-split_5_state}"
FORCE_REBUILD_LMDB="${PBMC_FORCE_REBUILD_LMDB:-false}"
LMDB_READY_FILE="${SPLIT_ROOT}/._SUCCESS"
LOCK_DIR="${SPLIT_ROOT}.build.lock"

if [ "${FORCE_REBUILD_LMDB}" != "true" ] && [ -f "${LMDB_READY_FILE}" ]; then
  echo "LMDB already exists: ${SPLIT_ROOT}"
  exit 0
fi

while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  if [ "${FORCE_REBUILD_LMDB}" != "true" ] && [ -f "${LMDB_READY_FILE}" ]; then
    echo "LMDB was built by another job: ${SPLIT_ROOT}"
    exit 0
  fi
  echo "Waiting for LMDB build lock: ${LOCK_DIR}"
  sleep 60
done
trap 'rm -rf "${LOCK_DIR}"' EXIT

if [ "${FORCE_REBUILD_LMDB}" = "true" ]; then
  rm -rf "$SPLIT_ROOT"
fi

if [ "${FORCE_REBUILD_LMDB}" = "true" ] || [ ! -f "${LMDB_READY_FILE}" ]; then
  python tools/build_pbmc_state_split_lmdb.py \
    --raw-h5ad "$RAW_H5AD" \
    --split-toml "$SPLIT_TOML" \
    --out-root "$SPLIT_ROOT" \
    --split-name "$SPLIT_NAME" \
    --control-pert PBS \
    --map-size-gb 180
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LMDB_READY_FILE"
fi

echo "LMDB ready: ${SPLIT_ROOT}"
