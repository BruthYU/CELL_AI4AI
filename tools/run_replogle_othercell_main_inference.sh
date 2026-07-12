#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
cd "$PROJECT_DIR"

CELL_LINES="${REPLOGLE_CELL_LINES:-rpe1 hepg2 jurkat k562}"
OUT_PREFIX="${REPLOGLE_OUT_PREFIX:-replogle_inference_}"
AVG_DIR="${REPLOGLE_OTHERCELL_AVG_DIR:-${PROJECT_DIR}/dsets/replogle/main_correction_avg_delta}"
CONFIG_PATH="${REPLOGLE_CONFIG_PATH:-${PROJECT_DIR}/config/jit_llm_replogle_v3_statealign_resid_vpred_set512_dit36_infer128.yaml}"
AVG_PATH="${PROJECT_DIR}/dsets/replogle/replogle_train_avg_delta.pkl"
EXPECTED_DELTA_PCC="${REPLOGLE_EXPECTED_DELTA_PCC:-0.75}"
DELTA_PCC_TOLERANCE="${REPLOGLE_DELTA_PCC_TOLERANCE:-0.08}"
DRY_RUN="${REPLOGLE_DRY_RUN:-0}"

BACKUP_DIR="${PROJECT_DIR}/dsets/replogle/avg_delta_backups"
BACKUP_PATH=""

restore_avg_delta() {
  if [[ -n "$BACKUP_PATH" ]]; then
    cp -p "$BACKUP_PATH" "$AVG_PATH"
  fi
}
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$BACKUP_DIR"
  BACKUP_PATH="${BACKUP_DIR}/replogle_train_avg_delta.before_othercell_$(date +%Y%m%d_%H%M%S).pkl"
  cp -p "$AVG_PATH" "$BACKUP_PATH"
  trap restore_avg_delta EXIT
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: no files will be copied and main_inference_replogle.py will not run"
else
  echo "Backed up active avg_delta to: $BACKUP_PATH"
fi
echo "Cell lines: $CELL_LINES"
echo "Output prefix: $OUT_PREFIX"
echo "Correction avg_delta dir: $AVG_DIR"
echo "Config path: $CONFIG_PATH"
echo "Expected direct delta PCC: ${EXPECTED_DELTA_PCC} +/- ${DELTA_PCC_TOLERANCE}"

read -r -a CELL_LINE_ARRAY <<< "$CELL_LINES"
for CELL_LINE in "${CELL_LINE_ARRAY[@]}"; do
  OTHERCELL_AVG="${AVG_DIR}/replogle_train_avg_delta_main_correction_for_${CELL_LINE}.pkl"
  OUT_DIR="${PROJECT_DIR}/benchmark/workspace/${OUT_PREFIX}${CELL_LINE}"
  LOG_DIR="${OUT_DIR}/logs"
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/main_inference_replogle_${CELL_LINE}_${RUN_TIMESTAMP}.log"
  PRE_MANIFEST="${LOG_DIR}/main_inference_replogle_${CELL_LINE}_${RUN_TIMESTAMP}_pre_manifest.json"
  POST_MANIFEST="${LOG_DIR}/main_inference_replogle_${CELL_LINE}_${RUN_TIMESTAMP}_post_manifest.json"
  ARTIFACT_BACKUP_DIR="${PROJECT_DIR}/benchmark/workspace/replogle_inference_backups/${OUT_PREFIX}${CELL_LINE}_${RUN_TIMESTAMP}"

  test -f "$CONFIG_PATH"
  test -f "$OTHERCELL_AVG"
  /usr/bin/python tools/check_replogle_avg_delta_payload.py \
    --path "$OTHERCELL_AVG" \
    --expected-cell-line "$CELL_LINE" \
    --require-source
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$LOG_DIR"
  fi
  for EXISTING in \
    "${OUT_DIR}/replogle_real_${CELL_LINE}.h5ad" \
    "${OUT_DIR}/replogle_pred_${CELL_LINE}.h5ad"
  do
    if [[ -f "$EXISTING" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY RUN: would back up existing artifact: $EXISTING -> $ARTIFACT_BACKUP_DIR/"
      else
        mkdir -p "$ARTIFACT_BACKUP_DIR"
        cp -p "$EXISTING" "$ARTIFACT_BACKUP_DIR/"
        echo "Backed up existing artifact: $EXISTING -> $ARTIFACT_BACKUP_DIR/"
      fi
    fi
  done

  echo "=== Generating ${CELL_LINE} ==="
  echo "Config: $CONFIG_PATH"
  echo "Using avg_delta: $OTHERCELL_AVG"
  echo "Output dir: $OUT_DIR"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN: would copy $OTHERCELL_AVG -> $AVG_PATH"
    echo "DRY RUN: would run /usr/bin/python -u main_inference_replogle.py --config $CONFIG_PATH --cell-line $CELL_LINE --out-dir $OUT_DIR --timestamp \"\""
    echo "DRY RUN: would check avg-delta missing=0, h5ad sanity, and direct delta PCC"
    continue
  fi

  cp -p "$OTHERCELL_AVG" "$AVG_PATH"

  /usr/bin/python tools/write_replogle_generation_manifest.py \
    --out "$PRE_MANIFEST" \
    --status pre_inference \
    --cell-line "$CELL_LINE" \
    --config "$CONFIG_PATH" \
    --avg-delta "$OTHERCELL_AVG" \
    --active-avg-delta "$AVG_PATH" \
    --active-avg-delta-backup "$BACKUP_PATH" \
    --out-dir "$OUT_DIR" \
    --log-file "$LOG_FILE" \
    --expected-delta-pcc "$EXPECTED_DELTA_PCC" \
    --delta-pcc-tolerance "$DELTA_PCC_TOLERANCE"

  /usr/bin/python -u main_inference_replogle.py \
    --config "$CONFIG_PATH" \
    --cell-line "$CELL_LINE" \
    --out-dir "$OUT_DIR" \
    --timestamp "" \
    2>&1 | tee "$LOG_FILE"

  if ! grep -q "missing=0" "$LOG_FILE"; then
    echo "Avg-delta coverage check failed for ${CELL_LINE}; see ${LOG_FILE}" >&2
    exit 1
  fi

  /usr/bin/python tools/check_replogle_h5ad_sanity.py \
    --step-dir "$OUT_DIR" \
    --cell-lines "$CELL_LINE"

  /usr/bin/python tools/check_replogle_direct_delta_pcc.py \
    --step-dir "$OUT_DIR" \
    --cell-lines "$CELL_LINE" \
    --expected-mean "$EXPECTED_DELTA_PCC" \
    --tolerance "$DELTA_PCC_TOLERANCE"

  /usr/bin/python tools/write_replogle_generation_manifest.py \
    --out "$POST_MANIFEST" \
    --status post_checks_passed \
    --cell-line "$CELL_LINE" \
    --config "$CONFIG_PATH" \
    --avg-delta "$OTHERCELL_AVG" \
    --active-avg-delta "$AVG_PATH" \
    --active-avg-delta-backup "$BACKUP_PATH" \
    --out-dir "$OUT_DIR" \
    --log-file "$LOG_FILE" \
    --expected-delta-pcc "$EXPECTED_DELTA_PCC" \
    --delta-pcc-tolerance "$DELTA_PCC_TOLERANCE"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run completed."
else
  echo "Generation and pre-eval checks completed."
fi
