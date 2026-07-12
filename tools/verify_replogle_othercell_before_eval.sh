#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
cd "$PROJECT_DIR"

CELL_LINES="${REPLOGLE_CELL_LINES:-hepg2 jurkat k562}"
STEP_DIR_PREFIX="${REPLOGLE_STEP_DIR_PREFIX:-replogle_inference_}"
EXPECTED_MEAN="${REPLOGLE_EXPECTED_DELTA_PCC:-0.75}"
TOLERANCE="${REPLOGLE_DELTA_PCC_TOLERANCE:-0.08}"
SKIP_METRICS_DEFAULT='REPLOGLE_SKIP_METRICS="${REPLOGLE_SKIP_METRICS-pearson_edistance,clustering_agreement}"'

echo "=== Replogle generated h5ad pre-eval verification ==="
echo "Cell lines: ${CELL_LINES}"
echo "Step dir prefix: ${STEP_DIR_PREFIX}"
echo "Expected direct delta PCC: ${EXPECTED_MEAN} +/- ${TOLERANCE}"

if ! grep -Fq "$SKIP_METRICS_DEFAULT" run_evaluate_replogle_rjob.sh; then
  echo "run_evaluate_replogle_rjob.sh does not default to skipping pearson_edistance,clustering_agreement" >&2
  exit 1
fi

read -r -a CELLS <<< "$CELL_LINES"
for CELL_LINE in "${CELLS[@]}"; do
  STEP_DIR="${STEP_DIR_PREFIX}${CELL_LINE}"
  PRED_PATH="benchmark/workspace/${STEP_DIR}/replogle_pred_${CELL_LINE}.h5ad"
  REAL_PATH="benchmark/workspace/${STEP_DIR}/replogle_real_${CELL_LINE}.h5ad"

  echo "=== Checking ${CELL_LINE} (${STEP_DIR}) ==="
  test -f "$PRED_PATH"
  test -f "$REAL_PATH"
  stat -c 'mtime=%y size=%s path=%n' "$PRED_PATH" "$REAL_PATH"

  /usr/bin/python tools/check_replogle_h5ad_sanity.py \
    --step-dir "$STEP_DIR" \
    --cell-lines "$CELL_LINE"

  /usr/bin/python tools/check_replogle_direct_delta_pcc.py \
    --step-dir "$STEP_DIR" \
    --cell-lines "$CELL_LINE" \
    --expected-mean "$EXPECTED_MEAN" \
    --tolerance "$TOLERANCE" \
    --clip-min 0
done

echo "PASSED generated h5ad pre-eval verification"
echo "Eval command, when ready:"
echo "  REPLOGLE_CELL_LINES=\"${CELL_LINES}\" bash submit_evaluate_replogle_rjob.sh"
