#!/bin/bash
set -euo pipefail

PROJECT_DIR="/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow"
cd "$PROJECT_DIR"

CELL_LINES="${REPLOGLE_CELL_LINES:-hepg2 jurkat k562}"
AVG_DIR="${REPLOGLE_OTHERCELL_AVG_DIR:-${PROJECT_DIR}/dsets/replogle/main_correction_avg_delta}"
CONFIG_PATH="${REPLOGLE_CONFIG_PATH:-${PROJECT_DIR}/config/jit_llm_replogle_v3_statealign_resid_vpred_set512_dit36_infer128.yaml}"
EXPECTED_DELTA_PCC="${REPLOGLE_EXPECTED_DELTA_PCC:-0.75}"
DELTA_PCC_TOLERANCE="${REPLOGLE_DELTA_PCC_TOLERANCE:-0.08}"

echo "=== Replogle other-cell generation preflight ==="
echo "Cell lines: ${CELL_LINES}"
echo "Correction avg_delta dir: ${AVG_DIR}"
echo "Config path: ${CONFIG_PATH}"
echo "Expected direct delta PCC: ${EXPECTED_DELTA_PCC} +/- ${DELTA_PCC_TOLERANCE}"

test -f "$CONFIG_PATH"

read -r -a CELL_LINE_ARRAY <<< "$CELL_LINES"
for CELL_LINE in "${CELL_LINE_ARRAY[@]}"; do
  /usr/bin/python tools/check_replogle_avg_delta_payload.py \
    --path "${AVG_DIR}/replogle_train_avg_delta_main_correction_for_${CELL_LINE}.pkl" \
    --expected-cell-line "$CELL_LINE" \
    --require-source
done

/usr/bin/python tools/check_replogle_correction_delta_pcc.py \
  --cell-lines "${CELL_LINE_ARRAY[@]}" \
  --expected-mean "$EXPECTED_DELTA_PCC" \
  --tolerance "$DELTA_PCC_TOLERANCE"

REPLOGLE_DRY_RUN=1 \
  REPLOGLE_CELL_LINES="$CELL_LINES" \
  REPLOGLE_OTHERCELL_AVG_DIR="$AVG_DIR" \
  REPLOGLE_CONFIG_PATH="$CONFIG_PATH" \
  REPLOGLE_EXPECTED_DELTA_PCC="$EXPECTED_DELTA_PCC" \
  REPLOGLE_DELTA_PCC_TOLERANCE="$DELTA_PCC_TOLERANCE" \
  bash tools/run_replogle_othercell_main_inference.sh

if ! grep -q 'REPLOGLE_SKIP_METRICS="${REPLOGLE_SKIP_METRICS-pearson_edistance,clustering_agreement}"' run_evaluate_replogle_rjob.sh; then
  echo "run_evaluate_replogle_rjob.sh does not default to skipping pearson_edistance,clustering_agreement" >&2
  exit 1
fi

echo "PASSED Replogle generation preflight"
