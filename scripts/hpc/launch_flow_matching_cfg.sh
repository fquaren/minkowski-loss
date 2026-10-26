#!/bin/bash
# Flow-matching training with an explicit config and backbone (DECISIONS §38). launch_flow_matching.sh is
# unchanged; it hardcodes config.yaml and the 15-min backbone.
#   launch_flow_matching_cfg.sh <config> <backbone checkpoint> [<data_pct>]
# fp32 (--no_amp: fp16 overflows once the residual is correctly scaled); GPU 1; run from the project root so
# runs/ lands there. Check the startup line reads "config FM_RESIDUAL_SCALE=...".
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u
CONFIG="$1"
BACKBONE="$2"
DATA_PCT="${3:-100.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/flow_matching_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"
cd "$PROJECT_ROOT"
echo "Training Flow Matching (config=${CONFIG}, backbone=${BACKBONE}, data=${DATA_PCT}%) — log: $LOG_FILE"
micromamba run -n dl-stable env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python \
    "${PROJECT_ROOT}/scripts/train/train_flow_matching.py" "$CONFIG" \
    --backbone "$BACKBONE" --data_percentage "$DATA_PCT" --no_amp \
    > "$LOG_FILE" 2>&1
echo "Done."
