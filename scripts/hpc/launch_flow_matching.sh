#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u

CONFIG="${PROJECT_ROOT}/config.yaml"
WEIGHT_GEOM="${2:-0.0}"
DATA_PCT="${1:-100.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/flow_matching_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Training Flow Matching (w_geom=${WEIGHT_GEOM}) — log: $LOG_FILE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/train/train_flow_matching.py" \
    "$CONFIG" \
    --backbone "/home/fquareng/work/ch2/minkowski-loss/runs/sr_analytical/UNet_Ana_20260623_144958/unet_best.pth" \
    --data_percentage "$DATA_PCT" \
    --no_amp \
    > "$LOG_FILE" 2>&1
echo "Done."

    # --max_batches 20 \
