#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u

CONFIG="${4:-${PROJECT_ROOT}/config.yaml}"
PARAMS="${1:-${PROJECT_ROOT}/configs/unet_analytical.yaml}"
WEIGHT_GEOM="${2:-0.001}"
DATA_PCT="${3:-100.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/unet_ana_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Training UNet+analytical (w_geom=${WEIGHT_GEOM}) — log: $LOG_FILE"

micromamba run -n dl-stable env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python "${PROJECT_ROOT}/scripts/train/train_unet_analytical.py" \
    "$CONFIG" \
    --params_path "$PARAMS" \
    --weight_geom "$WEIGHT_GEOM" \
    --data_percentage "$DATA_PCT" \
    > "$LOG_FILE" 2>&1

echo "Done."
