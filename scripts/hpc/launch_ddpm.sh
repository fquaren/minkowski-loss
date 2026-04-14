#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONFIG="${PROJECT_ROOT}/config.yaml"
PARAMS="${1:-${PROJECT_ROOT}/configs/ddpm.yaml}"
DATA_PCT="${2:-100.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/ddpm_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Training DDPM — log: $LOG_FILE"

micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/train/train_ddpm.py" \
    "$CONFIG" \
    --params_path "$PARAMS" \
    --data_percentage "$DATA_PCT" \
    > "$LOG_FILE" 2>&1

echo "Done."
