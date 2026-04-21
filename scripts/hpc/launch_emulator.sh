#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

set +u
source "${SCRIPT_DIR}/env.sh"
set -u

CONFIG="${PROJECT_ROOT}/config.yaml"
ARCH="${1:-Constrained}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/emulator_${ARCH}_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Training emulator (${ARCH}) — log: $LOG_FILE"

micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/train/train_emulator.py" \
    "$CONFIG" \
    --arch "$ARCH" \
    --data_fraction 1.0 \
    > "$LOG_FILE" 2>&1

echo "Done."
