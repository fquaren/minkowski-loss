#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

set +u
source "${SCRIPT_DIR}/env.sh"
set -u

CONFIG="${PROJECT_ROOT}/config.yaml"
EMULATOR_CKPT="${1:?Usage: $0 <emulator_checkpoint_path>}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/evaluation_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Running evaluation pipeline — log: $LOG_FILE"

{
    echo "=== Evaluation started at $(date) ==="

    echo "--- Emulator evaluation ---"
    micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/evaluate/eval_emulator.py" \
        "$CONFIG" --checkpoint "$EMULATOR_CKPT"

    # echo "--- Baselines ---"
    # micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/evaluate/eval_baselines.py" \
    #     "$CONFIG"

    echo "--- Feature inversion ---"
    micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/evaluate/eval_inversion.py" \
        "$CONFIG" --checkpoint "$EMULATOR_CKPT"

    echo "=== Evaluation complete at $(date) ==="
} > "$LOG_FILE" 2>&1

echo "Done. Check $LOG_FILE"
