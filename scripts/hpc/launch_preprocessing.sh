#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Enforce strict mode for the scientific pipeline
set -euo pipefail

CONFIG="${PROJECT_ROOT}/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/preprocessing_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting preprocessing pipeline — log: $LOG_FILE"

{
    echo "=== Preprocessing started at $(date) on $(hostname) ==="

    # echo "--- Stage 0: Download digital elevation model ---"
    # bash "${PROJECT_ROOT}/scripts/preprocess/download_dem.sh" "$CONFIG"

    # echo "--- Stage 1: Generate metadata ---"
    # micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/generate_metadata.py" "$CONFIG"

    # echo "--- Stage 2: Split and shuffle metadata ---"
    # micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/split_metadata.py" "$CONFIG"

    # echo "--- Stage 3: Zarr store creation ---"
    # micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/preprocess_data.py" "$CONFIG"

    # echo "--- Stage 4: Persistence thresholds ---"
    # micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/compute_persistence_thresholds.py" "$CONFIG"

    echo "--- Stage 5: Gamma targets ---"
    micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/compute_gamma_targets.py" "$CONFIG"

    echo "--- Stage 6: Mixup augmentation ---"
    micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/preprocess/apply_mixup.py" "$CONFIG"

    echo "=== Preprocessing complete at $(date) ==="
} > "$LOG_FILE" 2>&1

echo "Done. Check $LOG_FILE"
