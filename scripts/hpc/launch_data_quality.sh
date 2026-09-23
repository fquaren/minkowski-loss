#!/bin/bash
# Data-quality audit over the raw OPERA archive (CPU only). See scripts/data_quality/README.md.
#
#   setsid bash scripts/hpc/launch_data_quality.sh [OUT_DIR] [WORKERS] > /dev/null 2>&1 &
#
# Stages: climatology -> tile features (with clim_*) -> split audit -> summary -> galleries.
# Restartable: features use --skip_existing. Safe to run while the fetcher writes (both
# scanners skip day stores without .zmetadata). The job is pinned to cores 4-11 by env.sh;
# default 6 workers leaves the fetcher's 2 inside the 8-core budget.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"
set -euo pipefail

CONFIG="${PROJECT_ROOT}/config.yaml"
OUT="${1:-/home/fquareng/work/data/extremes/OPERA/quality}"
WORKERS="${2:-6}"
# 8-core budget on node34 (env.sh pins to cores 4-11); the fetcher takes 2 of them.
(( WORKERS <= NODE_MAX_CORES )) || { echo "WORKERS=$WORKERS exceeds the ${NODE_MAX_CORES}-core budget" >&2; exit 1; }
PY="${PYTHON}"
DQ="${PROJECT_ROOT}/scripts/data_quality"
LOG_FILE="${PROJECT_ROOT}/logs/data_quality_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")" "$OUT"
echo "Data-quality audit — log: $LOG_FILE"

{
    echo "=== started $(date) on $(hostname), out=$OUT, workers=$WORKERS ==="

    echo "--- 1. clutter climatology ---"
    "$PY" "$DQ/clutter_climatology.py" "$CONFIG" --out_dir "$OUT" --workers "$WORKERS"

    echo "--- 2. tile features ---"
    "$PY" "$DQ/compute_tile_features.py" "$CONFIG" --out_dir "$OUT" --workers "$WORKERS" \
        --climatology "$OUT/clutter_climatology.npz" --skip_existing

    echo "--- 3. split audit ---"
    "$PY" "$DQ/audit_splits.py" "$CONFIG" --out_dir "$OUT"

    echo "--- 4. summary ---"
    "$PY" "$DQ/summarize_features.py" "$CONFIG" --features_dir "$OUT/features" \
        --out_dir "$OUT/summary"

    echo "--- 5. galleries ---"
    G=("$PY" "$DQ/patch_gallery.py" "$CONFIG" --features_dir "$OUT/features"
       --out_dir "$OUT/gallery")
    "${G[@]}" --select top_max --min_max 31 --n 48
    "${G[@]}" --select random --min_max 31 --n 96
    "${G[@]}" --select unflagged --min_max 31 --n 48
    for rule in spike speckle isolated_peak flicker sea_clutter spoke static_clutter \
                qind_low declutter_isolated unphysical; do
        "${G[@]}" --select "rule:$rule" --min_max 31 --n 24 || echo "gallery $rule: none"
    done

    echo "=== finished $(date) ==="
} >> "$LOG_FILE" 2>&1
