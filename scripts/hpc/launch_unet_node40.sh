#!/bin/bash
# U-Net training on node40 through node40_run.sh (GPU 0-3 only, busy-GPU refusal, CPUs 0-5, nice 10).
#   launch_unet_node40.sh <gpu 0-3> <config> <weight_geom> <seed> <label> [data_percentage]   -> prints the log path
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 5 ] || { sed -n 2,3p "$0"; exit 1; }
GPU=$1; CONFIG=$2; W=$3; SEED=$4; LABEL=$5; PCT=${6:-100.0}
exec "$PROJECT_ROOT/scripts/hpc/node40_run.sh" "$GPU" "$LABEL" \
  "$PROJECT_ROOT/scripts/train/train_unet_analytical.py" "$CONFIG" \
  --params_path "$PROJECT_ROOT/configs/unet_analytical.yaml" --weight_geom "$W" \
  --data_percentage "$PCT" --seed "$SEED"
