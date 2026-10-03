#!/bin/bash
# Audit flags over the whole archive (scan_flags.py): temporal support, range rings, nearest
# radar, QIND at the maximum. Flags only; nothing is removed. Waits for the job named in
# WAIT_LOG (if any) to print "=== done" so the 8-core budget is respected.
#   setsid nohup bash scripts/dataset_v2/run_flags.sh > logs/scan_flags.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=/home/fquareng/work/data/extremes/OPERA/quality_v2
W=${WORKERS:-6}
if [ -n "${WAIT_LOG:-}" ]; then
  echo "=== waiting for $WAIT_LOG ($(date)) ==="
  until grep -q "^=== done" "$WAIT_LOG" 2>/dev/null; do sleep 120; done
fi
echo "=== ring mask ($(date)) ==="
[ -f $Q/ring_mask.npz ] || taskset -c 4-9 $E/bin/python -u scripts/data_quality/ring_climatology.py
echo "=== flags ($(date)) ==="
taskset -c 4-9 $E/bin/python -u scripts/dataset_v2/scan_flags.py config.yaml \
    --climatology $Q/clutter_climatology.npz --ring_mask $Q/ring_mask.npz \
    --out_dir $Q --workers "$W" --skip_existing
echo "=== done ($(date)) ==="
