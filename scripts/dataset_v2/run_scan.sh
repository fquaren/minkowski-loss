#!/bin/bash
# v2 dataset, stage 1: clutter climatology over every complete day store, then the tile scan.
#   setsid nohup bash scripts/dataset_v2/run_scan.sh > logs/dataset_v2_scan.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/fquareng/work/data/extremes/OPERA/quality_v2
W=${WORKERS:-6}
mkdir -p "$OUT"
echo "=== climatology $(date) ==="
if [ -f "$OUT/clutter_climatology.npz" ]; then echo "exists, skipping"; else
  taskset -c 4-9 $E/bin/python -u scripts/data_quality/clutter_climatology.py config.yaml \
      --out_dir "$OUT" --workers "$W" || { echo "climatology FAILED"; exit 1; }
fi
echo "=== scan $(date) ==="
taskset -c 4-9 $E/bin/python -u scripts/dataset_v2/scan_tiles.py config.yaml \
    --climatology "$OUT/clutter_climatology.npz" --out_dir "$OUT" --workers "$W" --skip_existing
echo "=== done $(date) ==="
