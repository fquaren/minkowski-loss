#!/bin/bash
# Raw vs corrected per-pixel rate frequencies (>= 1, >= 31 mm/h) on every 7th day 2016-2025, then the maps.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
PY=/work/fquareng/.micromamba/envs/dl-stable/bin/python
taskset -c 4-11 nice -n 5 $PY -u scripts/data_quality/rate_climatology.py run --step 7 --workers "${1:-4}" 2>&1 \
  | grep --line-buffered -v -E "Warn|umr_sum" || exit 1
taskset -c 4-11 $PY -u scripts/data_quality/rate_climatology.py maps 2>&1 | grep --line-buffered -v -E "Warn|umr_sum" || exit 1
echo RATE_CLIM_DONE
