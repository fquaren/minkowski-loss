#!/bin/bash
# Background fetch of the archive days not on disk: 2013 (ODYSSEY) first, then
# 2025-10-23 -> 2026-09-29 (NIMBUS). Cores 10-11, 2 workers; resumable (--skip_existing).
#   setsid nohup bash scripts/data/fetch_remaining.sh > logs/fetch_remaining.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLOSC_NTHREADS=1
R=/home/fquareng/work/data/extremes/OPERA/raw/OPERA
for range in "2013-01-01:2013-12-31" "2025-10-23:2026-09-29"; do
  echo "=== ${range} ($(date)) ==="
  taskset -c 10,11 $E/bin/python -u scripts/data/fetch_opera_archive.py \
      --start ${range%%:*} --end ${range##*:} --out $R --reference $R/20230801 \
      --skip_existing --workers 2
done
echo "=== done ($(date)) ==="
