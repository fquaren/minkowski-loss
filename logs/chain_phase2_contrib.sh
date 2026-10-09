#!/bin/bash
# 2026-10-09: Phase 2 rerun with the contributing-radar ownership geometry (DECISIONS §27):
# radar pass over the full archive, then ceilings, radar-frame flags and the radar-day ranking.
# Previous (database-geometry) outputs archived in quality_v4/phase2_db_geometry/.
set -uo pipefail
cd /work/fquareng/ch2/minkowski-loss
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 4-11 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA
echo "=== radar pass ($(date)) ==="
$PY scripts/dataset_v2/scan_radars.py config.yaml --climatology $D/quality_v2/clutter_climatology.npz \
    --out_dir $D/quality_v4 --quality configs/quality_v4.yaml --workers 8 --skip_existing || exit 1
for s in ceilings frames rank; do
  echo "=== $s ($(date)) ==="
  $PY scripts/data_quality/radar_screen_v4.py $s --quality configs/quality_v4.yaml --workers 8 || exit 1
done
echo "=== PHASE2_DONE ($(date)) ==="
