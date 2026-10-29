#!/bin/bash
# Phase-0 check of the iso31 rule-5 signal: radar pass, frames, rank on the calibration days.
set -uo pipefail
cd /work/fquareng/ch2/minkowski-loss
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 4-11 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA; C=$D/quality_v4/calib
echo "=== radar pass ($(date)) ==="
$PY scripts/dataset_v2/scan_radars.py config.yaml --climatology $D/quality_v2/clutter_climatology.npz \
    --out_dir $C --days_file $C/days.txt --workers 8 --skip_existing || exit 1
echo "=== frames ($(date)) ==="
$PY scripts/data_quality/radar_screen_v4.py frames --quality $C/quality_calib.yaml --keep_frames --workers 8 || exit 1
echo "=== rank ($(date)) ==="
$PY scripts/data_quality/radar_screen_v4.py rank --quality $C/quality_calib.yaml || exit 1
echo "=== done ($(date)) ==="
