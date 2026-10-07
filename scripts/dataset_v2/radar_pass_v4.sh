#!/bin/bash
# Phase 2 of the v4 build (EXPERIMENTS §5, DECISIONS §19): the radar pass over every day,
# then the ceiling table, the radar-frame flags, the radar-day ranking and the review gallery.
# Ends with quality_v4/review.csv + review.md for the researcher; the tile scan (Phase 3)
# starts only after the reviewed exclusions are in configs/quality_v4.yaml.
# ~9 h on 8 workers; uses cores 4-11, so nothing else should run alongside.
#   setsid nohup bash scripts/dataset_v2/radar_pass_v4.sh > logs/radar_pass_v4.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
PY="taskset -c 4-11 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA
Q=$D/quality_v4
CLIM=$D/quality_v2/clutter_climatology.npz
mkdir -p $Q

echo "=== radar pass ($(date)) ==="
$PY scripts/dataset_v2/scan_radars.py config.yaml --climatology $CLIM --out_dir $Q \
    --workers $W --skip_existing || { echo "radar pass FAILED"; exit 1; }
n_days=$(ls -d $D/raw/OPERA/[0-9]*/.zmetadata | wc -l); n_npz=$(ls $Q/radars/*.npz | wc -l)
echo "radar pass: $n_npz day files for $n_days day stores"
[ "$n_npz" -ge "$n_days" ] || { echo "not every day scanned, stopping"; exit 1; }

for step in ceilings frames rank gallery; do
    echo "=== $step ($(date)) ==="
    $PY scripts/data_quality/radar_screen_v4.py $step --quality configs/quality_v4.yaml \
        --workers $W || { echo "$step FAILED"; exit 1; }
done
echo "=== done: review $Q/review.md and fill $Q/review.csv ($(date)) ==="
