#!/bin/bash
# B1 (DECISIONS §43): the paired ODYSSEY/NIMBUS store for rung R2, after the ODYSSEY scan of the 101 paired days.
# Waits for the scan and for the single-core analyses (gauge_levels, analogue_distance) so that the store build
# (2 workers) never pushes the total past the 8-core budget next to a training run (5).
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=
P="taskset -c 4-11 $E/bin/python -u"
S=/home/fquareng/work/data/extremes/OPERA/quality_v4/hourly_odyssey_paired
echo "=== paired build, code $(git rev-parse --short HEAD), $(date)"
until [ "$(ls $S/hours 2>/dev/null | wc -l)" -ge 101 ] && ! pgrep -f scan_hours.py > /dev/null; do sleep 300; done
while pgrep -f 'gauge_levels.py|analogue_distance.py' > /dev/null; do sleep 300; done
for st in meta store target gamma; do
  echo "=== $st $(date)"
  $P scripts/dataset_v2/build_paired_odyssey.py --stage $st 2>&1 | grep -v -i "warn\|pystepsrc\|it/s\]" | tail -15 \
    || { echo "PAIRED $st FAILED"; exit 1; }
done
echo "=== PAIRED_BUILD_DONE $(date)"
