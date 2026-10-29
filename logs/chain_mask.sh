#!/bin/bash
# 2026-10-09: after the advection tests, build the per-pixel outlier masks and run their
# RADKLIM acceptance test (DECISIONS §25).
cd /work/fquareng/ch2/minkowski-loss
while pgrep -u fquareng -f "advection_test.py run|advection_gauges.py run" > /dev/null; do sleep 60; done
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/pixel_outliers.py --workers 4 && taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/mask_validate.py run --workers 8 && taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/mask_validate.py summary && echo MASK_DONE
