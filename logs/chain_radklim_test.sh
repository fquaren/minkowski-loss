#!/bin/bash
# 2026-10-08: after the RADKLIM fetch and the main extremes_effect run, run the RADKLIM removal test
# on the calibration days, then write every report.
cd /work/fquareng/ch2/minkowski-loss
while pgrep -u fquareng -f "fetch_radklim.py" > /dev/null || pgrep -u fquareng -f "extremes_effect.py run --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/extremes_effect --workers" > /dev/null; do sleep 60; done
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/radklim_test.py run --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/radklim_test --workers 5
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/radklim_test.py summary --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/radklim_test
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/extremes_effect.py report --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/extremes_effect
echo CHAIN_DONE
