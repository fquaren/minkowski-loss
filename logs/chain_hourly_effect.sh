#!/bin/bash
# waits for the hourly Phase-0 scan and the gauge rerun, then runs hourly_effect on the 199 days (2026-10-08)
cd /work/fquareng/ch2/minkowski-loss
while pgrep -u fquareng -f "scan_hours.py|gauge_hourly.py run" > /dev/null; do sleep 60; done
LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/hourly_effect.py run --days_file /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/days.txt --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/hourly_effect --workers 6
LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/hourly_effect.py report --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/hourly_effect --ts_days 20210714 20160529
LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/gauge_hourly.py summary --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/gauge_hourly
LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/gauge_hourly.py summary --out /home/fquareng/work/data/extremes/OPERA/quality_v4/calib/gauge_hourly --subset nohot
