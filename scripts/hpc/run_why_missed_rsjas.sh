#!/bin/bash
# Why the hourly screen misses the rsjas 2023 half-disk (DECISIONS §36): the two worst days of 2023.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/why_missed.py \
  --days 20230310 20230722 --label rsjas_2023 \
  --mask /home/fquareng/work/data/extremes/OPERA/quality_v4/rsjas_2023_halfdisk_mask.npy \
  2>&1 | grep --line-buffered -v -E "Warn|umr_sum|k = np"
echo WHY_MISSED_DONE
