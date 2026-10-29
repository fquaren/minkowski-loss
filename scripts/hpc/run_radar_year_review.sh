#!/bin/bash
# Review pages for the three winter-heavy radar-years of the systematic scan (DECISIONS §36).
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
taskset -c 4-11 nice -n 5 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/analysis/radar_year_review.py \
  mtgud:2017 esclg:2025 ukham:2025 --workers 3 2>&1 | grep --line-buffered -v -E "Warn|umr_sum|k = np"
echo REVIEW_DONE
