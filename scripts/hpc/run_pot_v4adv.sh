#!/bin/bash
# POT level on the corrected store (DECISIONS §35, §37): extract + fits; the rule of §32 is applied after.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
taskset -c 4-11 nice -n 5 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/pot_threshold.py \
  --version v4_adv --stage all --workers 3 2>&1 | grep --line-buffered -v Warn
echo POT_V4ADV_DONE
