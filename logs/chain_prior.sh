#!/bin/bash
# 2026-10-09: acceptance test of the Mahalanobis mask as a persistence prior (DECISIONS §25)
cd /work/fquareng/ch2/minkowski-loss
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/prior_validate.py run --workers 8
taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/data_quality/prior_validate.py summary
echo PRIOR_DONE
