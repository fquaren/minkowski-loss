#!/bin/bash
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=/work/fquareng/ch2/minkowski-loss
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u logs/r4_spectra_logspace.py 2>&1 | grep -v Warn
