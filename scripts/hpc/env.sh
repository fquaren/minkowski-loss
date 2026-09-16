#!/bin/bash
# Shared environment for all HPC launch scripts.
# Source this from each launcher: source "${SCRIPT_DIR}/env.sh"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Prevent thread thrashing in multiprocessing workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Conda/micromamba environment
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:${LD_LIBRARY_PATH:-}

# Pin the interpreter. Every launcher falls back to `command -v python3`, which resolves to
# /usr/bin/python3 -- no torch -- whenever the launching shell is not interactive, because the
# `source ~/.bashrc` below returns early in that case. Setting PYTHON here, ahead of that
# source, makes the launchers independent of how they were started (setsid, nohup, cron,
# Slurm). An explicit PYTHON= on the command line still wins.
export PYTHON="${PYTHON:-/work/fquareng/.micromamba/envs/dl-stable/bin/python}"

# Add project root to PYTHONPATH for module imports
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Activate environment
source /home/fquareng/.bashrc
