#!/bin/bash
# Shared environment for all HPC launch scripts.
# Source this from each launcher: source "${SCRIPT_DIR}/env.sh"

# Shared node (node34): GPU 1 ONLY, never GPU 0, and at most 8 of 12 cores (0-3 stay free
# for other users). PCI_BUS_ID makes index 1 mean the card at 83:00.0, as in nvidia-smi;
# CUDA's default order can differ. Launchers may pass GPU=1 explicitly; nothing may use 0.
# src/node_limits.py enforces the same limits again inside Python.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export NODE_CPUS="4-11"
export NODE_MAX_CORES=8
if [[ "$(hostname)" == node34* ]]; then
    # Pin the launching shell; every child (micromamba run, python, DataLoader workers)
    # inherits the affinity, so a job can never spread beyond these 8 cores.
    taskset -cp "${NODE_CPUS}" $$ > /dev/null
fi
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
