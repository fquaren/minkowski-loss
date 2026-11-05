#!/bin/bash
# Shared environment for all HPC launch scripts.
# Source this from each launcher: source "${SCRIPT_DIR}/env.sh"

# Our group is assigned one GPU, by UUID, not a node: it was on node34 until 2026-11-03 and is
# on node32 since. On whichever node holds it: that GPU only, and at most 8 of 12 cores (0-3 stay
# free for other users). PCI_BUS_ID makes the index follow nvidia-smi's order (1 = 83:00.0 on
# node32); CUDA's default order can differ. Launchers may pass GPU=1 explicitly, which is right
# only where our card is index 1. src/node_limits.py enforces the same limits again inside Python.
export GPU_UUID=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4
GPU_INDEX=$(i=0; for d in /proc/driver/nvidia/gpus/*; do
    grep -qi "${GPU_UUID}" "$d/information" 2>/dev/null && { echo $i; break; }; i=$((i+1)); done)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_INDEX:-1}
export NODE_CPUS="4-11"
export NODE_MAX_CORES=8
if [[ -n "${GPU_INDEX}" ]]; then
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
