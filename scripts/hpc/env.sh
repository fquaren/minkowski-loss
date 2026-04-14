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
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

# Add project root to PYTHONPATH for module imports
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Activate environment
source /home/fquareng/.bashrc
