#!/bin/bash
# ============================================================================
# Shared environment for all HPC launch scripts (SLURM-managed cluster).
# Source this from each launcher: source "${SCRIPT_DIR}/env.sh"
#
# Usage patterns:
#   1. As a SLURM job:   sbatch scripts/hpc/launch_emulator.sh Constrained
#   2. Interactively:    bash scripts/hpc/launch_emulator.sh Constrained
# ============================================================================

# --- Resolve PROJECT_ROOT relative to this file ---
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${ENV_DIR}/../.." && pwd)"

# --- Python / CUDA ---
export PYTHONUNBUFFERED=1

# Let SLURM manage GPU visibility when running as a job;
# only set manually for interactive use.
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# --- Thread control (critical for multiprocessing workers) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# --- Conda / micromamba ---
export LD_LIBRARY_PATH="/work/fquareng/.micromamba/envs/dl-stable/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Activate the environment.
if ! command -v micromamba &> /dev/null; then
    source /home/fquareng/.bashrc
fi

# --- Logging directory ---
export LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# --- Helper: generate a timestamped log path ---
# Usage: LOG_FILE=$(log_path "emulator_Constrained")
log_path() {
    local prefix="${1:-job}"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "${LOG_DIR}/${prefix}_${SLURM_JOB_ID}_${ts}.log"
    else
        echo "${LOG_DIR}/${prefix}_${ts}.log"
    fi
}

# --- Helper: print job info header ---
print_header() {
    echo "============================================="
    echo "Job:       ${SLURM_JOB_NAME:-interactive}"
    echo "Host:      $(hostname)"
    echo "Date:      $(date)"
    echo "Project:   ${PROJECT_ROOT}"
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "SLURM ID:  ${SLURM_JOB_ID}"
        echo "Partition: ${SLURM_JOB_PARTITION}"
        echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-auto}"
        echo "CPUs:      ${SLURM_CPUS_PER_TASK:-N/A}"
        echo "Memory:    ${SLURM_MEM_PER_NODE:-N/A} MB"
    fi
    echo "============================================="
}
