#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name mink_eval
#SBATCH --output /scratch/fquareng/slurm_out/%j.out
#SBATCH --error  /scratch/fquareng/slurm_out/%j.err
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 6:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Arguments ---
EMULATOR_CKPT="${1:?Usage: sbatch launch_evaluation.sh <emulator_checkpoint>}"
CONFIG="${PROJECT_ROOT}/config.yaml"
LOG_FILE=$(log_path "evaluation")

print_header
echo "Checkpoint: ${EMULATOR_CKPT}"
echo "Log: ${LOG_FILE}"

{
    echo "=== Evaluation started at $(date) ==="

    echo "--- Emulator evaluation ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/evaluate/eval_emulator.py" \
        "$CONFIG" --checkpoint "$EMULATOR_CKPT"

    echo "--- Analytical baseline ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/evaluate/eval_baselines.py" "$CONFIG"

    echo "--- Feature inversion ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/evaluate/eval_inversion.py" \
        "$CONFIG" --checkpoint "$EMULATOR_CKPT"

    echo "=== Evaluation complete at $(date) ==="
} 2>&1 | tee "$LOG_FILE"
