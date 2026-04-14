#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name mink_emulator
#SBATCH --output /scratch/fquareng/slurm_out/%j.out
#SBATCH --error  /scratch/fquareng/slurm_out/%j.err
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Arguments ---
ARCH="${1:-Constrained}"
DATA_FRAC="${2:-1.0}"
CONFIG="${PROJECT_ROOT}/config.yaml"
LOG_FILE=$(log_path "emulator_${ARCH}")

print_header
echo "Architecture: ${ARCH}"
echo "Data fraction: ${DATA_FRAC}"
echo "Log: ${LOG_FILE}"

micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/train/train_emulator.py" \
    "$CONFIG" \
    --arch "$ARCH" \
    --data_fraction "$DATA_FRAC" \
    2>&1 | tee "$LOG_FILE"
