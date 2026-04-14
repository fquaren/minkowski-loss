#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name mink_ddpm
#SBATCH --output /scratch/fquareng/slurm_out/%j.out
#SBATCH --error  /scratch/fquareng/slurm_out/%j.err
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 72:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Arguments ---
DATA_PCT="${1:-100.0}"
CONFIG="${PROJECT_ROOT}/config.yaml"
PARAMS="${PROJECT_ROOT}/configs/ddpm.yaml"
LOG_FILE=$(log_path "ddpm")

print_header
echo "Data percentage: ${DATA_PCT}"
echo "Log: ${LOG_FILE}"

micromamba run -n dl-stable python "${PROJECT_ROOT}/scripts/train/train_ddpm.py" \
    "$CONFIG" \
    --params_path "$PARAMS" \
    --data_percentage "$DATA_PCT" \
    2>&1 | tee "$LOG_FILE"
