#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name mink_preproc
#SBATCH --output /scratch/fquareng/slurm_out/%j.out
#SBATCH --error  /scratch/fquareng/slurm_out/%j.err
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 128G
#SBATCH --time 48:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONFIG="${PROJECT_ROOT}/config.yaml"
LOG_FILE=$(log_path "preprocessing")

print_header
echo "Log: ${LOG_FILE}"

{
    echo "=== Preprocessing started at $(date) ==="

    echo "--- Stage 1: Zarr store creation ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/preprocess/preprocess_data.py" "$CONFIG"

    echo "--- Stage 2: Persistence thresholds ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/preprocess/compute_persistence_thresholds.py" "$CONFIG"

    echo "--- Stage 3: Gamma targets ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/preprocess/compute_gamma_targets.py" "$CONFIG"

    echo "--- Stage 4: Mixup augmentation ---"
    micromamba run -n dl-stable python \
        "${PROJECT_ROOT}/scripts/preprocess/apply_mixup.py" "$CONFIG"

    echo "=== Preprocessing complete at $(date) ==="
} 2>&1 | tee "$LOG_FILE"
