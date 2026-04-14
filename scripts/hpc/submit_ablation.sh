#!/bin/bash
# Submit the emulator ablation study: one SLURM job per architecture.
# Usage: bash scripts/hpc/submit_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting emulator ablation study..."

for arch in Baseline Lipschitz Constrained; do
    JOB_ID=$(sbatch --parsable \
        --job-name "mink_emu_${arch}" \
        "${SCRIPT_DIR}/launch_emulator.sh" "$arch")
    echo "  ${arch}: submitted as job ${JOB_ID}"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
