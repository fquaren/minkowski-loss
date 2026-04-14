#!/bin/bash
# Submit the full super-resolution experiment suite.
# The emulator trains first; SR jobs start after it completes.
#
# Usage: bash scripts/hpc/submit_sr_experiments.sh [emulator_checkpoint]
#   If no checkpoint is given, trains the Constrained emulator first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-}"

if [ -z "$CKPT" ]; then
    echo "No checkpoint provided — submitting emulator training first."
    EMU_JOB=$(sbatch --parsable \
        --job-name "mink_emu_Constrained" \
        "${SCRIPT_DIR}/launch_emulator.sh" Constrained)
    echo "  Emulator job: ${EMU_JOB}"
    DEPEND="--dependency=afterok:${EMU_JOB}"
else
    echo "Using existing checkpoint: ${CKPT}"
    DEPEND=""
fi

echo "Submitting SR experiments..."

# UNet baseline (MSE only, no dependency on emulator)
J1=$(sbatch --parsable --job-name "mink_unet_base" \
    "${SCRIPT_DIR}/launch_unet_emulator.sh" 0.0)
echo "  UNet baseline:      job ${J1}"

# UNet + emulator loss
J2=$(sbatch --parsable ${DEPEND} --job-name "mink_unet_emu" \
    "${SCRIPT_DIR}/launch_unet_emulator.sh" 0.001)
echo "  UNet + emulator:    job ${J2}"

# UNet + analytical loss (no dependency on emulator)
J3=$(sbatch --parsable --job-name "mink_unet_ana" \
    "${SCRIPT_DIR}/launch_unet_analytical.sh" 1.0)
echo "  UNet + analytical:  job ${J3}"

# DDPM baseline (no dependency on emulator)
J4=$(sbatch --parsable --job-name "mink_ddpm" \
    "${SCRIPT_DIR}/launch_ddpm.sh")
echo "  DDPM:               job ${J4}"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
