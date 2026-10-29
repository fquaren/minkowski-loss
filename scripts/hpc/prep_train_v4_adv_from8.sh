#!/bin/bash
# Resume prep_train_v4_adv.sh at step 8 (steps 1-6 passed 2026-10-21; step 7, the dataset checks, failed on
# two bugs in check_datasets.py, fixed and re-run by hand: ALL CHECKS PASSED). Same commands as the chain.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
P="taskset -c 4-11 $E/bin/python -u"
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
CFG=configs/config_v4_adv.yaml
set -o pipefail
echo "=== resume at step 8, code $(git rev-parse --short HEAD), $(date)"
echo "=== 8. validate training on a real batch (GPU 1) $(date)"
# 600 overfit steps: at 150 the real hourly batch reached MSE 36% of start (bar 20%); at 600, 20% and the
# structural term -73% (validation/training/20261021_1052_5b66c37.md). Criterion unchanged, step budget only.
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 8 --steps 600 2>&1 | grep -v Warn | tail -20 \
  || { echo "TRAINING VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== 9. smoke run $(date)"
SMOKE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_smoke.yaml
$E/bin/python src/sweep_util.py patch --base $CFG --out $SMOKE --set NUM_EPOCHS=1 || { echo "PATCH FAILED"; exit 1; }
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 0.5 $SMOKE || { echo "SMOKE FAILED"; exit 1; }
SMOKE_LOG=$(ls -t logs/unet_ana_*.log | head -1)
echo "=== smoke log: $SMOKE_LOG"; tail -20 "$SMOKE_LOG"
echo "=== 10. validate the smoke run's log $(date)"
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 8 --steps 600 --log "$SMOKE_LOG" 2>&1 \
  | grep -v Warn | tail -20 || { echo "SMOKE LOG VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== PREP_TRAIN_V4_ADV_DONE $(date)"
