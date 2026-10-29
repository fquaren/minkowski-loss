#!/bin/bash
# Training preparation on v4 (DECISIONS §33), started after build_v4.sh so it gets all 8 cores:
# persistence thresholds and gamma targets on the hourly store, dataset checks, and a smoke run
# (UNet backbone, 1 epoch, 0.5% of train, GPU 1), with the store and gamma validators as gates.
# Rerunnable; gamma skips splits already done.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
P="taskset -c 4-11 $E/bin/python -u"
BUILD_LOG=/work/fquareng/ch2/minkowski-loss/logs/build_v4.log
CFG=configs/config_v4.yaml
set -o pipefail
echo "=== prep_train_v4, code $(git rev-parse --short HEAD), $(date); waiting for the store build"
until grep -q V4_BUILD_DONE $BUILD_LOG; do
  grep -q "FAILED" $BUILD_LOG && { echo "BUILD FAILED, stopping"; exit 1; }
  sleep 120
done
echo "=== 0. validate the store (scripts/validate) $(date)"
$P scripts/validate/validate_stage.py store --config $CFG --n_days 8 --workers 4 2>&1 | grep -v Warn | tail -25 \
  || { echo "STORE VALIDATION FAILED: see validation/store/"; exit 1; }
echo "=== 1. persistence thresholds $(date)"
$P scripts/preprocess/compute_persistence_thresholds.py $CFG 2>&1 | tail -5 || { echo "PERSISTENCE FAILED"; exit 1; }
echo "=== 2. gamma targets $(date)"
$P scripts/preprocess/compute_gamma_targets.py $CFG 2>&1 | grep -v "it/s\]" | tail -15 || { echo "GAMMA FAILED"; exit 1; }
echo "=== 2b. validate the gamma targets $(date)"
$P scripts/validate/validate_stage.py gamma --config $CFG 2>&1 | grep -v Warn | tail -25 \
  || { echo "GAMMA VALIDATION FAILED: see validation/gamma/"; exit 1; }
echo "=== 3. dataset checks $(date)"
$P scripts/dataset_v2/check_datasets.py $CFG 2>&1 | tail -25 || { echo "CHECKS FAILED"; exit 1; }
echo "=== 3b. validate training on a real batch (GPU 1) $(date)"
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 8 --steps 150 2>&1 | grep -v Warn | tail -20 \
  || { echo "TRAINING VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== 4. smoke run $(date)"
SMOKE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_smoke.yaml
$E/bin/python src/sweep_util.py patch --base $CFG --out $SMOKE --set NUM_EPOCHS=1 || { echo "PATCH FAILED"; exit 1; }
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 0.5 $SMOKE || { echo "SMOKE FAILED"; exit 1; }
SMOKE_LOG=$(ls -t logs/unet_ana_*.log | head -1)
echo "=== smoke log: $SMOKE_LOG"
tail -20 "$SMOKE_LOG"
echo "=== 5. validate the smoke run's log $(date)"
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 4 --steps 30 --log "$SMOKE_LOG" 2>&1 \
  | grep -v Warn | tail -20 || { echo "SMOKE LOG VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== PREP_TRAIN_V4_DONE $(date)"
