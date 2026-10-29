#!/bin/bash
# Training preparation on the ADVECTION-CORRECTED target (DECISIONS §35), after build_v4.sh:
# advection store, plain-vs-corrected gallery, then the gated chain of prep_train_v4.sh on
# configs/config_v4_adv.yaml (store / gamma / training validators, dataset checks, smoke run).
# Rerunnable: the advection store is reused if built; gamma skips splits already done.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
P="taskset -c 4-11 $E/bin/python -u"
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
BUILD_LOG=/work/fquareng/ch2/minkowski-loss/logs/build_v4.log
ADV=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv
CFG=configs/config_v4_adv.yaml
set -o pipefail
echo "=== prep_train_v4_adv, code $(git rev-parse --short HEAD), $(date); waiting for the store build"
until grep -q V4_BUILD_DONE $BUILD_LOG; do
  grep -q "FAILED" $BUILD_LOG && { echo "BUILD FAILED, stopping"; exit 1; }
  sleep 120
done
if [ ! -f $ADV/meta/build_report.txt ]; then
  echo "=== 1. advection store $(date)"
  $P scripts/dataset_v2/build_adv_store.py --workers 8 2>&1 | tail -25 || { echo "ADV STORE FAILED"; exit 1; }
fi
echo "=== 2. plain vs corrected gallery $(date)"
$P scripts/analysis/v4_gallery_compare.py 2>&1 | grep -v Warn | tail -8 || echo "(gallery failed; continuing)"
echo "=== 3. validate the store $(date)"
$P scripts/validate/validate_stage.py store --config $CFG --n_days 8 --workers 4 2>&1 | grep -v Warn | tail -25 \
  || { echo "STORE VALIDATION FAILED: see validation/store/"; exit 1; }
echo "=== 4. persistence thresholds $(date)"
$P scripts/preprocess/compute_persistence_thresholds.py $CFG 2>&1 | tail -5 || { echo "PERSISTENCE FAILED"; exit 1; }
echo "=== 5. gamma targets $(date)"
$P scripts/preprocess/compute_gamma_targets.py $CFG 2>&1 | grep -v "it/s\]" | tail -15 || { echo "GAMMA FAILED"; exit 1; }
echo "=== 6. validate the gamma targets $(date)"
$P scripts/validate/validate_stage.py gamma --config $CFG 2>&1 | grep -v Warn | tail -25 \
  || { echo "GAMMA VALIDATION FAILED: see validation/gamma/"; exit 1; }
echo "=== 7. dataset checks $(date)"
$P scripts/dataset_v2/check_datasets.py $CFG 2>&1 | tail -25 || { echo "CHECKS FAILED"; exit 1; }
echo "=== 8. validate training on a real batch (GPU 1) $(date)"
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 8 --steps 150 2>&1 | grep -v Warn | tail -20 \
  || { echo "TRAINING VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== 9. smoke run $(date)"
SMOKE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_smoke.yaml
$E/bin/python src/sweep_util.py patch --base $CFG --out $SMOKE --set NUM_EPOCHS=1 || { echo "PATCH FAILED"; exit 1; }
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 0.5 $SMOKE || { echo "SMOKE FAILED"; exit 1; }
SMOKE_LOG=$(ls -t logs/unet_ana_*.log | head -1)
echo "=== smoke log: $SMOKE_LOG"; tail -20 "$SMOKE_LOG"
echo "=== 10. validate the smoke run's log $(date)"
$GPU $P scripts/validate/validate_stage.py training --config $CFG --batch 4 --steps 30 --log "$SMOKE_LOG" 2>&1 \
  | grep -v Warn | tail -20 || { echo "SMOKE LOG VALIDATION FAILED: see validation/training/"; exit 1; }
echo "=== PREP_TRAIN_V4_ADV_DONE $(date)"
