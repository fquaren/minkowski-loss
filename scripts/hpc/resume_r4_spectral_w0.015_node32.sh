#!/bin/bash
# Resume r4_spectral_full_s42_w0.015 (stopped 2026-11-11 for the node32 GPU0 reset) from its epoch-21
# unet_latest, evaluate it, then continue the bracket chain (step 0 skipped) and the R4 queue.
# Run detached:  setsid nohup bash scripts/hpc/resume_r4_spectral_w0.015_node32.sh >> logs/r4_spectral_bracket.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
[ "$(hostname -s)" = node32 ] || { echo "not node32"; exit 1; }
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
OUTE=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/eval_results
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
LAB=r4_spectral_full_s42_w0.015 RD=UNet_Ana_20261110_213321
log=logs/unet_ana_$(date +%Y%m%d_%H%M%S)_${LAB}_resume.log
echo "=== resume $LAB in $RD $(date); log $log"
$P scripts/train/train_unet_analytical.py logs/r4_configs/$LAB.yaml --params_path configs/unet_analytical.yaml \
  --weight_geom 0.015 --data_percentage 100.0 --seed 42 --resume runs/sr_analytical/$RD/unet_latest.pth > $log 2>&1
grep "resumed from" runs/sr_analytical/$RD/train.log | tail -1
grep -q "Epoch 25 |" runs/sr_analytical/$RD/train.log || { echo "TRAIN FAILED $LAB: no epoch 25"; tail -5 $log; exit 1; }
grep -q "^$LAB " logs/r4_queue_done.txt || echo "$LAB $RD" >> logs/r4_queue_done.txt
$P scripts/evaluate/eval_backbone.py $CFGE --checkpoint runs/sr_analytical/$RD/unet_latest.pth --split val \
  --output_dir $OUTE/backbone/v4adv_val/$LAB 2>&1 | grep -v Warn | tail -3 || echo "BACKBONE EVAL FAILED $LAB"
$P scripts/evaluate/eval_extremes.py $CFGE --model backbone --checkpoint runs/sr_analytical/$RD/unet_latest.pth \
  --split val --pot_threshold 50.1 --tag $LAB --output_dir $OUTE/extremes/v4adv_val 2>&1 | grep -v Warn | tail -3 \
  || echo "EXTREMES EVAL FAILED $LAB"
exec bash scripts/hpc/run_r4_spectral_bracket_node32.sh none
