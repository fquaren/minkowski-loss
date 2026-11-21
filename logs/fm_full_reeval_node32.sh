#!/bin/bash
# Re-eval of the FM full-list control and FM + Minkowski (2026-11-21) on valext / valsub with the current
# eval_extremes.py: their 10-30/10-31 evals predate the conditioned scores (exceedance ratio, hit, false alarms,
# twCRPS), so the scorecard shows n/a. New tags *_cond; the old dirs are kept. Our card by UUID, cores 4-11.
# sigma_r: the eval configs pin FM_RESIDUAL_SCALE 0.01266805361956358 (full backbone).
set -uo pipefail
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4 taskset -c 4-11 $E/bin/python -u"
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
OUTE=$W/eval_results/extremes
BB=$W/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
echo "=== FM full re-eval, code $(git rev-parse --short HEAD), $(date)"
for spec in "fm_energy_ctl_cond FMenergyctl_v4adv_20261030_141741" "fm_energy_qm_cond FMenergyqm_v4adv_20261031_022957"; do
  set -- $spec
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/$1 ] && { echo "exists: $1 $sub"; continue; }
    echo "=== $1 $sub $(date)"
    $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model fm \
      --fm_checkpoint $W/runs/sr_fm_minkowski/$2/fm_mink_latest.pth --backbone $BB --ensemble 16 --split val \
      --pot_threshold 50.1 --tag $1 --output_dir $OUTE/v4adv_val$sub > logs/fm_full_reeval_$1_$sub.log 2>&1 \
      && tail -2 logs/fm_full_reeval_$1_$sub.log || echo "EVAL FAILED $1 $sub"
  done
done
echo "=== FM_REEVAL_DONE $(date)"
