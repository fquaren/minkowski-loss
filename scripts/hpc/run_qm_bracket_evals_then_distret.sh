#!/bin/bash
# DECISIONS §41-§42 (researcher, 2026-11-02):
#   1. evaluate each U-Net quantile_mass bracket run as soon as it finishes, with the §37 protocol (full validation
#      list, unet_latest.pth, eval_backbone + eval_extremes at u = 50.1 mm); labels mink_qm_<w>;
#   2. after the bracket and its evals: dist_qm with retention (config_v4_adv_fm_dist_qm_ret.yaml: w_retain 1,
#      w_reward 1.0 by the energy_qm parity rule), gradient budget check, fine-tune, eval on both subsets.
# GPU 1, cores 4-11; the evals run beside the bracket training with 2 loader workers (5 + 3 = 8 cores).
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CFG=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 nice -n 5 $E/bin/python -u"
R=runs/sr_analytical
ev() {   # label run_dir
  local lab=$1 run=$2
  echo "=== eval $lab ($run) $(date)"
  [ -d eval_results/backbone/v4adv_val/$lab ] && { echo "exists, skipping $lab"; return; }
  $P scripts/evaluate/eval_backbone.py $CFG --checkpoint $R/$run/unet_latest.pth --split val \
      --output_dir eval_results/backbone/v4adv_val/$lab 2>&1 | grep -v Warn | tail -4 || echo "BACKBONE EVAL FAILED $lab"
  $P scripts/evaluate/eval_extremes.py $CFG --model backbone --checkpoint $R/$run/unet_latest.pth --split val \
      --pot_threshold 50.1 --tag $lab --output_dir eval_results/extremes/v4adv_val 2>&1 | grep -v Warn | tail -4 \
      || echo "EXTREMES EVAL FAILED $lab"
}
for w in 2.60e-04 7.70e-04 2.30e-03 7.70e-03; do
  until grep -q "run w=$w finished" logs/bracket_qm_v4adv.log; do sleep 300; done
  run=$(grep "run w=$w finished" logs/bracket_qm_v4adv.log | sed 's/.*run dir runs\/sr_analytical\///')
  grep -E "auxiliary structural loss|structural weight" $R/$run/train.log | head -2
  ev mink_qm_$w "$run"
done
until grep -q "QM_BRACKET_DONE" logs/bracket_qm_v4adv.log; do sleep 300; done
echo "=== QM_BRACKET_EVALS_DONE $(date)"

FMB=$PWD/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
VAN=$PWD/$R/UNet_Ana_20261021_211732/unet_latest.pth
cfg=configs/config_v4_adv_fm_dist_qm_ret.yaml
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
echo "=== gradient budget dist_qm_ret $(date)"
$P scripts/evaluate/grad_norm_terms.py $cfg --fm_checkpoint $FMB --backbone $VAN 2>&1 | grep -v Warn \
  > logs/grad_norm_terms_v4adv_dist_qm_ret.txt || { echo "GRAD NORM FAILED"; exit 1; }
tail -2 logs/grad_norm_terms_v4adv_dist_qm_ret.txt
echo "=== fine-tune dist_qm_ret $(date)"
log=logs/finetune_v4adv_dist_qm_ret.log
$P scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $FMB --backbone $VAN > $log 2>&1 \
  || { echo "FINETUNE dist_qm_ret FAILED"; exit 1; }
grep -E "FM_RESIDUAL_SCALE|reward mode=" $log | head -2
rd=$(ls -td runs/sr_fm_minkowski/FMdistqmret_v4adv_* | head -1)
echo "=== fine-tune finished $(date); run $rd"
for sub in ext sub; do
  $P scripts/evaluate/eval_extremes.py configs/config_v4_adv_fm_eval$sub.yaml --model fm \
    --fm_checkpoint $rd/fm_mink_latest.pth --backbone $VAN --ensemble 16 --split val \
    --pot_threshold 50.1 --tag fm_dist_qm_ret --output_dir eval_results/extremes/v4adv_val$sub 2>&1 | grep -v Warn | tail -2 \
    || { echo "EVAL dist_qm_ret $sub FAILED"; exit 1; }
done
echo "=== DIST_QM_RET_DONE $(date)"
