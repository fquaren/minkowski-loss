#!/bin/bash
# DECISIONS §41: corrected FM reward fine-tunes, one after the other on GPU 1, each evaluated on both subsets.
#   1. dist_qm     distributional reward, quantile_mass form, fixed anchor
#   2. energy_ctl  w_reward = 0 control (retention + anchor), same budget as the energy runs
#   3. energy_qm   energy reward, quantile_mass form, w_reward at gradient-norm parity with retention
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
FMB=$PWD/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
VAN=$PWD/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
PY="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
for run in dist_qm energy_ctl energy_qm; do
  cfg=configs/config_v4_adv_fm_$run.yaml
  name=$(grep '^EXPERIMENT_NAME:' $cfg | awk '{print $2}')
  log=logs/finetune_v4adv_$run.log
  echo "=== fine-tune $run ($name) $(date)"
  $PY scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $FMB --backbone $VAN > $log 2>&1 \
    || { echo "FINETUNE $run FAILED"; exit 1; }
  grep -E "FM_RESIDUAL_SCALE|reward mode=" $log | head -2
  rd=$(ls -td runs/sr_fm_minkowski/${name}_* | head -1)
  echo "=== fine-tune $run finished $(date); run $rd"
  for sub in ext sub; do
    $PY scripts/evaluate/eval_extremes.py configs/config_v4_adv_fm_eval$sub.yaml --model fm \
      --fm_checkpoint $rd/fm_mink_latest.pth --backbone $VAN --ensemble 16 --split val \
      --pot_threshold 50.1 --tag fm_$run --output_dir eval_results/extremes/v4adv_val$sub 2>&1 | grep -v Warn | tail -2 \
      || { echo "EVAL $run $sub FAILED"; exit 1; }
  done
  echo "=== eval $run done $(date)"
done
echo "=== QM_FINETUNES_DONE $(date)"
