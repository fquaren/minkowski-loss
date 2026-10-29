#!/bin/bash
# DECISIONS §38 step 3 (second part): evaluate each Minkowski fine-tune as soon as it finishes, with the
# protocol of the FM rows (16 members, heun-16, u = 50.1, both fixed validation subsets).
# Evaluates fm_mink_latest.pth, not fm_mink_best.pth: "best" is selected on the training reward (CLAUDE.md).
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
VAN=$PWD/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
for mode in dist energy; do
  until grep -q "fine-tune $mode finished" logs/finetunes_v4adv.log; do
    grep -q "FINETUNE $mode FAILED" logs/finetunes_v4adv.log && { echo "=== $mode failed, stopping"; exit 1; }
    sleep 300
  done
  RUN=$(ls -td runs/sr_fm_minkowski/FM${mode}_v4adv_* | head -1)
  echo "=== eval fm_$mode from $RUN $(date)"
  for sub in sub ext; do
    OUT=eval_results/extremes/v4adv_val$sub
    [ -d $OUT/fm_$mode ] || $P scripts/evaluate/eval_extremes.py configs/config_v4_adv_fm_eval$sub.yaml --model fm \
        --fm_checkpoint $PWD/$RUN/fm_mink_latest.pth --backbone $VAN --ensemble 16 --split val \
        --pot_threshold 50.1 --tag fm_$mode --output_dir $OUT 2>&1 | grep -v Warn | tail -3
  done
  echo "=== EVAL_${mode}_DONE $(date)"
done
echo "=== AFTER_FINETUNES_DONE $(date)"
