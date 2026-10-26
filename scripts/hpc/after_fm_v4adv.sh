#!/bin/bash
# DECISIONS §38 step 1 -> 3 (first part): after the FM training ends, evaluate vanilla backbone, 6e-5 backbone and FM
# on the two fixed validation subsets, then run the coupling checks for both fine-tunes. Stops there: the
# fine-tunes are launched after the coupling results are read.
#   subsets: val_subset4096 (random, general metrics), val_extremes1024 (corrected tile max >= 20.2 mm, tail columns)
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
R=runs/sr_analytical
VAN=$PWD/$R/UNet_Ana_20261021_211732/unet_latest.pth
W6=$PWD/$R/UNet_Ana_20261025_042024/unet_latest.pth
until grep -q "Done." logs/fm_v4adv_launcher.log; do sleep 300; done
FMRUN=$(ls -td runs/sr_flow_matching/Extremes_20261026_09* | head -1)
echo "=== FM finished $(date); run $FMRUN"; grep -a "INFO - Epoch" $FMRUN/train.log | tail -3
FMB=$PWD/$FMRUN/fm_best.pth
for sub in sub ext; do
  CFG=configs/config_v4_adv_fm_eval$sub.yaml
  OUT=eval_results/extremes/v4adv_val$sub
  for spec in vanilla:$VAN mink_6e-05:$W6; do
    lab=${spec%%:*}; ck=${spec#*:}
    [ -d $OUT/$lab ] || $P scripts/evaluate/eval_extremes.py $CFG --model backbone --checkpoint $ck --split val \
        --pot_threshold 50.1 --tag $lab --output_dir $OUT 2>&1 | grep -v Warn | tail -2
  done
  echo "=== FM eval on $sub $(date)"
  [ -d $OUT/fm ] || $P scripts/evaluate/eval_extremes.py $CFG --model fm --fm_checkpoint $FMB --backbone $VAN \
      --ensemble 16 --split val --pot_threshold 50.1 --tag fm --output_dir $OUT 2>&1 | grep -v Warn | tail -3
done
echo "=== coupling checks $(date)"
$P scripts/evaluate/diagnose_coupling.py configs/config_v4_adv_fm.yaml --fm_checkpoint $FMB --backbone $VAN \
    --mode distributional --ensemble 4 --extreme_batch 2>&1 | grep -v Warn > logs/coupling_v4adv_distributional.log
$P scripts/evaluate/diagnose_coupling.py configs/config_v4_adv_fm.yaml --fm_checkpoint $FMB --backbone $VAN \
    --mode energy --ensemble 16 --extreme_batch 2>&1 | grep -v Warn > logs/coupling_v4adv_energy.log
echo "=== AFTER_FM_DONE $(date)"
