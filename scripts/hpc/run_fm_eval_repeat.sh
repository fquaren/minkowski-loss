#!/bin/bash
# DECISIONS §38: eval-to-eval sampling noise. eval_extremes does not seed the FM noise, so a second evaluation
# of the same clean FM checkpoint on the heavy subset measures how much the tail columns move between draws.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
VAN=$PWD/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
FMB=$PWD/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
OUT=eval_results/extremes/v4adv_valext
[ -d $OUT/fm_rep2 ] || env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u \
    scripts/evaluate/eval_extremes.py configs/config_v4_adv_fm_evalext.yaml --model fm --fm_checkpoint $FMB \
    --backbone $VAN --ensemble 16 --split val --pot_threshold 50.1 --tag fm_rep2 --output_dir $OUT 2>&1 | grep -v Warn | tail -3
echo "=== FM_REPEAT_DONE $(date)"
