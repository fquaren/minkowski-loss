#!/bin/bash
# DECISIONS §38 step 2: the two Minkowski fine-tunes of the hourly FM, one after the other on GPU 1.
#   (a) distributional (configs/config_v4_adv_fm_dist.yaml), (b) energy + retention (config_v4_adv_fm_energy.yaml).
# Both start from the FM best checkpoint with the hourly vanilla backbone; fp32 (the reward trainer has no AMP).
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
FMB=$PWD/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
VAN=$PWD/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
for mode in dist energy; do
  echo "=== fine-tune $mode $(date)"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u scripts/train/train_fm_minkowski.py \
    configs/config_v4_adv_fm_$mode.yaml --fm_checkpoint $FMB --backbone $VAN > logs/finetune_v4adv_$mode.log 2>&1 \
    || { echo "FINETUNE $mode FAILED"; exit 1; }
  echo "=== fine-tune $mode finished $(date); run $(ls -td runs/sr_fm_minkowski/FM${mode}* 2>/dev/null | head -1)"
done
echo "=== FINETUNES_DONE $(date)"
