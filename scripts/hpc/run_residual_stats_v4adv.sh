#!/bin/bash
# Study 2 step 0a: sigma_r of the hourly vanilla backbone (last checkpoint) on the corrected store's train split.
# Writes patches_v4_adv/residual_stats.json and residual_std.npy; the value is then pinned as FM_RESIDUAL_SCALE.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u scripts/preprocess/compute_residual_stats.py \
  configs/config_v4_adv.yaml --backbone runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth --split train \
  2>&1 | grep --line-buffered -v -E "Warn|it/s\]"
echo RESIDUAL_STATS_DONE
