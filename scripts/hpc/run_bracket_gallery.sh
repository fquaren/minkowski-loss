#!/bin/bash
# Bracket review images (DECISIONS §37): the five models on the same 12 heavy-rain validation patches.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
R=runs/sr_analytical
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u scripts/evaluate/bracket_gallery.py \
  configs/config_v4_adv.yaml --n 12 --min_max 20.2 --out notes/data_analysis_v4/bracket_gallery.pdf --models \
  vanilla=$R/UNet_Ana_20261021_211732/unet_latest.pth \
  w6e-5=$R/UNet_Ana_20261025_042024/unet_latest.pth \
  w1.8e-4=$R/UNet_Ana_20261022_160540/unet_latest.pth \
  w5.4e-4=$R/UNet_Ana_20261023_121956/unet_latest.pth \
  w1.8e-3=$R/UNet_Ana_20261024_083723/unet_latest.pth 2>&1 | grep -v Warn | tail -3
