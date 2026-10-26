#!/bin/bash
# Validate the striping measure on the hourly bracket (DECISIONS §37/§38): same 1,024 validation patches.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
R=runs/sr_analytical
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 nice -n 5 $E/bin/python -u scripts/evaluate/striping_eval.py \
  configs/config_v4_adv.yaml --n 1024 --min_max 1.0 --heavy 20.2 --out notes/data_analysis_v4/striping_bracket.csv --models \
  vanilla=$R/UNet_Ana_20261021_211732/unet_latest.pth \
  w6e-5=$R/UNet_Ana_20261025_042024/unet_latest.pth \
  w1.8e-4=$R/UNet_Ana_20261022_160540/unet_latest.pth \
  w5.4e-4=$R/UNet_Ana_20261023_121956/unet_latest.pth \
  w1.8e-3=$R/UNet_Ana_20261024_083723/unet_latest.pth 2>&1 | grep -v -E "Warn|apply"
echo STRIPING_DONE
