#!/bin/bash
# Structural-weight calibration, step 1a (DECISIONS §37): Minkowski's gradient share at initialisation
# on the hourly corrected data, same settings as the 15-min audit of 2026-09-23 (512 x 128, anneal 0.1,
# 10% of the data), so the two parity weights compare directly.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 \
  /work/fquareng/.micromamba/envs/dl-stable/bin/python -u tools/gradient_audit.py configs/config_v4_adv.yaml \
  --n_batches 512 --batch_size 128 --grad_at init --losses minkowski --skip_eval --anneal 0.1 \
  --data_percentage 10 --num_workers 1 --out_dir eval_results/gradient_audit/2026-10-21_v4adv_init \
  2>&1 | grep --line-buffered -v Warn
echo WEIGHT_CAL_INIT_DONE
