#!/bin/bash
# SSIM own-loss and equilibrium check for the R4 SSIM runs (w=0.016), against vanilla v4 (unet_latest).
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
OUT=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/eval_results/gradient_audit/2026-11-09_r4_ssim_own
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u tools/gradient_audit.py \
  configs/config_v4_adv.yaml --run vanilla=UNet_Ana_20261021_211732 ssim_cut=UNet_Ana_20261105_203710:ssim \
  ssim_full=UNet_Ana_20261106_162658:ssim mse_cut=UNet_Ana_20261105_142458:ssim mse_full=UNet_Ana_20261105_150530:ssim \
  --runs vanilla mse_full mse_cut ssim_full ssim_cut --grad_at ssim_full ssim_cut vanilla --losses ssim \
  --which latest --n_batches 256 --batch_size 128 --data_percentage 10 --num_workers 2 --out_dir $OUT 2>&1 | grep -v Warn
echo "=== SSIM_OWN_DONE $(date)"
