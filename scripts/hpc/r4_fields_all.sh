#!/bin/bash
# R4 field panels, all losses but optical flow, seed 42 (six §38 val_extremes1024 patches). Arg: full | cut. node40 via node40_run.sh.
set -e; cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4
if [ "$1" = full ]; then
exec /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/evaluate/dump_fields.py /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/configs/config_v4_adv_fm_evalext.yaml \
  --model "MSE|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261105_150530/unet_latest.pth|" \
  --model "Minkowski|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261105_150601/unet_latest.pth|" \
  --model "area-only|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261109_125328/unet_latest.pth|" \
  --model "SSIM|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261106_162658/unet_latest.pth|" \
  --model "wet area|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261112_003004/unet_latest.pth|" \
  --model "spectral 0.15|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261108_080705/unet_latest.pth|" \
  --model "spectral 0.05|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261110_013608/unet_latest.pth|" \
  --model "spectral 0.015|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261110_213321/unet_latest.pth|" \
  --model "FM clean|fm|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth" \
  --split val --indices 354 519 979 257 393 764 --output eval_results/fields/v4adv_valext_r4_noflow_full.npz
else
exec /work/fquareng/.micromamba/envs/dl-stable/bin/python -u scripts/evaluate/dump_fields.py /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/configs/config_v4_adv_fm_evalext.yaml \
  --model "MSE|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261105_142458/unet_latest.pth|" \
  --model "MSE randdrop|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261107_151025/unet_latest.pth|" \
  --model "Minkowski|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261105_150429/unet_latest.pth|" \
  --model "area-only|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261109_125213/unet_latest.pth|" \
  --model "SSIM|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261105_203710/unet_latest.pth|" \
  --model "wet area|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261109_035723/unet_latest.pth|" \
  --model "spectral 0.15|backbone|/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical/UNet_Ana_20261107_122302/unet_latest.pth|" \
  --split val --indices 354 519 979 257 393 764 --output eval_results/fields/v4adv_valext_r4_noflow_cut.npz
fi
