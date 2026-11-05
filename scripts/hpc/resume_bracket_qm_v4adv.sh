#!/bin/bash
# Resume the last U-Net quantile_mass bracket run (w = 7.7e-3), cut at epoch 15/25 by the node reboot of
# 2026-11-03 16:11, from its unet_latest.pth (end of epoch 14). Then write the two lines that
# run_bracket_qm_v4adv.sh would have written, so run_qm_bracket_evals_then_distret.sh continues.
# The bracket's runs and logs live in the radar-day-exclusion worktree.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion || exit 1
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
w=7.70e-03
rd=runs/sr_analytical/UNet_Ana_20261103_041030
log=logs/unet_ana_$(date +%Y%m%d_%H%M%S)_resume.log
echo "=== resume run w=$w from $rd/unet_latest.pth $(date); log $log" >> logs/bracket_qm_v4adv.log
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u \
  scripts/train/train_unet_analytical.py configs/config_v4_adv_qm.yaml --params_path configs/unet_analytical.yaml \
  --weight_geom $w --data_percentage 100.0 --resume $rd/unet_latest.pth > $log 2>&1 \
  || { echo "=== RESUME w=$w FAILED $(date)" >> logs/bracket_qm_v4adv.log; exit 1; }
echo "=== run w=$w finished $(date); run dir $rd" >> logs/bracket_qm_v4adv.log
echo "=== QM_BRACKET_DONE $(date)" >> logs/bracket_qm_v4adv.log
