#!/bin/bash
# Same diagnostic at lr 1e-4: separates a small step from a coupling that cannot move.
cd /work/fquareng/ch2/minkowski-loss
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
WT=$PWD/.claude/worktrees/radar-day-exclusion
while [ $(grep -c "^exit" logs/fm_spectral/coupling_check.out) -lt 2 ]; do sleep 20; done
echo "=== diagnose_coupling lr 1e-4 $(date)"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u \
  scripts/evaluate/diagnose_coupling.py logs/fm_spectral/config_v4_adv_fm_spectral.yaml \
  --fm_checkpoint $WT/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth \
  --backbone $WT/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth \
  --extreme_batch --ensemble 16 --lr 1e-4 > logs/fm_spectral/diagnose_coupling_lr1e-4.log 2>&1
echo "exit $? $(date)"
