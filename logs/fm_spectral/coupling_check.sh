#!/bin/bash
# Spectral FM reward: single-batch coupling diagnostic, then gradient-norm budget. node32, GPU 1, cores 4-11.
cd /work/fquareng/ch2/minkowski-loss
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
WT=$PWD/.claude/worktrees/radar-day-exclusion
FMB=$WT/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
VAN=$WT/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
CFG=logs/fm_spectral/config_v4_adv_fm_spectral.yaml
PY="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
echo "=== diagnose_coupling $(date)"
$PY scripts/evaluate/diagnose_coupling.py $CFG --fm_checkpoint $FMB --backbone $VAN \
    --extreme_batch --ensemble 16 > logs/fm_spectral/diagnose_coupling.log 2>&1
echo "exit $? $(date)"
echo "=== grad_norm_terms $(date)"
$PY scripts/evaluate/grad_norm_terms.py $CFG --fm_checkpoint $FMB --backbone $VAN \
    > logs/fm_spectral/grad_norm_terms.log 2>&1
echo "exit $? $(date)"
