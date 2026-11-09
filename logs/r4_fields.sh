#!/bin/bash
# R4 field panels: the six §38 val_extremes1024 patches, every finished R4 U-Net (unet_latest).
# Seconds of GPU; runs alongside the R4 training queue on GPU 1.
set -eo pipefail   # no -u: scripts/hpc/env.sh reads PROJECT_ROOT unset
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
source scripts/hpc/env.sh
R=$PWD/runs/sr_analytical
m=()
add() { m+=(--model "$1|backbone|$R/$2/unet_latest.pth|"); }
add "MSE full"       UNet_Ana_20261105_150530
add "MSE cut"        UNet_Ana_20261105_142458
add "MSE cut s43"    UNet_Ana_20261107_150924
add "MSE randdrop"   UNet_Ana_20261107_151025
add "Mink full"      UNet_Ana_20261105_150601
add "Mink cut"       UNet_Ana_20261105_150429
add "Mink cut s43"   UNet_Ana_20261107_150954
add "SSIM full"      UNet_Ana_20261106_162658
add "SSIM cut"       UNet_Ana_20261105_203710
add "spectral full"  UNet_Ana_20261108_080705
add "spectral cut"   UNet_Ana_20261107_122302
PY="micromamba run -n dl-stable env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u"
$PY scripts/evaluate/dump_fields.py configs/config_v4_adv_fm_evalext.yaml "${m[@]}" \
    --split val --indices 354 519 979 257 393 764 --output eval_results/fields/v4adv_valext_r4.npz
$PY scripts/evaluate/make_plots.py --field_bundle eval_results/fields/v4adv_valext_r4.npz \
    --out figures/v4adv_valext_fields_r4 --field_mode both
echo "=== R4_FIELDS_DONE $(date)"
