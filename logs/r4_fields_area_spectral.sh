#!/bin/bash
# R4 field panels, area-only and spectral bracket: the six §38 val_extremes1024 patches (unet_latest).
# Seconds of GPU; runs alongside the R4 training queue on GPU 1.
set -eo pipefail   # no -u: scripts/hpc/env.sh reads PROJECT_ROOT unset
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
source scripts/hpc/env.sh
R=$PWD/runs/sr_analytical
m=()
add() { m+=(--model "$1|backbone|$R/$2/unet_latest.pth|"); }
add "MSE full"            UNet_Ana_20261105_150530
add "Mink full"           UNet_Ana_20261105_150601
add "area-only full"      UNet_Ana_20261109_125328
add "area-only cut"       UNet_Ana_20261109_125213
add "spectral full 0.15"  UNet_Ana_20261108_080705
add "spectral full 0.05"  UNet_Ana_20261110_013608
add "spectral full 0.015" UNet_Ana_20261110_213321
PY="micromamba run -n dl-stable env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u"
$PY scripts/evaluate/dump_fields.py configs/config_v4_adv_fm_evalext.yaml "${m[@]}" \
    --split val --indices 354 519 979 257 393 764 --output eval_results/fields/v4adv_valext_r4_area_spectral.npz
$PY scripts/evaluate/make_plots.py --field_bundle eval_results/fields/v4adv_valext_r4_area_spectral.npz \
    --out figures/v4adv_valext_fields_r4_area_spectral --field_mode both
echo "=== R4_FIELDS_AREA_SPECTRAL_DONE $(date)"
