#!/bin/bash
# Figures for the finished G2 U-Net runs and the FM cut-arm rows (2026-11-21), on node32's card beside the FM jobs.
#  1. G2 cut models on the FM subsets valext / valsub (for the scorecard; the full ones exist).
#  2. Field panels on the six §38 val_extremes1024 patches: G2 full, G2 cut, FM full vs cut (member 0).
# Our card by UUID, cores 4-11; packed read path (this worktree's code).
set -uo pipefail
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4 taskset -c 4-11 $E/bin/python -u"
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
F=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4
R=$W/runs/sr_analytical
OUTE=$W/eval_results/extremes
CFX=$W/configs/config_v4_adv_fm_evalext.yaml
echo "=== G2 figures, code $(git rev-parse --short HEAD), $(date)"

for spec in "r4_wmse_cut_s42 UNet_Ana_20261120_003042" "r4_quantile_cut_s42 UNet_Ana_20261120_004843" \
            "r4_spectral_hann_cut_s42 UNet_Ana_20261120_004443"; do
  set -- $spec
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/$1 ] && continue
    $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model backbone \
      --checkpoint $R/$2/unet_latest.pth --split val --pot_threshold 50.1 --tag $1 --output_dir $OUTE/v4adv_val$sub \
      2>&1 | grep -v Warn | tail -1 || echo "SUBSET EVAL FAILED $1 $sub"
  done
done

IDX="354 519 979 257 393 764"
fields() {   # name model-args...
  local name=$1; shift
  $P scripts/evaluate/dump_fields.py $CFX "$@" --split val --indices $IDX \
    --output $W/eval_results/fields/v4adv_valext_$name.npz || { echo "DUMP FAILED $name"; return; }
  $P scripts/evaluate/make_plots.py --field_bundle $W/eval_results/fields/v4adv_valext_$name.npz \
    --out figures/v4adv_valext_fields_$name --field_mode both || echo "PLOT FAILED $name"
}
u() { echo "--model"; echo "$1|backbone|$R/$2/unet_latest.pth|"; }
mapfile -t A < <(u "MSE full" UNet_Ana_20261105_150530; u "Mink full" UNet_Ana_20261105_150601
  u "wmse full" UNet_Ana_20261118_151313; u "quantile full" UNet_Ana_20261118_151343
  u "fss full" UNet_Ana_20261118_151413; u "spectral hann full" UNet_Ana_20261118_151446)
fields r4_g2_full "${A[@]}"
mapfile -t A < <(u "MSE cut" UNet_Ana_20261105_142458; u "Mink cut" UNet_Ana_20261105_150429
  u "wmse cut" UNet_Ana_20261120_003042; u "quantile cut" UNet_Ana_20261120_004843
  u "spectral hann cut" UNet_Ana_20261120_004443)
fields r4_g2_cut "${A[@]}"
BBF=$R/UNet_Ana_20261021_211732/unet_latest.pth; BBC=$R/UNet_Ana_20261105_142458/unet_latest.pth
C=$F/logs/fm_r4/configs
fields fm_r4_cut \
  --model "FM clean full|fm|$W/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth|$BBF|$W/configs/config_v4_adv_fm.yaml" \
  --model "FM ctl full|fm|$W/runs/sr_fm_minkowski/FMenergyctl_v4adv_20261030_141741/fm_mink_latest.pth|$BBF|$W/configs/config_v4_adv_fm_energy_ctl.yaml" \
  --model "FM Mink full|fm|$W/runs/sr_fm_minkowski/FMenergyqm_v4adv_20261031_022957/fm_mink_latest.pth|$BBF|$W/configs/config_v4_adv_fm_energy_qm.yaml" \
  --model "FM clean cut|fm|$(cat $F/logs/fm_r4/fmb_cut.txt)|$BBC|$C/fm_base_cut.yaml" \
  --model "FM ctl cut|fm|$F/runs/sr_fm_minkowski/FMr4_ctl_cut_s42_20261120_090336/fm_mink_latest.pth|$BBC|$C/ctl_cut_s42.yaml" \
  --model "FM Mink cut|fm|$F/runs/sr_fm_minkowski/FMr4_mink_cut_s42_20261120_211914/fm_mink_latest.pth|$BBC|$C/mink_cut_s42.yaml"
echo "=== G2_FIGS_DONE $(date)"
