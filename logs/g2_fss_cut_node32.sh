#!/bin/bash
# fss cut landed 2026-11-21 22:44 (node40): its valext/valsub evals for the scorecard, and the G2 cut field panels
# redone with fss in. Our card by UUID, cores 4-11; packed read path.
set -uo pipefail
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4 taskset -c 4-11 $E/bin/python -u"
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
R=$W/runs/sr_analytical
OUTE=$W/eval_results/extremes
echo "=== fss cut figures, code $(git rev-parse --short HEAD), $(date)"
for sub in ext sub; do
  [ -d $OUTE/v4adv_val$sub/r4_fss_cut_s42 ] && continue
  $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model backbone \
    --checkpoint $R/UNet_Ana_20261120_071846/unet_latest.pth --split val --pot_threshold 50.1 --tag r4_fss_cut_s42 \
    --output_dir $OUTE/v4adv_val$sub > logs/g2_fss_cut_eval_$sub.log 2>&1 && tail -1 logs/g2_fss_cut_eval_$sub.log \
    || echo "SUBSET EVAL FAILED $sub"
done
IDX="354 519 979 257 393 764"
u() { echo "--model"; echo "$1|backbone|$R/$2/unet_latest.pth|"; }
mapfile -t A < <(u "MSE cut" UNet_Ana_20261105_142458; u "Mink cut" UNet_Ana_20261105_150429
  u "wmse cut" UNet_Ana_20261120_003042; u "quantile cut" UNet_Ana_20261120_004843
  u "fss cut" UNet_Ana_20261120_071846; u "spectral hann cut" UNet_Ana_20261120_004443)
$P scripts/evaluate/dump_fields.py $W/configs/config_v4_adv_fm_evalext.yaml "${A[@]}" --split val --indices $IDX \
  --output $W/eval_results/fields/v4adv_valext_r4_g2_cut.npz \
  && $P scripts/evaluate/make_plots.py --field_bundle $W/eval_results/fields/v4adv_valext_r4_g2_cut.npz \
       --out figures/v4adv_valext_fields_r4_g2_cut --field_mode both || echo "PANELS FAILED"
echo "=== G2_FSS_CUT_DONE $(date)"
