#!/bin/bash
# Baselines v2 G2 QM rows (eval only): MSE + global quantile mapping, both arms, seed 42, on one node40 GPU.
# Per arm: sample 16,384 rows of the arm's train list -> train-split histograms of the MSE model -> fit the map ->
# eval_extremes with --qm on val (u = 50.1) and on the FM subsets valext / valsub. eval_backbone has no --qm, so these
# rows have no MAE column. The cut model is fitted on the cut list, so its map can only extrapolate above u.
#   node40_qm_rows.sh <gpu 0-3>      (run detached, CPUs 0-5, as the cells)
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1
E=/work/fquareng/.micromamba/envs/dl-stable
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
OUTE=$W/eval_results/extremes
Q=$PWD/logs/node40/qm; mkdir -p $Q
lockpid() { cat /tmp/fquareng_node40_gpu$GPU.pid; }
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
run() {   # label script args... ; waits for it
  local L; L=$(scripts/hpc/node40_run.sh $GPU "$@") || { echo "LAUNCH FAILED $1: $L"; return 1; }
  waitpid $(lockpid); tail -2 $L
}

echo "=== QM rows on node40 GPU $GPU, code $(git rev-parse --short HEAD), $(date)"
for spec in "full UNet_Ana_20261105_150530 full_train.txt" "cut UNet_Ana_20261105_142458 full_train_cut50.1.txt"; do
  set -- $spec; ARM=$1; CK=$W/runs/sr_analytical/$2/unet_latest.pth; LIST=$META/$3; LAB=r4_mse_qm_${ARM}_s42
  [ -f $CK ] || { echo "NO CHECKPOINT $CK"; continue; }
  if [ ! -s $Q/qm_$ARM.json ]; then
    [ -s $Q/fit_$ARM.txt ] || $E/bin/python scripts/evaluate/quantile_map.py sample $LIST $Q/fit_$ARM.txt --n 16384
    $E/bin/python - $CFGE $Q/fit_$ARM.txt $Q/cfg_fit_$ARM.yaml <<'PY'
import sys, yaml
base, lst, out = sys.argv[1:4]
c = yaml.safe_load(open(base)); c["TRAIN_METADATA_FILE"] = lst
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
    echo "=== $ARM: train histograms ($CK) $(date)"
    run qm_hist_$ARM scripts/evaluate/eval_extremes.py $Q/cfg_fit_$ARM.yaml --model backbone --checkpoint $CK \
      --split train --pot_threshold 50.1 --qm_hist $Q/hist_$ARM.npz --tag qmfit_$ARM --output_dir $Q/fit_eval_$ARM
    [ -s $Q/hist_$ARM.npz ] || { echo "NO HISTOGRAM $ARM"; continue; }
    $E/bin/python scripts/evaluate/quantile_map.py fit $Q/hist_$ARM.npz $Q/qm_$ARM.json || { echo "FIT FAILED $ARM"; continue; }
  fi
  echo "=== $ARM: evals with $Q/qm_$ARM.json $(date)"
  [ -d $OUTE/v4adv_val/$LAB ] || run eval_ex_$LAB scripts/evaluate/eval_extremes.py $CFGE --model backbone \
      --checkpoint $CK --qm $Q/qm_$ARM.json --split val --pot_threshold 50.1 --tag $LAB --output_dir $OUTE/v4adv_val
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/$LAB ] || run eval_${sub}_$LAB scripts/evaluate/eval_extremes.py \
      $W/configs/config_v4_adv_fm_eval$sub.yaml --model backbone --checkpoint $CK --qm $Q/qm_$ARM.json \
      --split val --pot_threshold 50.1 --tag $LAB --output_dir $OUTE/v4adv_val$sub
  done
done
echo "=== QM_ROWS_DONE $(date)"
