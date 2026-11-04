#!/bin/bash
# R4 runs of the extreme-evaluation ladder (DECISIONS §43 and amendments, 2026-11-04), one after another on our GPU.
#   0. wait for the qm-bracket eval chain (run_qm_bracket_evals_then_distret.sh) to finish
#   1. parity audits at the vanilla checkpoint (UNet_Ana_20261021_211732, best): the competing losses
#      (spectral, ssim, wetarea, opticalflow), and Minkowski full and area-only (channel weights 1,0,0), both on
#      the grid truncated below the cut (MINKOWSKI_MAX_THRESHOLD 50.1: 13 thresholds, 0.1-31.7 mm; §43 amendment)
#   2. weights by gradient-norm parity: each loss's Adam share = Minkowski (threshold form) at 6.0e-5,
#      i.e. 8.833 x 0.6 = 5.300 (audit 2026-10-22); w = lambda_used x 5.300 / adam_share, 2 significant figures
#   3. trainings (25 epochs, configs/config_v4_adv.yaml), each evaluated on validation right after
#      (§37 protocol: full_val, unet_latest.pth, u = 50.1; labels r4_<loss>_<arm>_s<seed>):
#      seed 42: MSE cut, Minkowski (truncated grid) cut + full, area-only (truncated) cut + full, ssim, spectral,
#      wetarea, opticalflow (cut + full each), MSE random-drop control; then seeds 43 and 44 of MSE and Minkowski
#      (truncated), cut + full. Full-data MSE seed 42 exists: UNet_Ana_20261021_211732. The full-grid Minkowski
#      6.0e-5 run (UNet_Ana_20261025_042024) stays the Study-1 model for R1; R4 compares truncated with truncated.
# Resumable: a run whose label is already in logs/r4_queue_done.txt is skipped.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
OUTE=$W/eval_results
CD=$PWD/logs/r4_configs; mkdir -p $CD
DONE=$PWD/logs/r4_queue_done.txt; touch $DONE
AUD=$OUTE/gradient_audit/2026-11-04_r4_parity
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"

echo "=== R4 queue, code $(git rev-parse --short HEAD), $(date)"
until grep -q -E "DIST_QM_RET_DONE|FAILED" $W/logs/qm_bracket_evals_then_distret.log; do sleep 300; done
echo "=== chain finished: $(grep -E 'DIST_QM_RET_DONE|FAILED' $W/logs/qm_bracket_evals_then_distret.log | tail -1)"

mkcfg() {   # name loss train_list channel_weights u_max|none -> config path
  $E/bin/python - "$CD/$1.yaml" "$2" "$3" "$4" "${5:-none}" <<'PY'
import sys, yaml
out, loss, train, cw, umax = sys.argv[1:6]
c = yaml.safe_load(open("configs/config_v4_adv.yaml"))
c["STRUCTURAL_LOSS"] = loss
c["TRAIN_METADATA_FILE"] = train
c["MINKOWSKI_FORM"] = "threshold"
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [float(x) for x in cw.split(",")]
c["MINKOWSKI_MAX_THRESHOLD"] = None if umax == "none" else float(umax)
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(out)
PY
}

if [ ! -f $AUD/weights.json ]; then
  mkdir -p $AUD/competing $AUD/mink_trunc $AUD/area_trunc
  echo "=== 1. parity audits $(date)"
  mkcfg audit_mink_trunc minkowski $META/full_train.txt 1,1,1 50.1 > /dev/null
  mkcfg audit_area_trunc minkowski $META/full_train.txt 1,0,0 50.1 > /dev/null
  $P tools/gradient_audit.py configs/config_v4_adv.yaml --run vanilla_v4=UNet_Ana_20261021_211732 --grad_at vanilla_v4 \
    --losses spectral ssim wetarea opticalflow --skip_eval --which best --n_batches 512 --batch_size 128 --anneal 0.1 \
    --data_percentage 10 --num_workers 2 --out_dir $AUD/competing 2>&1 | grep -v Warn | tail -12 || { echo "AUDIT FAILED"; exit 1; }
  for A in mink_trunc area_trunc; do
    $P tools/gradient_audit.py $CD/audit_$A.yaml --run vanilla_v4=UNet_Ana_20261021_211732 --grad_at vanilla_v4 \
      --losses minkowski --skip_eval --which best --n_batches 512 --batch_size 128 --anneal 0.1 \
      --data_percentage 10 --num_workers 2 --out_dir $AUD/$A 2>&1 | grep -v Warn | tail -6 || { echo "AUDIT FAILED"; exit 1; }
  done
  $E/bin/python - $AUD <<'PY' || { echo "WEIGHTS FAILED"; exit 1; }
import json, math, sys
aud = sys.argv[1]
target = 8.833116738640596 * 6.0e-5 / 1e-4          # Minkowski threshold form at 6.0e-5 (audit 2026-10-22)
w = {}
for sub, names in (("competing", ["spectral", "ssim", "wetarea", "opticalflow"]),
                   ("mink_trunc", ["minkowski"]), ("area_trunc", ["minkowski"])):
    t = [x for x in json.load(open(f"{aud}/{sub}/gradients.json")) if x["checkpoint"] == "vanilla_v4"][0]["terms"]
    for n in names:
        a, lam = t[n]["adam_share"], t[n]["lambda_used"]
        if not (a and math.isfinite(a) and a > 0):
            raise SystemExit(f"no finite Adam share for {n}")
        key = sub if sub in ("mink_trunc", "area_trunc") else n
        w[key] = float(f"{lam * target / a:.2g}")
        print(f"{key}: adam_share {a:.4g} at {lam:g} -> w {w[key]:.2e}")
json.dump({"target_adam_share": target, "weights": w}, open(f"{aud}/weights.json", "w"), indent=1)
PY
fi
wt() { $E/bin/python -c "import json; print(json.load(open('$AUD/weights.json'))['weights']['$1'])"; }

ev() {   # label run_dir
  local lab=$1 run=$2
  [ -d $OUTE/backbone/v4adv_val/$lab ] && { echo "exists, skipping eval $lab"; return; }
  $P scripts/evaluate/eval_backbone.py $CFGE --checkpoint runs/sr_analytical/$run/unet_latest.pth --split val \
    --output_dir $OUTE/backbone/v4adv_val/$lab 2>&1 | grep -v Warn | tail -3 || echo "BACKBONE EVAL FAILED $lab"
  $P scripts/evaluate/eval_extremes.py $CFGE --model backbone --checkpoint runs/sr_analytical/$run/unet_latest.pth \
    --split val --pot_threshold 50.1 --tag $lab --output_dir $OUTE/extremes/v4adv_val 2>&1 | grep -v Warn | tail -3 \
    || echo "EXTREMES EVAL FAILED $lab"
}

run() {   # label loss arm(cut|full|randdrop) weight seed channel_weights u_max
  local lab=$1 loss=$2 arm=$3 w=$4 seed=$5 cw=${6:-1,1,1} umax=${7:-none}
  grep -q "^$lab " $DONE && { echo "done already: $lab"; return; }
  local list=$META/full_train.txt
  [ $arm = cut ] && list=$META/full_train_cut50.1.txt
  [ $arm = randdrop ] && list=$META/full_train_randdrop50.1.txt
  local cfg=$(mkcfg $lab $loss $list $cw $umax)
  echo "=== train $lab: loss $loss, w $w, list $(basename $list), seed $seed, channels $cw, u_max $umax $(date)"
  bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml "$w" 100.0 $cfg $seed || { echo "TRAIN FAILED $lab"; return; }
  local rd=$(ls -td runs/sr_analytical/UNet_Ana_* | head -1 | xargs basename)
  head -2 runs/sr_analytical/$rd/train.log
  echo "$lab $rd" >> $DONE
  ev $lab $rd
}

echo "=== 3. trainings $(date)"
run r4_mse_cut_s42        minkowski cut  0               42
run r4_mink_cut_s42       minkowski cut  $(wt mink_trunc) 42 1,1,1 50.1
run r4_mink_full_s42      minkowski full $(wt mink_trunc) 42 1,1,1 50.1
run r4_area_cut_s42       minkowski cut  $(wt area_trunc) 42 1,0,0 50.1
run r4_area_full_s42      minkowski full $(wt area_trunc) 42 1,0,0 50.1
for L in ssim spectral wetarea opticalflow; do
  run r4_${L}_cut_s42     $L        cut  $(wt $L)       42
  run r4_${L}_full_s42    $L        full $(wt $L)       42
done
run r4_mse_randdrop_s42   minkowski randdrop 0          42
for S in 43 44; do
  run r4_mse_cut_s$S      minkowski cut  0              $S
  run r4_mink_cut_s$S     minkowski cut  $(wt mink_trunc) $S 1,1,1 50.1
  run r4_mse_full_s$S     minkowski full 0               $S
  run r4_mink_full_s$S    minkowski full $(wt mink_trunc) $S 1,1,1 50.1
done
echo "=== R4_QUEUE_DONE $(date)"
