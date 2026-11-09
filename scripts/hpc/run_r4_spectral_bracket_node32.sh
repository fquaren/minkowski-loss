#!/bin/bash
# Spectral weight bracket for R4 (researcher, 2026-11-09), node32 GPU 1. Replaces run_r4_competing_node32.sh after
# its wetarea-cut run: spectral at the parity weight 0.15 games the tail (full s42: MAE +9.3% vs MSE, false alarms
# on 9.7% of val cases, RL100 bias +24).
#   0. wait for the running wetarea-cut training (PID given as $1) to exit; record it in the done list; evaluate it
#   1. spectral full list, seed 42, at w = 0.05 and 0.015; val evals (§37 protocol, u = 50.1)
#   2. choose the largest bracket weight that passes all of (stated before any bracket result exists):
#        active      MAE >= 1.01 x vanilla        (vanilla = UNet_Ana_20261021_211732, eval_results/.../vanilla)
#        accuracy    MAE <= 1.05 x vanilla        (§16)
#        anisotropy  <= 1.2 x vanilla             (§16)
#        no hacking  conditioned false-alarm rate <= 1e-3  (Minkowski full s42: 4.5e-4; MSE ~1e-4)
#      none passes -> no cut rerun, a FLAG line, carry on
#   3. spectral cut list, seed 42, at the chosen weight (its full arm is the bracket run at that weight)
#   4. exec run_r4_queue.sh: everything not yet in logs/r4_queue_done.txt (area-only pair, wetarea full,
#      opticalflow pair, MSE / Minkowski full s43, the four s44 runs)
# Labels: r4_spectral_<arm>_s42_w<weight>; the w = 0.15 rows (r4_spectral_*_s42) stay as they are.
# Run detached:  setsid nohup bash scripts/hpc/run_r4_spectral_bracket_node32.sh <pid> >> logs/r4_spectral_bracket.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
WAITPID=${1:?pid of the running wetarea-cut training}
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
OUTE=$W/eval_results
DONE=$PWD/logs/r4_queue_done.txt; touch $DONE
CD=$PWD/logs/r4_configs; mkdir -p $CD
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"
BRACKET="0.05 0.015"

echo "=== R4 spectral bracket on node32, code $(git rev-parse --short HEAD), $(date)"
[ "$(hostname -s)" = node32 ] || { echo "not node32"; exit 1; }

ev() {   # label run_dir  (as run_r4_competing_node32.sh)
  local lab=$1 run=$2
  [ -d $OUTE/backbone/v4adv_val/$lab ] && { echo "exists, skipping eval $lab"; return; }
  $P scripts/evaluate/eval_backbone.py $CFGE --checkpoint runs/sr_analytical/$run/unet_latest.pth --split val \
    --output_dir $OUTE/backbone/v4adv_val/$lab 2>&1 | grep -v Warn | tail -3 || echo "BACKBONE EVAL FAILED $lab"
  $P scripts/evaluate/eval_extremes.py $CFGE --model backbone --checkpoint runs/sr_analytical/$run/unet_latest.pth \
    --split val --pot_threshold 50.1 --tag $lab --output_dir $OUTE/extremes/v4adv_val 2>&1 | grep -v Warn | tail -3 \
    || echo "EXTREMES EVAL FAILED $lab"
}

run() {   # label loss arm(cut|full) weight seed  (as run_r4_competing_node32.sh)
  local lab=$1 loss=$2 arm=$3 w=$4 seed=$5
  grep -q "^$lab " $DONE && { echo "done already: $lab"; return; }
  local list=$META/full_train.txt; [ $arm = cut ] && list=$META/full_train_cut50.1.txt
  local cfg=$CD/$lab.yaml
  $E/bin/python - "$cfg" "$loss" "$list" <<'PY' || { echo "CONFIG FAILED $lab"; return; }
import sys, yaml
out, loss, train = sys.argv[1:4]
c = yaml.safe_load(open("configs/config_v4_adv.yaml"))
c["STRUCTURAL_LOSS"] = loss
c["TRAIN_METADATA_FILE"] = train
c["MINKOWSKI_FORM"] = "threshold"
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [1.0, 1.0, 1.0]
c["MINKOWSKI_MAX_THRESHOLD"] = None
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
  local log=logs/unet_ana_$(date +%Y%m%d_%H%M%S)_$lab.log
  echo "=== train $lab: loss $loss, w $w, list $(basename $list), seed $seed $(date); log $log"
  $P scripts/train/train_unet_analytical.py $cfg --params_path configs/unet_analytical.yaml --weight_geom $w \
    --data_percentage 100.0 --seed $seed > $log 2>&1 &
  local pid=$! rd=""
  for i in $(seq 1 60); do
    rd=$(ls -l /proc/$pid/fd 2>/dev/null | grep -oE "runs/sr_analytical/UNet_Ana_[0-9_]+/train.log" | head -1 | cut -d/ -f3)
    [ -n "$rd" ] && break; sleep 10
  done
  [ -n "$rd" ] || { echo "NO RUN DIR $lab (pid $pid)"; tail -5 $log; wait $pid; return; }
  echo "run $rd"; head -2 runs/sr_analytical/$rd/train.log
  wait $pid || { echo "TRAIN FAILED $lab"; tail -5 $log; return; }
  [ -f runs/sr_analytical/$rd/unet_latest.pth ] || { echo "TRAIN FAILED $lab: no checkpoint"; return; }
  grep -q "^$lab " $DONE || echo "$lab $rd" >> $DONE
  ev $lab $rd
}

# 0. the wetarea-cut run started by run_r4_competing_node32.sh (its driver is stopped, so record it here)
WLAB=r4_wetarea_cut_s42
WRD=$(ls -l /proc/$WAITPID/fd 2>/dev/null | grep -oE "runs/sr_analytical/UNet_Ana_[0-9_]+/train.log" | head -1 | cut -d/ -f3)
echo "=== 0. waiting for $WLAB (pid $WAITPID, run ${WRD:-?}) $(date)"
[ -n "$WRD" ] || { echo "pid $WAITPID has no run dir open; is it the wetarea run?"; exit 1; }
while [ -d /proc/$WAITPID ]; do sleep 120; done
if [ -f runs/sr_analytical/$WRD/unet_latest.pth ] && grep -q "Epoch 25 |" runs/sr_analytical/$WRD/train.log; then
  grep -q "^$WLAB " $DONE || echo "$WLAB $WRD" >> $DONE
  ev $WLAB $WRD
else
  echo "TRAIN FAILED $WLAB: no epoch 25 / checkpoint in $WRD"
fi

# 1. bracket, full list
for w in $BRACKET; do run r4_spectral_full_s42_w$w spectral full $w 42; done

# 2. choose the weight
CHOICE=$($E/bin/python - $OUTE "$BRACKET" <<'PY'
import sys, yaml
oute, ws = sys.argv[1], sys.argv[2].split()
ld = lambda p: yaml.safe_load(open(p))
v_mae = ld(f"{oute}/backbone/v4adv_val/vanilla/backbone_summary.yaml")["mae_overall"]
v_an = ld(f"{oute}/extremes/v4adv_val/vanilla/extremes_summary.yaml")["anisotropy_axis_over_diag"]
ok = []
for w in ws:
    lab = f"r4_spectral_full_s42_w{w}"
    try:
        mae = ld(f"{oute}/backbone/v4adv_val/{lab}/backbone_summary.yaml")["mae_overall"]
        x = ld(f"{oute}/extremes/v4adv_val/{lab}/extremes_summary.yaml")
    except OSError:
        print(f"{lab}: no eval", file=sys.stderr); continue
    an, fa = x["anisotropy_axis_over_diag"], x["conditioned"]["false_alarm_rate"]
    r = mae / v_mae
    passed = 1.01 <= r <= 1.05 and an <= 1.2 * v_an and fa <= 1e-3
    print(f"{lab}: MAE x{r:.3f}, anisotropy {an:.2f} (cap {1.2 * v_an:.2f}), false alarms {fa:.2e} -> "
          f"{'pass' if passed else 'fail'}", file=sys.stderr)
    if passed: ok.append(float(w))
print(f"{max(ok):g}" if ok else "NONE")
PY
)
echo "=== 2. bracket choice: $CHOICE $(date)"
if [ "$CHOICE" = NONE ]; then
  echo "FLAG: no spectral weight in {$BRACKET} passes; no cut rerun (researcher to decide)"
else
  # 3. cut arm at the chosen weight
  run r4_spectral_cut_s42_w$CHOICE spectral cut $CHOICE 42
fi
echo "=== R4_SPECTRAL_BRACKET_DONE $(date)"

# 4. the rest of the R4 queue (resumable: skips every label in the done list)
exec bash scripts/hpc/run_r4_queue.sh
