#!/bin/bash
# R4 competing losses on node32 GPU 1 (researcher, 2026-11-05: after dist_qm_ret), seed 42, cut + full each:
# ssim, spectral, wetarea, opticalflow, at gradient-norm parity with Minkowski 6.0e-5 (weights.json of the R4
# parity audit, DECISIONS §43). Same configs, training and §37 val evals as run_r4_queue.sh; differences:
#   - starts when the dist_qm_ret chain appends DIST_QM_RET_DONE or a FAILED line *after* this script starts;
#   - the run dir is read from the trainer's open train.log: runs/ is shared with the node40 cells, so
#     "newest UNet_Ana_* dir" (the queue's rule) could pick a node40 run;
#   - no `pgrep -f` waits (they match the waiting shell's own command line).
# Hardware: these run on the RTX PRO 6000; the Minkowski / MSE seed-42 R4 cells run on node40's V100s.
# Run detached:  setsid nohup bash scripts/hpc/run_r4_competing_node32.sh >> logs/r4_competing_node32.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
OUTE=$W/eval_results
AUD=$OUTE/gradient_audit/2026-11-04_r4_parity
CHAIN=$W/logs/qm_bracket_evals_then_distret.log
DONE=$PWD/logs/r4_queue_done.txt; touch $DONE
CD=$PWD/logs/r4_configs; mkdir -p $CD
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"

echo "=== R4 competing losses on node32, code $(git rev-parse --short HEAD), $(date)"
[ "$(hostname -s)" = node32 ] || { echo "not node32"; exit 1; }
N0=$(wc -l < $CHAIN)
until tail -n +$((N0 + 1)) $CHAIN | grep -qE "DIST_QM_RET_DONE|FAILED"; do sleep 300; done
echo "=== dist_qm_ret chain ended: $(tail -n +$((N0 + 1)) $CHAIN | grep -E 'DIST_QM_RET_DONE|FAILED' | tail -1) $(date)"
until [ -f $AUD/weights.json ]; do sleep 300; done
wt() { $E/bin/python -c "import json; print(json.load(open('$AUD/weights.json'))['weights']['$1'])"; }

ev() {   # label run_dir  (as run_r4_queue.sh)
  local lab=$1 run=$2
  [ -d $OUTE/backbone/v4adv_val/$lab ] && { echo "exists, skipping eval $lab"; return; }
  $P scripts/evaluate/eval_backbone.py $CFGE --checkpoint runs/sr_analytical/$run/unet_latest.pth --split val \
    --output_dir $OUTE/backbone/v4adv_val/$lab 2>&1 | grep -v Warn | tail -3 || echo "BACKBONE EVAL FAILED $lab"
  $P scripts/evaluate/eval_extremes.py $CFGE --model backbone --checkpoint runs/sr_analytical/$run/unet_latest.pth \
    --split val --pot_threshold 50.1 --tag $lab --output_dir $OUTE/extremes/v4adv_val 2>&1 | grep -v Warn | tail -3 \
    || echo "EXTREMES EVAL FAILED $lab"
}

run() {   # label loss arm(cut|full) weight seed
  local lab=$1 loss=$2 arm=$3 w=$4 seed=$5
  grep -q "^$lab " $DONE && { echo "done already: $lab"; return; }
  local list=$META/full_train.txt; [ $arm = cut ] && list=$META/full_train_cut50.1.txt
  local cfg=$CD/$lab.yaml
  $E/bin/python - "$cfg" "$loss" "$list" <<'PY' || { echo "CONFIG FAILED $lab"; return; }
import sys, yaml                      # mkcfg of run_r4_queue.sh with channels 1,1,1 and no u_max
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

for L in ssim spectral wetarea opticalflow; do
  run r4_${L}_cut_s42  $L cut  $(wt $L) 42
  run r4_${L}_full_s42 $L full $(wt $L) 42
done
echo "=== R4_COMPETING_DONE $(date)"
