#!/bin/bash
# One R4 cell on node40 (DECISIONS §43): train 25 epochs, then the §37 validation evals (full_val, unet_latest.pth,
# u = 50.1), then record "<label> <run>" in logs/r4_queue_done.txt so run_r4_queue.sh skips it.
#   node40_r4_cell.sh <gpu 0-3> <label> <arm cut|full> <weight|mink_trunc|area_trunc> <seed> [channels] [u_max]
# A named weight is read from the R4 parity audit's weights.json, waiting for it if the audits are still running.
# Run detached:  setsid nohup scripts/hpc/node40_r4_cell.sh ... > logs/node40/cell_<label>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1; LAB=$2; ARM=$3; WSPEC=$4; SEED=$5; CW=${6:-1,1,1}; UMAX=${7:-none}
E=/work/fquareng/.micromamba/envs/dl-stable
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
CFGE=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
OUTE=$W/eval_results
AUD=$OUTE/gradient_audit/2026-11-04_r4_parity
DONE=$PWD/logs/r4_queue_done.txt; touch $DONE
CD=$PWD/logs/r4_configs; mkdir -p $CD

waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
lockpid() { cat /tmp/fquareng_node40_gpu$GPU.pid; }

echo "=== cell $LAB on node40 GPU $GPU, code $(git rev-parse --short HEAD), $(date)"
grep -q "^$LAB " $DONE && { echo "done already: $LAB"; exit 0; }
case $WSPEC in
  mink_trunc|area_trunc)
    until [ -f $AUD/weights.json ]; do sleep 120; done
    WT=$($E/bin/python -c "import json; print(json.load(open('$AUD/weights.json'))['weights']['$WSPEC'])") ;;
  *) WT=$WSPEC ;;
esac
# The trainer names its run dir by the start second (UNet_Ana_<YYYYmmdd_HHMMSS>): stagger the GPUs so two cells
# released by the same weights.json never start in the same second, and wait for this GPU to be free.
sleep $((GPU * 30))
while [ -f /tmp/fquareng_node40_gpu$GPU.pid ] && kill -0 "$(lockpid)" 2>/dev/null; do sleep 60; done
LIST=$META/full_train.txt; [ $ARM = cut ] && LIST=$META/full_train_cut50.1.txt
CFG=$CD/$LAB.yaml
$E/bin/python - "$CFG" "$LIST" "$CW" "$UMAX" <<'PY' || { echo "CONFIG FAILED"; exit 1; }
import sys, yaml                      # same fields as mkcfg in run_r4_queue.sh
out, train, cw, umax = sys.argv[1:5]
c = yaml.safe_load(open("configs/config_v4_adv.yaml"))
c["STRUCTURAL_LOSS"] = "minkowski"
c["TRAIN_METADATA_FILE"] = train
c["MINKOWSKI_FORM"] = "threshold"
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [float(x) for x in cw.split(",")]
c["MINKOWSKI_MAX_THRESHOLD"] = None if umax == "none" else float(umax)
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
echo "=== train $LAB: w $WT, list $(basename $LIST), seed $SEED, channels $CW, u_max $UMAX $(date)"
L=$(scripts/hpc/launch_unet_node40.sh $GPU $CFG $WT $SEED $LAB) || { echo "LAUNCH FAILED $L"; exit 1; }
P=$(lockpid); echo "log $L pid $P"
RUN=""
for i in $(seq 1 60); do
  RUN=$(ls -l /proc/$P/fd 2>/dev/null | grep -oE "runs/sr_analytical/UNet_Ana_[0-9_]+/train.log" | head -1 | cut -d/ -f3)
  [ -n "$RUN" ] && break; sleep 10
done
[ -n "$RUN" ] || { echo "NO RUN DIR (pid $P)"; tail -5 $L; exit 1; }
echo "run $RUN"; head -2 runs/sr_analytical/$RUN/train.log
waitpid $P
CK=runs/sr_analytical/$RUN/unet_latest.pth
[ -f $CK ] || { echo "TRAIN FAILED $LAB: no $CK"; tail -5 $L; exit 1; }
grep -q "^$LAB " $DONE || echo "$LAB $RUN" >> $DONE
echo "=== trained $LAB $(date)"

if [ ! -d $OUTE/backbone/v4adv_val/$LAB ]; then
  L=$(scripts/hpc/node40_run.sh $GPU eval_bb_$LAB scripts/evaluate/eval_backbone.py $CFGE --checkpoint $CK \
      --split val --output_dir $OUTE/backbone/v4adv_val/$LAB) || { echo "BACKBONE EVAL LAUNCH FAILED $L"; exit 1; }
  waitpid $(lockpid); tail -3 $L
fi
L=$(scripts/hpc/node40_run.sh $GPU eval_ex_$LAB scripts/evaluate/eval_extremes.py $CFGE --model backbone \
    --checkpoint $CK --split val --pot_threshold 50.1 --tag $LAB --output_dir $OUTE/extremes/v4adv_val) \
    || { echo "EXTREMES EVAL LAUNCH FAILED $L"; exit 1; }
waitpid $(lockpid); tail -3 $L
echo "=== CELL_DONE $LAB $(date)"
