#!/bin/bash
# FM area-only reward: weight bracket below parity (DECISIONS §16), node32, our GPU by UUID, cores 4-11.
# At the parity weight 41 both arms trip the peak-ratio guard after epoch 1 (full 1.204 > 1.150, cut
# 1.401 > 1.164), so area_{full,cut}_s42 are 1-epoch models. Bracket {41/3, 41/10} = {14, 4.1}, seed 42:
#   full@14; if it passes the guard: cut@14, then full@4.1 (whole bracket reported);
#   else: full@4.1, then cut@4.1 if it passes.
# Runs beside run_fm_r4_queue.sh on the same card. Abort = the trainer's "ABORT: reward hacking" line
# (it exits 0 then, which the queue cannot see). Results: logs/fm_r4/area_bracket.txt.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
R=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
OUTE=$W/eval_results/extremes
L=$PWD/logs/fm_r4; RES=$L/area_bracket.txt; touch $RES
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4 taskset -c 4-11 $E/bin/python -u"
EQM=$W/configs/config_v4_adv_fm_energy_qm.yaml

FMB_FULL=$W/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
BB_FULL=$W/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
SIG_FULL=0.01266805361956358
BB_CUT=$R/runs/sr_analytical/UNet_Ana_20261105_142458/unet_latest.pth
FMB_CUT=$(cat $L/fmb_cut.txt); SIG_CUT=$(cat $L/sigma_cut.txt)

mkcfg() {   # out weight train_list backbone sigma name  (as run_fm_r4_queue.sh, area-only energy)
  $E/bin/python - "$@" "$EQM" <<'PY'
import sys, yaml
out, w, train, bb, sig, name, base = sys.argv[1:8]
c = yaml.safe_load(open(base))
c["REWARD_MODE"] = "energy"
c["REWARD_WEIGHT"] = float(w)
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [1.0, 0.0, 0.0]
c["TRAIN_METADATA_FILE"] = train
c["BACKBONE_CHECKPOINT"] = bb
c["FM_RESIDUAL_SCALE"] = float(sig)
c["EXPERIMENT_NAME"] = name
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
}

run() {   # arm weight -> prints PASS or ABORT (or FAIL); evaluates passing runs on valext/valsub
  local arm=$1 w=$2 lab=area_$1_s42_w$2
  local hit=$(awk -v l=$lab '$1==l {print $2}' $RES)
  [ -n "$hit" ] && { echo $hit; return; }
  local list=$META/full_train.txt bb=$BB_FULL fmb=$FMB_FULL sig=$SIG_FULL
  [ $arm = cut ] && { list=$META/full_train_cut50.1.txt; bb=$BB_CUT; fmb=$FMB_CUT; sig=$SIG_CUT; }
  local cfg=$L/configs/$lab.yaml name=FMr4_$lab log=$L/ft_$lab.log
  mkcfg $cfg $w $list $bb $sig $name
  echo "=== fine-tune $lab: w $w, $(basename $list), sigma $sig $(date)" >&2
  $P scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $fmb --backbone $bb --seed 42 > $log 2>&1 \
    || { echo "$lab FAIL -" >> $RES; echo FAIL; return; }
  grep -q "config FM_RESIDUAL_SCALE" $log || { echo "$lab FAIL sigma_not_from_config" >> $RES; echo FAIL; return; }
  local rd=$(ls -td runs/sr_fm_minkowski/${name}_* | head -1)
  local ep=$(grep -ac "INFO - Epoch" $log)
  if grep -aq "ABORT: reward hacking" $log; then
    echo "$lab ABORT $rd epochs=$ep $(grep -aoE 'ABORT: reward hacking detected -- [^(]*' $log | head -1 | cut -d- -f3-)" >> $RES
    echo ABORT; return
  fi
  echo "$lab PASS $rd epochs=$ep" >> $RES
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/fm_r4_$lab ] && continue
    $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model fm \
      --fm_checkpoint $rd/fm_mink_latest.pth --backbone $bb --ensemble 16 --split val --pot_threshold 50.1 \
      --tag fm_r4_$lab --output_dir $OUTE/v4adv_val$sub 2>&1 | grep -v Warn | tail -2 >&2 || echo "EVAL FAILED $lab $sub" >&2
  done
  echo PASS
}

echo "=== FM area-only bracket, code $(git rev-parse --short HEAD), $(date)"
r14=$(run full 14); echo "full@14: $r14"
if [ "$r14" = PASS ]; then
  echo "cut@14: $(run cut 14)"
  echo "full@4.1: $(run full 4.1)"
else
  r4=$(run full 4.1); echo "full@4.1: $r4"
  [ "$r4" = PASS ] && echo "cut@4.1: $(run cut 4.1)"
fi
cat $RES
echo "=== AREA_BRACKET_DONE $(date)"
