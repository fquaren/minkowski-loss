#!/bin/bash
# FM G3 queue (EXPERIMENTS G3): one seed (42) per competing reward, both arms, after the R4 seed queue.
# Rows from fm_g3_common.sh; weights from logs/fm_g3/weights.txt (fm_g3_parity.sh). 10 fine-tunes,
# ~12 h each. Resumable: finished labels are in logs/fm_g3/done.txt.
# The trainer exits 0 when its reward-hacking guard aborts and still writes checkpoints, so an aborted
# run is recorded as ABORT and not evaluated (the R4 queue evaluated two 1-epoch area models as complete).
source "$(dirname "$0")/fm_g3_common.sh"
DONE=$L/done.txt; touch $DONE
echo "=== FM G3 queue, code $(git rev-parse --short HEAD) $(git status --short src | wc -l) uncommitted in src, $(date)"
until grep -q "FM_R4_QUEUE_DONE" $L4/queue.log; do sleep 300; done
echo "=== R4 seed queue finished $(date)"
grep -q FM_G3_PARITY_DONE $L/parity.log || { echo "PARITY NOT DONE, STOP"; exit 1; }

ev_fm() {   # label run_dir backbone arm
  # the eval configs carry the full backbone's sigma_r; the cut arm needs its own (logs/tab/configs)
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/fm_g3_$1 ] && { echo "exists, skipping eval $1 $sub"; continue; }
    local cfg=$W/configs/config_v4_adv_fm_eval$sub.yaml; [ $4 = cut ] && cfg=$PWD/logs/tab/configs/config_v4_adv_fm_eval${sub}_cut.yaml
    $P scripts/evaluate/eval_extremes.py $cfg --model fm \
      --fm_checkpoint $2/fm_mink_latest.pth --backbone $3 --ensemble 16 --split val --pot_threshold 50.1 \
      --tag fm_g3_$1 --output_dir $OUTE/v4adv_val$sub 2>&1 | grep -v Warn | tail -2 || echo "EVAL FAILED $1 $sub"
  done
}

for arm in full cut; do
  for spec in "${ROWS[@]}"; do
    set -- $spec; row=$1 mode=$2 cw=$3 form=$4; lab=${row}_${arm}_s42
    grep -q "^$lab " $DONE && { echo "done already: $lab"; continue; }
    w=$(awk -v m="$row" '$1==m {print $2}' $L/weights.txt)
    [ -z "$w" ] && { echo "NO WEIGHT, SKIPPED $lab"; continue; }
    list=$META/full_train.txt bb=$BB_FULL fmb=$FMB_FULL sig=$SIG_FULL
    if [ $arm = cut ]; then
      list=$META/full_train_cut50.1.txt bb=$BB_CUT fmb=$(cat $L4/fmb_cut.txt) sig=$(cat $L4/sigma_cut.txt)
    fi
    cfg=$L/configs/$lab.yaml name=FMg3_$lab
    mkcfg $cfg $mode $w $cw $form $list $bb $sig $name
    echo "=== fine-tune $lab: mode $mode, form $form, w $w, channels $cw, $(basename $list), seed 42 $(date)"
    $P scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $fmb --backbone $bb --seed 42 \
      > $L/ft_$lab.log 2>&1 || { echo "FINETUNE FAILED $lab"; continue; }
    grep -aE "sigma_r|reward mode=|channel weights|reward:|guard baseline" $L/ft_$lab.log | head -5
    grep -q "config FM_RESIDUAL_SCALE" $L/ft_$lab.log || { echo "SIGMA NOT FROM CONFIG, NOT EVALUATED $lab"; continue; }
    rd=$(ls -td runs/sr_fm_minkowski/${name}_* | head -1)
    ep=$(grep -ac "Epoch [0-9]* |" $L/ft_$lab.log)
    if grep -aq "ABORT: reward hacking" $L/ft_$lab.log; then
      echo "$lab ABORT $rd $w epochs=$ep $(grep -aoE 'ABORT: reward hacking detected -- [^(]*' $L/ft_$lab.log | head -1 | cut -d- -f3-)" | tee -a $DONE
      continue
    fi
    echo "$lab $rd $w epochs=$ep" >> $DONE
    ev_fm $lab $rd $bb $arm
    echo "=== $lab done $(date)"
  done
done
echo "=== FM_G3_QUEUE_DONE $(date)"
