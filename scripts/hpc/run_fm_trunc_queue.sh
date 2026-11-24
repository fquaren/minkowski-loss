#!/bin/bash
# FM reward with the grid truncated below 50.1 mm (MINKOWSKI_MAX_THRESHOLD, as the U-Net loss; researcher 2026-11-24).
# Replaces the full-grid FM + Minkowski rows and the G3 area-only row. Order (researcher: truncated before G3):
#   1. parity (|g_fm| / |g_reward|, as fm_g3_parity.sh) for: Minkowski on the full grid (known answer: the energy_qm
#      weight 1.0), Minkowski truncated, area-only truncated; a 3-step smoke run of each truncated config
#   2. minktr s42 full, cut      3. areatr s42 full, cut
#   4. the G3 queue for its other rows (its area row is marked DROPPED in logs/fm_g3/done.txt)
#   5. minktr s43, s44, both arms
# Evals: valext and valsub, 16 members, into v4adv_val{ext,sub}_tab (the main-table directories), cut arm with the
# cut sigma_r. Resumable: finished labels in logs/fm_trunc/done.txt.
source "$(dirname "$0")/fm_g3_common.sh"
LT=$PWD/logs/fm_trunc; mkdir -p $LT/configs
DONE=$LT/done.txt; touch $DONE
TRUNC="MINKOWSKI_MAX_THRESHOLD=50.1"
echo "=== FM trunc queue, code $(git rev-parse --short HEAD) $(git status --short src | wc -l) uncommitted in src, $(date)"
until grep -q "FM_R4_QUEUE_DONE" $L4/queue.log; do sleep 300; done
echo "=== R4 seed queue finished $(date)"

# ---- 1. parity + smoke ----------------------------------------------------------------------
# label channels form extra
PAR=("minkfull 1,1,1 quantile_mass -" "minktr 1,1,1 quantile_mass $TRUNC" "areatr 1,0,0 quantile_mass_max $TRUNC")
for spec in "${PAR[@]}"; do
  set -- $spec; lab=$1 cw=$2 form=$3 extra=$4; [ "$extra" = - ] && extra=""
  grep -q "^$lab " $LT/weights.txt 2>/dev/null && continue
  mkcfg $LT/configs/parity_$lab.yaml energy 1.0 $cw $form $META/full_train.txt $BB_FULL $SIG_FULL parity $extra
  $P scripts/evaluate/grad_norm_terms.py $LT/configs/parity_$lab.yaml --fm_checkpoint $FMB_FULL \
    --backbone $BB_FULL --n_batches 4 > $LT/parity_$lab.log 2>&1 || { echo "PARITY FAILED $lab"; continue; }
  w=$(grep "^mean |g|" $LT/parity_$lab.log | awk '{printf "%.2g", $8 / $4}')
  [ -z "$w" ] && { echo "PARITY FAILED $lab"; continue; }
  echo "$lab $w" >> $LT/weights.txt
  echo "parity $lab: w $w | $(grep -E '^reward grid|^\[grad\] mode|share' $LT/parity_$lab.log | tr '\n' ' ')"
  [ -z "$extra" ] && continue
  mkcfg $LT/configs/smoke_$lab.yaml energy $w $cw $form $META/full_train.txt $BB_FULL $SIG_FULL smokeTR_$lab \
    REWARD_EPOCHS=1 REWARD_VAL_BATCHES=2 REWARD_GUARD_BATCHES=2 $extra
  $P scripts/train/train_fm_minkowski.py $LT/configs/smoke_$lab.yaml --fm_checkpoint $FMB_FULL \
    --backbone $BB_FULL --seed 42 --max_steps 3 > $LT/smoke_$lab.log 2>&1 \
    && echo "smoke $lab OK: $(grep -aE 'sigma_r|reward mode=|reward grid|channel weights' $LT/smoke_$lab.log | cut -c25- | tr '\n' ' ')" \
    || echo "SMOKE FAILED $lab: $(grep -aE 'Error|Traceback' $LT/smoke_$lab.log | tail -2 | tr '\n' ' ')"
done
echo "=== FM_TRUNC_PARITY_DONE $(date)"

ev_fm() {   # label run_dir backbone arm
  for sub in ext sub; do
    local out=$OUTE/v4adv_val${sub}_tab
    [ -f $out/fm_$1/extremes_summary.yaml ] && { echo "exists, skipping eval $1 $sub"; continue; }
    local cfg=$W/configs/config_v4_adv_fm_eval$sub.yaml; [ $4 = cut ] && cfg=$PWD/logs/tab/configs/config_v4_adv_fm_eval${sub}_cut.yaml
    $P scripts/evaluate/eval_extremes.py $cfg --model fm \
      --fm_checkpoint $2/fm_mink_latest.pth --backbone $3 --ensemble 16 --split val --pot_threshold 50.1 \
      --tag fm_$1 --output_dir $out 2>&1 | grep -v Warn | tail -2 || echo "EVAL FAILED $1 $sub"
  done
}

ft() {   # row(minktr|areatr) arm seed
  local row=$1 arm=$2 seed=$3 lab=${1}_${2}_s${3}
  grep -q "^$lab " $DONE && { echo "done already: $lab"; return; }
  local w=$(awk -v m="$row" '$1==m {print $2}' $LT/weights.txt)
  [ -z "$w" ] && { echo "NO WEIGHT, SKIPPED $lab"; return; }
  grep -q "SMOKE FAILED $row" $LT/queue.log 2>/dev/null && { echo "SMOKE FAILED EARLIER, SKIPPED $lab"; return; }
  local cw=1,1,1 form=quantile_mass; [ $row = areatr ] && cw=1,0,0 form=quantile_mass_max
  local list=$META/full_train.txt bb=$BB_FULL fmb=$FMB_FULL sig=$SIG_FULL
  if [ $arm = cut ]; then
    list=$META/full_train_cut50.1.txt bb=$BB_CUT fmb=$(cat $L4/fmb_cut.txt) sig=$(cat $L4/sigma_cut.txt)
  fi
  local cfg=$LT/configs/$lab.yaml name=FMtr_$lab
  mkcfg $cfg energy $w $cw $form $list $bb $sig $name $TRUNC
  echo "=== fine-tune $lab: form $form, w $w, channels $cw, $(basename $list), seed $seed $(date)"
  $P scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $fmb --backbone $bb --seed $seed \
    > $LT/ft_$lab.log 2>&1 || { echo "FINETUNE FAILED $lab"; return; }
  grep -aE "sigma_r|reward mode=|reward grid|channel weights|guard baseline" $LT/ft_$lab.log | head -5
  grep -q "config FM_RESIDUAL_SCALE" $LT/ft_$lab.log || { echo "SIGMA NOT FROM CONFIG, NOT EVALUATED $lab"; return; }
  grep -q "MINKOWSKI_MAX_THRESHOLD=50.1" $LT/ft_$lab.log || { echo "GRID NOT TRUNCATED, NOT EVALUATED $lab"; return; }
  local rd=$(ls -td runs/sr_fm_minkowski/${name}_* | head -1)
  local ep=$(grep -ac "Epoch [0-9]* |" $LT/ft_$lab.log)
  if grep -aq "ABORT: reward hacking" $LT/ft_$lab.log; then
    echo "$lab ABORT $rd $w epochs=$ep $(grep -aoE 'ABORT: reward hacking detected -- [^(]*' $LT/ft_$lab.log | head -1 | cut -d- -f3-)" | tee -a $DONE
    return
  fi
  echo "$lab $rd $w epochs=$ep" >> $DONE
  ev_fm $lab $rd $bb $arm
  echo "=== $lab done $(date)"
}

ft minktr full 42; ft minktr cut 42
ft areatr full 42; ft areatr cut 42
echo "=== G3 queue (rows other than area) $(date)"
bash scripts/hpc/run_fm_g3_queue.sh
for S in 43 44; do ft minktr full $S; ft minktr cut $S; done
echo "=== FM_TRUNC_QUEUE_DONE $(date)"
