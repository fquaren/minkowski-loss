#!/bin/bash
# FM G3, before the queue: parity weight and a 3-step fine-tune smoke per reward, full arm.
# Runs beside run_fm_r4_queue.sh on the same card (minutes each). Writes logs/fm_g3/weights.txt
# ("label weight"), which run_fm_g3_queue.sh reads; it never computes a weight itself.
source "$(dirname "$0")/fm_g3_common.sh"
echo "=== FM G3 parity + smoke, code $(git rev-parse --short HEAD) $(git status --short src | wc -l) uncommitted in src, $(date)"
: > $L/weights.txt
for spec in "${ROWS[@]}"; do
  set -- $spec; lab=$1 mode=$2 cw=$3 form=$4
  mkcfg $L/configs/parity_$lab.yaml $mode 1.0 $cw $form $META/full_train.txt $BB_FULL $SIG_FULL parity
  $P scripts/evaluate/grad_norm_terms.py $L/configs/parity_$lab.yaml --fm_checkpoint $FMB_FULL \
    --backbone $BB_FULL --n_batches 4 > $L/parity_$lab.log 2>&1 || { echo "PARITY FAILED $lab"; continue; }
  w=$(grep "^mean |g|" $L/parity_$lab.log | awk '{printf "%.2g", $8 / $4}')   # |g_fm| / |g_reward|
  [ -z "$w" ] && { echo "PARITY FAILED $lab"; continue; }
  echo "$lab $w" >> $L/weights.txt
  echo "parity $lab: w $w | $(grep -E '^\[grad\] mode|share' $L/parity_$lab.log | tr '\n' ' ')"
  # smoke: the real trainer path with the parity weight, 3 steps, short validation
  mkcfg $L/configs/smoke_$lab.yaml $mode $w $cw $form $META/full_train.txt $BB_FULL $SIG_FULL smokeG3_$lab \
    REWARD_EPOCHS=1 REWARD_VAL_BATCHES=2 REWARD_GUARD_BATCHES=2
  $P scripts/train/train_fm_minkowski.py $L/configs/smoke_$lab.yaml --fm_checkpoint $FMB_FULL \
    --backbone $BB_FULL --seed 42 --max_steps 3 > $L/smoke_$lab.log 2>&1 \
    && echo "smoke $lab OK: $(grep -aE 'sigma_r|reward mode=|channel weights|reward:|Epoch 1' $L/smoke_$lab.log | cut -c25- | tr '\n' ' ')" \
    || echo "SMOKE FAILED $lab: $(grep -aE 'Error|Traceback' $L/smoke_$lab.log | tail -2 | tr '\n' ' ')"
done
echo "=== FM_G3_PARITY_DONE $(date)"
