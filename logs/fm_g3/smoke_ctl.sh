#!/bin/bash
# Control for the G3 smoke aborts: the same 3-step smoke with w_reward = 0.
source /work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4/scripts/hpc/fm_g3_common.sh
mkcfg $L/configs/smoke_ctl.yaml pixel 0 1,1,1 quantile_mass $META/full_train.txt $BB_FULL $SIG_FULL smokeG3_ctl \
  REWARD_EPOCHS=1 REWARD_VAL_BATCHES=2 REWARD_GUARD_BATCHES=2
$P scripts/train/train_fm_minkowski.py $L/configs/smoke_ctl.yaml --fm_checkpoint $FMB_FULL \
  --backbone $BB_FULL --seed 42 --max_steps 3 > $L/smoke_ctl.log 2>&1
grep -aE 'guard baseline|Epoch 1|ABORT' $L/smoke_ctl.log | cut -c25-220
echo SMOKE_CTL_DONE
