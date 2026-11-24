#!/bin/bash
source "$(dirname "$0")/../../scripts/hpc/fm_g3_common.sh"
O=$PWD/logs/tab/smoke_out
$P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_evalext.yaml --model backbone --checkpoint $BB_FULL \
  --split val --pot_threshold 50.1 --max_batches 2 --tag det --output_dir $O 2>&1 | grep -v Warn | tail -3
$P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_evalext.yaml --model fm --fm_checkpoint $FMB_FULL \
  --backbone $BB_FULL --ensemble 3 --split val --pot_threshold 50.1 --max_batches 1 --tag fm --output_dir $O 2>&1 | grep -v Warn | tail -3
echo SMOKE_TAB_DONE
