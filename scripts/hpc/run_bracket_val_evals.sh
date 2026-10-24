#!/bin/bash
# Bracket selection on VALIDATION (DECISIONS §37, researcher 2026-10-22/24): last checkpoint of each run,
# POT u = 50.1 mm, FSS at 20.2 / 31.7 / 50.1 mm (the top three loss levels). Vanilla and the finished runs
# now; each later run as soon as the bracket chain reports it finished. GPU 1, 2 loader workers (training runs
# beside it). Outputs: eval_results/backbone/v4adv_val/<label>, eval_results/extremes/v4adv_val/.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CFG=/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 nice -n 5 $E/bin/python -u"
R=runs/sr_analytical
ev() {   # label run_dir
  local lab=$1 run=$2
  echo "=== eval $lab ($run) $(date)"
  [ -d eval_results/backbone/v4adv_val/$lab ] && { echo "exists, skipping $lab"; return; }
  $P scripts/evaluate/eval_backbone.py $CFG --checkpoint $R/$run/unet_latest.pth --split val \
      --output_dir eval_results/backbone/v4adv_val/$lab 2>&1 | grep -v Warn | tail -4 || echo "BACKBONE EVAL FAILED $lab"
  $P scripts/evaluate/eval_extremes.py $CFG --model backbone --checkpoint $R/$run/unet_latest.pth --split val \
      --pot_threshold 50.1 --tag $lab --output_dir eval_results/extremes/v4adv_val 2>&1 | grep -v Warn | tail -4 \
      || echo "EXTREMES EVAL FAILED $lab"
}
ev vanilla UNet_Ana_20261021_211732
ev mink_1.80e-04 UNet_Ana_20261022_160540
ev mink_5.40e-04 UNet_Ana_20261023_121956
for w in 1.80e-03 6.00e-05; do
  until grep -q "run w=$w finished" logs/bracket_v4adv.log; do sleep 300; done
  run=$(grep "run w=$w finished" logs/bracket_v4adv.log | sed 's/.*run dir //')
  ev mink_$w "$run"
done
echo "=== VAL_EVALS_DONE $(date)"
