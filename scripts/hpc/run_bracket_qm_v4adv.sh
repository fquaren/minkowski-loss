#!/bin/bash
# DECISIONS §41, step 3: U-Net + Minkowski in the quantile_mass form, weight by the §37 rule.
#   1. wait for the FM fine-tune chain (run_qm_finetunes_v4adv.sh) to end; stop if it failed;
#   2. step 1b: gradient audit of the qm form at the hourly vanilla checkpoint (same settings as §37);
#   3. w* = 1e-4 x 16.127 / (hourly Adam share at 1e-4), 2 significant figures;
#   4. bracket {w*/3, w*, 3w*, 10w*}, full budget, same seed, one run after another on GPU 1.
#      w*/3 first: it won the threshold-form bracket (§37).
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CFG=configs/config_v4_adv_qm.yaml
BASE_RUN=UNet_Ana_20261021_211732
OUT=eval_results/gradient_audit/2026-10-30_v4adv_qm_vanilla
until grep -q -E "QM_FINETUNES_DONE|FAILED" logs/run_qm_finetunes_v4adv.log; do sleep 300; done
grep -q FAILED logs/run_qm_finetunes_v4adv.log && { echo "FM chain failed; bracket not started $(date)"; exit 1; }
echo "=== FM chain done; step 1b audit (quantile_mass) at $BASE_RUN $(date)"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u tools/gradient_audit.py \
  $CFG --run vanilla_v4=$BASE_RUN --grad_at vanilla_v4 --losses minkowski --skip_eval \
  --which best --n_batches 512 --batch_size 128 --anneal 0.1 --data_percentage 10 --num_workers 2 \
  --out_dir $OUT 2>&1 | grep -v Warn || { echo "AUDIT FAILED"; exit 1; }
W=$($E/bin/python - <<'PY'
import json, math
g = json.load(open("eval_results/gradient_audit/2026-10-30_v4adv_qm_vanilla/gradients.json"))
m = [x for x in g if x["checkpoint"] == "vanilla_v4"][0]["terms"]["minkowski"]
a = m["adam_share"]
if not (a and math.isfinite(a) and a > 0):
    raise SystemExit("no finite Adam share")
print(f"{float(f'{1e-4 * 16.127 / a:.2g}'):.2e}")
PY
) || { echo "W* FAILED"; exit 1; }
echo "=== step 1b: w*_qm = $W (threshold form: 1.8e-04; init ratio predicts ~2.4e-04) $(date)"
echo "$W" > $OUT/w_star.txt
WEIGHTS=$($E/bin/python -c "w=float('$W'); print(' '.join(f'{float(f\"{x:.2g}\"):.2e}' for x in (w/3, w, 3*w, 10*w)))")
echo "=== bracket weights (in run order): $WEIGHTS"
for w in $WEIGHTS; do
  echo "=== bracket run w=$w $(date)"
  bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml "$w" 100.0 $CFG
  rd=$(ls -td runs/sr_analytical/* | head -1)
  echo "=== run w=$w finished $(date); run dir $rd"
done
echo "=== QM_BRACKET_DONE $(date)"
