#!/bin/bash
# DECISIONS §37 (researcher, 2026-10-22: launch without further approval once the baseline is done).
#   1. wait for the vanilla v4_adv baseline;
#   2. step 1b: gradient audit at its checkpoint (same settings as the 15-min audit of 2026-09-23);
#   3. w* = 1e-4 x 16.127 / (hourly Adam share at 1e-4)  (16.127 = 15-min vanilla Adam share at 1e-4);
#   4. the bracket {w*, 3w*, 10w*, w*/3}, full budget, same seed, one run after another on GPU 1.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
BASE_RUN=UNet_Ana_20261021_211732
OUT=eval_results/gradient_audit/2026-10-22_v4adv_vanilla
until grep -q VANILLA_V4ADV_DONE logs/exclude_then_vanilla_hrulj.log; do sleep 300; done
echo "=== baseline done; step 1b audit at $BASE_RUN $(date)"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u tools/gradient_audit.py \
  configs/config_v4_adv.yaml --run vanilla_v4=$BASE_RUN --grad_at vanilla_v4 --losses minkowski --skip_eval \
  --which best --n_batches 512 --batch_size 128 --anneal 0.1 --data_percentage 10 --num_workers 2 \
  --out_dir $OUT 2>&1 | grep -v Warn || { echo "AUDIT FAILED"; exit 1; }
W=$($E/bin/python - <<'PY'
import json, math
g = json.load(open("eval_results/gradient_audit/2026-10-22_v4adv_vanilla/gradients.json"))
m = [x for x in g if x["checkpoint"] == "vanilla_v4"][0]["terms"]["minkowski"]
a = m["adam_share"]
if not (a and math.isfinite(a) and a > 0):
    raise SystemExit("no finite Adam share")
w = 1e-4 * 16.127 / a
print(f"{float(f'{w:.2g}'):.2e}")
PY
) || { echo "W* FAILED"; exit 1; }
echo "=== step 1b: w* = $W (step 1a: 8.5e-05) $(date)"
echo "$W" > $OUT/w_star.txt
WEIGHTS=$($E/bin/python -c "w=float('$W'); print(' '.join(f'{float(f\"{x:.2g}\"):.2e}' for x in (w, 3*w, 10*w, w/3)))")
echo "=== bracket weights (in run order): $WEIGHTS"
for w in $WEIGHTS; do
  echo "=== bracket run w=$w $(date)"
  bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml "$w" 100.0 configs/config_v4_adv.yaml
  echo "=== run w=$w finished $(date); run dir $(ls -t runs/sr_analytical | head -1)"
done
echo "=== BRACKET_DONE $(date)"
