#!/bin/bash
# Step 1 of run_r4_queue.sh (R4 parity audits at the vanilla checkpoint and weights.json), on node40.
# Same tool, flags and weight rule as the queue; only the launcher differs (node40_run.sh guards).
#   node40_r4_audits.sh <gpu 0-3>      (run detached with setsid nohup)
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1
E=/work/fquareng/.micromamba/envs/dl-stable
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
AUD=$W/eval_results/gradient_audit/2026-11-04_r4_parity
CD=$PWD/logs/r4_configs
# The audit runs fp32 with retain_graph at batch 128: its peak sits at the V100's 32 GB and fragmentation tipped it
# over (OOM at batch ~125, 3.9 GB reserved but unallocated, 2026-11-05). The allocator setting changes no numbers;
# a smaller batch would change the gradient-norm statistics the parity rule compares against.
export PYTORCH_ALLOC_CONF=expandable_segments:True
[ -f $AUD/weights.json ] && { echo "weights.json exists"; cat $AUD/weights.json; exit 0; }
mkdir -p $AUD/competing $AUD/mink_trunc $AUD/area_trunc
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
run() {   # label config out_dir losses...
  local lab=$1 cfg=$2 out=$3; shift 3
  local L
  L=$(scripts/hpc/node40_run.sh $GPU $lab tools/gradient_audit.py $cfg --run vanilla_v4=UNet_Ana_20261021_211732 \
      --grad_at vanilla_v4 --losses "$@" --skip_eval --which best --n_batches 512 --batch_size 128 --anneal 0.1 \
      --data_percentage 10 --num_workers 2 --out_dir $out) || { echo "AUDIT LAUNCH FAILED $lab: $L"; exit 1; }
  waitpid $(cat /tmp/fquareng_node40_gpu$GPU.pid)
  grep -v Warn $L | tail -6
  [ -f $out/gradients.json ] || { echo "AUDIT FAILED $lab"; exit 1; }
}
echo "=== R4 parity audits on node40 GPU $GPU, code $(git rev-parse --short HEAD), $(date)"
run audit_mink_trunc $CD/audit_mink_trunc.yaml $AUD/mink_trunc minkowski
run audit_area_trunc $CD/audit_area_trunc.yaml $AUD/area_trunc minkowski
run audit_competing configs/config_v4_adv.yaml $AUD/competing spectral ssim wetarea opticalflow
$E/bin/python - $AUD <<'PY' || { echo "WEIGHTS FAILED"; exit 1; }
import json, math, sys
aud = sys.argv[1]
target = 8.833116738640596 * 6.0e-5 / 1e-4          # Minkowski threshold form at 6.0e-5 (audit 2026-10-22)
w = {}
for sub, names in (("competing", ["spectral", "ssim", "wetarea", "opticalflow"]),
                   ("mink_trunc", ["minkowski"]), ("area_trunc", ["minkowski"])):
    t = [x for x in json.load(open(f"{aud}/{sub}/gradients.json")) if x["checkpoint"] == "vanilla_v4"][0]["terms"]
    for n in names:
        a, lam = t[n]["adam_share"], t[n]["lambda_used"]
        if not (a and math.isfinite(a) and a > 0):
            raise SystemExit(f"no finite Adam share for {n}")
        key = sub if sub in ("mink_trunc", "area_trunc") else n
        w[key] = float(f"{lam * target / a:.2g}")
        print(f"{key}: adam_share {a:.4g} at {lam:g} -> w {w[key]:.2e}")
json.dump({"target_adam_share": target, "weights": w}, open(f"{aud}/weights.json", "w"), indent=1)
PY
echo "=== AUDITS_DONE $(date)"
