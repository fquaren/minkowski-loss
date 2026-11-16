#!/bin/bash
# Baselines v2, G2 step 1: parity audits at the vanilla checkpoint for wmse, quantile, fss (grid truncated below
# 50.1 mm, as in both R4 arms) and the Hann-windowed spectral loss, then weights.json. Same tool, flags and weight
# rule as node40_r4_audits.sh (target: the Adam share of Minkowski at 6.0e-5), two GPUs in parallel.
#   node40_v2_audits.sh <gpu a> <gpu b>      (run detached with setsid nohup)
set -uo pipefail
cd "$(dirname "$0")/../.."
GA=$1; GB=$2
E=/work/fquareng/.micromamba/envs/dl-stable
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
AUD=$W/eval_results/gradient_audit/2026-11-16_v2_parity
CD=$PWD/logs/r4_configs
export PYTORCH_ALLOC_CONF=expandable_segments:True      # see node40_r4_audits.sh
[ -f $AUD/weights.json ] && { echo "weights.json exists"; cat $AUD/weights.json; exit 0; }
mkdir -p $AUD/trunc $AUD/spectral_hann $CD
$E/bin/python - $CD <<'PY' || { echo "CONFIG FAILED"; exit 1; }
import sys, yaml
cd = sys.argv[1]
c = yaml.safe_load(open("configs/config_v4_adv.yaml"))
c["MINKOWSKI_MAX_THRESHOLD"] = 50.1           # wmse / fss grids, as the training runs of both arms
yaml.safe_dump(c, open(f"{cd}/audit_v2_trunc.yaml", "w"), sort_keys=False)
c = yaml.safe_load(open("configs/config_v4_adv.yaml"))
c["SPECTRAL_WINDOW"] = "hann"
yaml.safe_dump(c, open(f"{cd}/audit_v2_hann.yaml", "w"), sort_keys=False)
PY
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
launch() {   # gpu label config out_dir losses...   -> prints the pid
  local g=$1 lab=$2 cfg=$3 out=$4; shift 4
  local L
  L=$(scripts/hpc/node40_run.sh $g $lab tools/gradient_audit.py $cfg --run vanilla_v4=UNet_Ana_20261021_211732 \
      --grad_at vanilla_v4 --losses "$@" --skip_eval --which best --n_batches 512 --batch_size 128 --anneal 0.1 \
      --data_percentage 10 --num_workers 2 --out_dir $out) || { echo "AUDIT LAUNCH FAILED $lab: $L" >&2; exit 1; }
  echo "$lab log $L" >&2
  cat /tmp/fquareng_node40_gpu$g.pid
}
echo "=== v2 parity audits on node40 GPUs $GA $GB, code $(git rev-parse --short HEAD), $(date)"
PA=$(launch $GA audit_v2_trunc $CD/audit_v2_trunc.yaml $AUD/trunc wmse quantile fss) || exit 1
sleep 30
PB=$(launch $GB audit_v2_hann $CD/audit_v2_hann.yaml $AUD/spectral_hann spectral) || exit 1
waitpid $PA; waitpid $PB
for o in trunc spectral_hann; do [ -f $AUD/$o/gradients.json ] || { echo "AUDIT FAILED $o"; exit 1; }; done
$E/bin/python - $AUD <<'PY' || { echo "WEIGHTS FAILED"; exit 1; }
import json, math, sys
aud = sys.argv[1]
target = 8.833116738640596 * 6.0e-5 / 1e-4          # as node40_r4_audits.sh
w = {}
for sub, names in (("trunc", ["wmse", "quantile", "fss"]), ("spectral_hann", ["spectral"])):
    t = [x for x in json.load(open(f"{aud}/{sub}/gradients.json")) if x["checkpoint"] == "vanilla_v4"][0]["terms"]
    for n in names:
        a, lam = t[n]["adam_share"], t[n]["lambda_used"]
        if not (a and math.isfinite(a) and a > 0):
            raise SystemExit(f"no finite Adam share for {n}")
        key = "spectral_hann" if sub == "spectral_hann" else n
        w[key] = float(f"{lam * target / a:.2g}")
        print(f"{key}: adam_share {a:.4g} at {lam:g} -> w {w[key]:.2e}")
json.dump({"target_adam_share": target, "weights": w}, open(f"{aud}/weights.json", "w"), indent=1)
PY
echo "=== AUDITS_DONE $(date)"
