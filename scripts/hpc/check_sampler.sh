#!/bin/bash
# =====================================================================
# Sampler-fidelity check -- run this BEFORE the sweep.
#
# The reward is computed on generated samples during training, so a cheap sampler in the
# training loop shapes fields that are not the ones you deploy. Revision 1 of the sweep
# showed this is not a small effect: on the same clean checkpoint, euler-8/M=8 gives a
# median peak ratio of 0.44 against 0.85 for heun-16/M=16, and FSS at 89 mm/h of 0.00
# against 0.289. Optimising a structural reward against fields whose peaks are half the
# deployed height is optimising the wrong distribution.
#
# This script evaluates ONE checkpoint (the clean model) under three samplers on an
# identical patch set, so you can see how much fidelity each buys and choose the cheapest
# one that preserves the extreme regime. Compare peak_ratio_median, fss_thr89_win5 and
# gpd_xi_pred across the three: pick the cheapest whose peak ratio is within a few percent
# of heun-16, then set TRAIN_SAMPLER / TRAIN_ODE_STEPS in sweep_reward.sh accordingly.
#
# Cost: three evaluations at 32 batches. The heun-16 arm dominates (~1 h); the euler arms
# are 2-4x cheaper.
#
# Usage: bash scripts/hpc/check_sampler.sh
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then echo "no python interpreter found; set PYTHON=..." >&2; exit 1; fi
SWEEP_UTIL="${PROJECT_ROOT}/src/sweep_util.py"

BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/config.yaml}"
FM_CKPT="${FM_CKPT:-${PROJECT_ROOT}/runs/sr_flow_matching/flow_matching_20260716_071518/fm_best.pth}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/sweeps/sampler_check_$(date +%Y%m%d_%H%M%S)}"
RESULTS_CSV="${OUT_DIR}/results.csv"
GPU="${GPU:-1}"
# Shared node: GPU 1 only (EXPERIMENTS.md "Compute node"). Refuse anything else.
[[ "$GPU" == "1" ]] || { echo "GPU=$GPU is not allowed on this node: GPU 1 only." >&2; exit 1; }
M="${M:-16}"
BATCHES="${BATCHES:-32}"
POT_THRESHOLD="${POT_THRESHOLD:-31}"

# "label|sampler|ode_steps" -- heun-16 is the reference the others are judged against
ARMS=(
  "heun16|heun|16"
  "euler16|euler|16"
  "euler8|euler|8"
)

mkdir -p "${OUT_DIR}/logs"
echo "sampler-fidelity check -> $OUT_DIR"
echo "checkpoint: $FM_CKPT"
echo "protocol  : M=${M}, ${BATCHES} batches, POT u=${POT_THRESHOLD} mm/h (identical across arms)"
echo

for arm in "${ARMS[@]}"; do
  IFS='|' read -r label sampler steps <<< "$arm"
  if grep -q "^${label}," "$RESULTS_CSV" 2>/dev/null; then echo "[skip] ${label}"; continue; fi
  echo "[arm] ${label}: ${sampler}, ${steps} steps"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  "$PYTHON" "${PROJECT_ROOT}/scripts/evaluate/eval_extremes.py" "$BASE_CONFIG" \
      --model fm --fm_checkpoint "$FM_CKPT" \
      --ensemble "$M" --steps "$steps" --sampler "$sampler" \
      --pot_threshold "$POT_THRESHOLD" --max_batches "$BATCHES" --split test \
      --tag "$label" --output_dir "${OUT_DIR}/eval" \
      > "${OUT_DIR}/logs/${label}.log" 2>&1 \
      || { echo "  ! failed, see logs/${label}.log"; continue; }
  "$PYTHON" "$SWEEP_UTIL" collect \
      --summary "${OUT_DIR}/eval/${label}/extremes_summary.yaml" \
      --trial "$label" --overrides "${sampler}-${steps}" --checkpoint clean --csv "$RESULTS_CSV"
done

echo
echo "results: $RESULTS_CSV"
echo "Compare peak_ratio_median and fss_thr89_win5 against the heun16 row. Use the cheapest"
echo "arm that stays close to it as TRAIN_SAMPLER/TRAIN_ODE_STEPS; if none does, train with"
echo "heun-16 and halve TRAIN_STEPS instead."
if [ -f "$RESULTS_CSV" ]; then
  if command -v column > /dev/null 2>&1; then column -s, -t < "$RESULTS_CSV" | cut -c1-200
  else cat "$RESULTS_CSV"; fi
fi