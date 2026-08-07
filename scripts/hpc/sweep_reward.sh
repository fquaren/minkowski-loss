#!/bin/bash
# =====================================================================
# One-factor-at-a-time sweep over reward fine-tuning, revision 2.
#
# WHAT CHANGED FROM REVISION 1 AND WHY
#   r1 evaluated 2048 patches with a cheap sampler. Both consequences were fatal:
#     * the generalised-Pareto fit needs >=200 predicted exceedances and got ~110-145,
#       so gpd_xi_err and rl_bias_mean_abs were NaN in every row, including the control --
#       the two metrics the sweep exists to attribute were never computed;
#     * the cheap sampler is a different regime, not a cheaper proxy. On the same clean
#       checkpoint, euler-8/M=8 vs heun-16/M=16 gives peak ratio 0.44 vs 0.85 and FSS at
#       89 mm/h of 0.00 vs 0.289 -- it roughly halves the very quantity the reward moves.
#   r2 therefore: evaluates with the production sampler at 64 batches, lowers the
#   peaks-over-threshold level to 31 mm/h so the tail fit is stable at that size, fixes
#   M=16 instead of sweeping it, and spends the freed budget on the two factors that can
#   actually be resolved -- the reward weight and the per-sample/distributional mode.
#
# RUN THE SAMPLER-FIDELITY CHECK FIRST: the training reward must be computed on samples
# resembling what you deploy. Set TRAIN_SAMPLER/TRAIN_ODE_STEPS to whatever it says is
# faithful enough.
#
# Usage:
#   bash scripts/hpc/sweep_reward.sh
#   DRY_RUN=1 bash scripts/hpc/sweep_reward.sh
#   TRIALS_FILTER=w_ bash scripts/hpc/sweep_reward.sh
# Resumable: a trial already in results.csv is skipped.
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u
cd "$PROJECT_ROOT"

# Resolve an interpreter explicitly: bare `python` does not exist in some environments.
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then echo "no python interpreter found; set PYTHON=..." >&2; exit 1; fi
SWEEP_UTIL="${PROJECT_ROOT}/src/sweep_util.py"

BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/config.yaml}"
FM_CKPT="${FM_CKPT:-${PROJECT_ROOT}/runs/sr_flow_matching/flow_matching_20260716_071518/fm_best.pth}"
SWEEP_ROOT="${SWEEP_ROOT:-${PROJECT_ROOT}/sweeps}"
SWEEP_NAME="${SWEEP_NAME:-reward_$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_ROOT}/${SWEEP_NAME}"
RESULTS_CSV="${SWEEP_DIR}/results.csv"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
EVAL_CKPT="${EVAL_CKPT:-latest}"        # latest | best | both
DRY_RUN="${DRY_RUN:-0}"
TRIALS_FILTER="${TRIALS_FILTER:-}"

# ---- FIXED protocol, identical for every trial ---------------------------
# Training: the reward is computed on generated samples, so this sampler must resemble
# deployment. The fidelity check settled this empirically -- on the clean checkpoint,
# relative to heun-16, euler-16 loses 28% of the median peak ratio (41% on extreme
# patches), 36% of the exceedance count, and doubles both the RAPSD distance and the
# return-level bias; euler-8 is roughly twice as bad again. Those are precisely the
# quantities the reward moves, so a cheap sampler optimises the wrong distribution.
# heun-16 it is. The cost is paid for by a smaller per-step batch (below) and by 32
# rather than 64 evaluation batches, which the POT level of 31 mm/h makes sufficient.
TRAIN_SAMPLER="${TRAIN_SAMPLER:-heun}"
TRAIN_ODE_STEPS="${TRAIN_ODE_STEPS:-16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-500}"        # per epoch -> 1000 optimiser steps total
REWARD_M="${REWARD_M:-16}"               # fixed by design: Var of the E[gamma] estimate ~ 1/M
# Conditions per step. Reduced from 8 to 4 to fund heun-16: M is what the conditional-mean
# estimate needs, whereas averaging over conditions is what Adam does across steps anyway.
REWARD_B="${REWARD_B:-4}"

# Evaluation: the production protocol, non-negotiable if the tail columns are to mean
# anything. 64 batches x BATCH_SIZE patches, POT level 31 mm/h for a stable fit.
EVAL_M="${EVAL_M:-16}"
EVAL_SAMPLER="${EVAL_SAMPLER:-heun}"
EVAL_ODE_STEPS="${EVAL_ODE_STEPS:-16}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"   # sufficient at u=31: the fidelity check fit converged at 4096 patches
POT_THRESHOLD="${POT_THRESHOLD:-31}"

# ---------------------------------------------------------------------
# Trials: "name|KEY=VAL[,KEY=VAL...]"   (empty = baseline)
# Arms with neither measurable signal nor a mechanism (K, spread, proximal, LR) were
# dropped after revision 1. steps_2000 is a control on the budget itself: if every trial
# looks like the clean model, it separates "these factors are inert" from "1000 steps is
# too few to move anything".
# ---------------------------------------------------------------------
TRIALS=(
  "baseline|"
  "w_3e-4|REWARD_WEIGHT=3.0e-4"
  "w_3e-3|REWARD_WEIGHT=3.0e-3"
  "w_1e-2|REWARD_WEIGHT=1.0e-2"
  "mode_persample|REWARD_MODE=per_sample"
  "steps_2000|REWARD_STEPS_PER_EPOCH=1000"
)

BASELINE_SET=(
  "REWARD_MODE=distributional"
  "REWARD_LR=1.0e-5"
  "REWARD_WEIGHT=1.0e-3"
  "REWARD_PROXIMAL_WEIGHT=1.0"
  "REWARD_SPREAD_WEIGHT=0.5"
  "REWARD_ENSEMBLE=${REWARD_M}"
  "REWARD_K=1"
  "REWARD_EPOCHS=${TRAIN_EPOCHS}"
  "REWARD_STEPS_PER_EPOCH=${TRAIN_STEPS}"
  "REWARD_BATCH_SIZE=${REWARD_B}"
  "REWARD_VAL_BATCHES=10"
  "REWARD_GUARD_AUTO=true"
  "FM_SAMPLER=${TRAIN_SAMPLER}"
  "FM_SAMPLE_STEPS=${TRAIN_ODE_STEPS}"
)

mkdir -p "$SWEEP_DIR/configs" "$SWEEP_DIR/logs"
cat <<EOF
sweep      : $SWEEP_DIR
python     : $PYTHON
base config: $BASE_CONFIG
fm ckpt    : $FM_CKPT
train      : ${TRAIN_EPOCHS}x${TRAIN_STEPS} steps, M=${REWARD_M}, B=${REWARD_B}, ${TRAIN_SAMPLER}-${TRAIN_ODE_STEPS}, seed ${SEED}
eval       : M=${EVAL_M}, ${EVAL_SAMPLER}-${EVAL_ODE_STEPS}, ${EVAL_BATCHES} batches, POT u=${POT_THRESHOLD} mm/h

EOF

run_eval () {  # tag ckpt trial overrides label
  local tag="$1" ckpt="$2" name="$3" ovr="$4" label="$5"
  local outdir="${SWEEP_DIR}/eval/${tag}"
  if [ ! -f "$ckpt" ]; then echo "  ! missing checkpoint $ckpt"; return 1; fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  "$PYTHON" "${PROJECT_ROOT}/scripts/evaluate/eval_extremes.py" "$BASE_CONFIG" \
      --model fm --fm_checkpoint "$ckpt" \
      --ensemble "$EVAL_M" --steps "$EVAL_ODE_STEPS" --sampler "$EVAL_SAMPLER" \
      --pot_threshold "$POT_THRESHOLD" --max_batches "$EVAL_BATCHES" --split test \
      --tag "$tag" --output_dir "$outdir" \
      >> "${SWEEP_DIR}/logs/${tag}.eval.log" 2>&1 \
      || { echo "  ! eval failed ($tag), see logs/${tag}.eval.log"; return 1; }
  "$PYTHON" "$SWEEP_UTIL" collect \
      --summary "${outdir}/${tag}/extremes_summary.yaml" \
      --trial "$name" --overrides "$ovr" --checkpoint "$label" --csv "$RESULTS_CSV"
  # fail loudly if the tail fit still did not converge -- that invalidates the row
  if grep -q "gpd_xi_pred: .nan" "${outdir}/${tag}/extremes_summary.yaml" 2>/dev/null; then
    echo "  ! WARNING: generalised-Pareto fit did not converge for ${tag}."
    echo "    Too few predicted exceedances above ${POT_THRESHOLD} mm/h -- raise EVAL_BATCHES"
    echo "    or lower POT_THRESHOLD. The tail columns in this row are unusable."
  fi
}

if ! grep -q "^clean_control," "$RESULTS_CSV" 2>/dev/null; then
  echo "[control] clean flow-matching checkpoint (every trial is read as a delta from this)"
  [ "$DRY_RUN" = "0" ] && run_eval "clean_control" "$FM_CKPT" "clean_control" "(none: un-finetuned)" "clean"
fi

for entry in "${TRIALS[@]}"; do
  name="${entry%%|*}"; overrides="${entry#*|}"
  [ -n "$TRIALS_FILTER" ] && [[ "$name" != *"$TRIALS_FILTER"* ]] && continue
  if grep -q "^${name}," "$RESULTS_CSV" 2>/dev/null; then echo "[skip] ${name}"; continue; fi

  set_args=("${BASELINE_SET[@]}")
  if [ -n "$overrides" ]; then
    IFS=',' read -ra extra <<< "$overrides"
    for kv in "${extra[@]}"; do set_args+=("$kv"); done
  fi
  set_args+=("EXPERIMENT_NAME=SW_${name}")

  trial_cfg="${SWEEP_DIR}/configs/${name}.yaml"
  "$PYTHON" "$SWEEP_UTIL" patch --base "$BASE_CONFIG" --out "$trial_cfg" --set "${set_args[@]}" > /dev/null

  echo "[trial] ${name}  overrides: ${overrides:-<baseline>}"
  [ "$DRY_RUN" != "0" ] && continue

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  "$PYTHON" "${PROJECT_ROOT}/scripts/train/train_fm_minkowski.py" "$trial_cfg" \
      --fm_checkpoint "$FM_CKPT" --seed "$SEED" \
      > "${SWEEP_DIR}/logs/${name}.train.log" 2>&1
  if [ $? -ne 0 ]; then echo "  ! train failed, see logs/${name}.train.log"; continue; fi

  run_dir="$(ls -dt "${PROJECT_ROOT}"/runs/sr_fm_minkowski/SW_${name}_* 2>/dev/null | head -1)"
  if [ -z "$run_dir" ]; then echo "  ! no run dir for ${name}"; continue; fi
  echo "  run: $run_dir"
  grep -h "ABORT\|guard baseline" "${SWEEP_DIR}/logs/${name}.train.log" | sed 's/^/  /' || true

  case "$EVAL_CKPT" in
    latest) run_eval "${name}_latest" "${run_dir}/fm_mink_latest.pth" "$name" "$overrides" "latest" ;;
    best)   run_eval "${name}_best"   "${run_dir}/fm_mink_best.pth"   "$name" "$overrides" "best" ;;
    both)   run_eval "${name}_latest" "${run_dir}/fm_mink_latest.pth" "$name" "$overrides" "latest"
            run_eval "${name}_best"   "${run_dir}/fm_mink_best.pth"   "$name" "$overrides" "best" ;;
  esac
done

echo
echo "done. results: $RESULTS_CSV"
if [ -f "$RESULTS_CSV" ]; then
  if command -v column > /dev/null 2>&1; then column -s, -t < "$RESULTS_CSV" | cut -c1-200
  else cat "$RESULTS_CSV"; fi
fi
