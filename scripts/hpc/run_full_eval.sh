#!/bin/bash
# =====================================================================
# Full evaluation sweep + figures, end to end.
#
# Evaluates every trained model under ONE protocol and produces the comparison figures.
# Safe to re-run: anything already evaluated is skipped, so an interrupted run resumes.
#
#   bash scripts/hpc/run_full_eval.sh                 # everything
#   DRY_RUN=1 bash scripts/hpc/run_full_eval.sh       # print the plan, run nothing
#   STAGE=backbones bash scripts/hpc/run_full_eval.sh # one stage only
#   STAGE=figures   bash scripts/hpc/run_full_eval.sh # re-plot from existing evals
#
# Stages: backbones | bicubic | fm | figures | all (default)
#
# WHY THE ARCHIVE STEP: the evaluation scripts were patched to persist `gamma_target` and
# `pot_threshold`, which the gamma-curve and return-level figures need. Summaries written
# before that patch lack those arrays, and because the runner skips completed evaluations,
# a stale directory silently produces empty figures. The script therefore moves an
# unpatched eval_results aside once, rather than leaving you to debug blank plots.
#
# EVERY tail column is computed at the same peaks-over-threshold level. The GPD shape error
# and the return-level bias are fitted against an observed reference that depends on that
# level, so rows at different levels are not comparable -- this is the single most common
# way the table has gone wrong.
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh" 2>/dev/null
set -u
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
[ -z "$PYTHON" ] && { echo "no python interpreter; set PYTHON=..." >&2; exit 1; }

GPU="${GPU:-1}"
POT="${POT:-31}"                      # one level for every row
SPLIT="${SPLIT:-test}"
ENSEMBLE="${ENSEMBLE:-16}"
FM_BATCHES="${FM_BATCHES:-32}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"
FIGDIR="${FIGDIR:-figures/committee}"
EXT="${PROJECT_ROOT}/eval_results/extremes"
BB="${PROJECT_ROOT}/eval_results/backbone"
LOGS="${PROJECT_ROOT}/logs"
mkdir -p "$LOGS"

CUDA_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU}"

# ---- model registry -------------------------------------------------------------
# label|run-directory   (under runs/sr_analytical/)
BACKBONES=(
  "vanilla|UNet_Ana_20260623_144958"
  "minkowski|UNet_Ana_20260731_100128"
  "spectral_v2|UNet_Ana_20260829_033325"
  "ssim_v2|UNet_Ana_20260824_155753"
  "wetarea_v2|UNet_Ana_20260826_041128"
  "opticalflow_v2|UNet_Ana_20260827_151319"
)

VANILLA_BB="runs/sr_analytical/UNet_Ana_20260623_144958/unet_best.pth"
MINK_BB="runs/sr_analytical/UNet_Ana_20260731_100128/unet_best.pth"

# label|config|fm-checkpoint|backbone
# The config matters: it pins FM_RESIDUAL_SCALE. residual_stats.json currently holds the
# Minkowski-backbone value, so a vanilla-backbone run that falls back to it is mis-scaled.
FM_MODELS=(
  "fm_clean|configs/energy.yaml|runs/sr_flow_matching/flow_matching_20260716_071518/fm_best.pth|${VANILLA_BB}"
  "fm_energy|configs/energy.yaml|runs/sr_fm_minkowski/Energy_20260811_171310/fm_mink_latest.pth|${VANILLA_BB}"
  "fm_reward|configs/energy.yaml|runs/sr_fm_minkowski/Extremes_20260804_170256/fm_mink_latest.pth|${VANILLA_BB}"
  "fm_mink_backbone|configs/eval_fm_mink_backbone.yaml|runs/sr_flow_matching/Extremes_20260807_111521/fm_best.pth|${MINK_BB}"
)

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
run() {  # run "<description>" "<command>"
  if [ "$DRY_RUN" != "0" ]; then echo "    [dry-run] $2"; return 0; fi
  eval "$2"
}

# ---------------------------------------------------------------------
# Stage 0: archive an eval_results that predates the gamma_target patch
# ---------------------------------------------------------------------
stage_archive() {
  [ -d eval_results ] || return 0
  local stale=0
  # any extremes summary without pot_threshold_mmph was written before the patch
  while IFS= read -r f; do
    grep -q "pot_threshold_mmph" "$f" || stale=1
  done < <(find eval_results -name "extremes_summary.yaml" 2>/dev/null)
  if [ "$stale" = "1" ]; then
    local dest="eval_results_pre_patch_$(date +%Y%m%d_%H%M%S)"
    say "Archiving pre-patch eval_results -> ${dest}"
    echo "    (summaries lacking pot_threshold_mmph cannot produce the gamma / return-level figures)"
    run "archive" "mv eval_results '${dest}'"
  fi
}

# ---------------------------------------------------------------------
# Stage 1: deterministic backbones (full test set)
# ---------------------------------------------------------------------
stage_backbones() {
  say "Stage 1/4 — deterministic backbones (full test set, POT u=${POT})"
  for entry in "${BACKBONES[@]}"; do
    local label="${entry%%|*}" run_dir="${entry#*|}"
    local ckpt="runs/sr_analytical/${run_dir}/unet_best.pth"
    if [ ! -f "$ckpt" ]; then echo "  [skip] ${label}: no checkpoint at ${ckpt}"; continue; fi

    if [ -f "${EXT}/backbone_${label}/extremes_summary.yaml" ]; then
      echo "  [skip] ${label} extremes (done)"
    else
      echo "  [run ] ${label} extremes"
      run "$label" "${CUDA_ENV} '$PYTHON' scripts/evaluate/eval_extremes.py config.yaml \
          --model backbone --checkpoint '${ckpt}' --pot_threshold ${POT} --split ${SPLIT} \
          --tag 'backbone_${label}' --output_dir '${EXT}' \
          > '${LOGS}/eval_extremes_${label}.log' 2>&1" \
        || echo "    ! failed, see logs/eval_extremes_${label}.log"
    fi

    if [ -f "${BB}/${label}/backbone_summary.yaml" ]; then
      echo "  [skip] ${label} backbone (done)"
    else
      echo "  [run ] ${label} backbone"
      run "$label" "${CUDA_ENV} '$PYTHON' scripts/evaluate/eval_backbone.py config.yaml \
          --checkpoint '${ckpt}' --split ${SPLIT} --output_dir '${BB}/${label}' \
          > '${LOGS}/eval_backbone_${label}.log' 2>&1" \
        || echo "    ! failed, see logs/eval_backbone_${label}.log"
    fi
  done
}

# ---------------------------------------------------------------------
# Stage 2: bicubic reference
# ---------------------------------------------------------------------
stage_bicubic() {
  say "Stage 2/4 — bicubic reference"
  if [ -f "${EXT}/bicubic/extremes_summary.yaml" ]; then
    echo "  [skip] bicubic (done)"; return 0
  fi
  echo "  [run ] bicubic"
  run "bicubic" "${CUDA_ENV} '$PYTHON' scripts/evaluate/eval_extremes.py config.yaml \
      --model bicubic --pot_threshold ${POT} --split ${SPLIT} \
      --tag bicubic --output_dir '${EXT}' > '${LOGS}/eval_extremes_bicubic.log' 2>&1" \
    || echo "    ! failed, see logs/eval_extremes_bicubic.log"
}

# ---------------------------------------------------------------------
# Stage 3: flow-matching models
# ---------------------------------------------------------------------
stage_fm() {
  say "Stage 3/4 — flow matching (M=${ENSEMBLE}, heun-16, ${FM_BATCHES} batches)"
  for entry in "${FM_MODELS[@]}"; do
    IFS='|' read -r label cfg ckpt bbone <<< "$entry"
    if [ ! -f "$ckpt" ]; then echo "  [skip] ${label}: no checkpoint at ${ckpt}"; continue; fi
    if [ ! -f "$cfg" ]; then echo "  [skip] ${label}: no config at ${cfg}"; continue; fi
    if [ -f "${EXT}/${label}/extremes_summary.yaml" ]; then
      echo "  [skip] ${label} (done)"; continue
    fi
    echo "  [run ] ${label}  (config ${cfg})"
    run "$label" "${CUDA_ENV} '$PYTHON' scripts/evaluate/eval_extremes.py '${cfg}' \
        --model fm --fm_checkpoint '${ckpt}' --backbone '${bbone}' \
        --ensemble ${ENSEMBLE} --steps 16 --sampler heun \
        --pot_threshold ${POT} --max_batches ${FM_BATCHES} --split ${SPLIT} \
        --tag '${label}' --output_dir '${EXT}' > '${LOGS}/eval_extremes_${label}.log' 2>&1" \
      || echo "    ! failed, see logs/eval_extremes_${label}.log"

    # the residual scale must come from the config, never from residual_stats.json
    if [ "$DRY_RUN" = "0" ] && [ -f "${LOGS}/eval_extremes_${label}.log" ]; then
      local line
      line="$(grep -m1 'residual scale sigma_r' "${LOGS}/eval_extremes_${label}.log" || true)"
      echo "         ${line:-(no sigma_r line found)}"
      case "$line" in
        *residual_stats.json*|*DEFAULT*)
          echo "    !! WARNING: ${label} fell back to residual_stats.json. That file holds the"
          echo "       Minkowski-backbone value, so a vanilla-backbone run is mis-scaled."
          echo "       An empty or missing stats file instead yields DEFAULT 1.0, which is the"
          echo "       catastrophic case -- it once produced a 53x reconstruction error."
          echo "       Pin FM_RESIDUAL_SCALE in ${cfg} and re-run this model." ;;
      esac
    fi
  done
}

# ---------------------------------------------------------------------
# Stage 4: figures
# ---------------------------------------------------------------------
_model_args() {  # $@ = tag:label pairs, emits --model "label:dir[+backbone_dir]" for those that exist
  local args=""
  for pair in "$@"; do
    local tag="${pair%%=*}" label="${pair#*=}"
    [ -d "${EXT}/${tag}" ] || continue
    local spec="${EXT}/${tag}"
    # gamma_hat / gamma_target are written only by eval_backbone.py, into ${BB}/<label>.
    # Merge that directory in where it exists so the Minkowski figures are drawn rather
    # than skipped; the extremes directory stays first, so it keeps every shared key.
    local bb="${BB}/${tag#backbone_}"
    [ -d "$bb" ] && spec="${spec}+${bb}"
    args="${args} --model \"${label}:${spec}\""
  done
  echo "$args"
}

stage_figures() {
  say "Stage 4/4 — figures"

  # Study 1: does a structural loss help the deterministic backbone?
  local s1; s1="$(_model_args \
      "bicubic=Bicubic" \
      "backbone_vanilla=MSE" \
      "backbone_minkowski=+ Minkowski" \
      "backbone_spectral_v2=+ spectral" \
      "backbone_ssim_v2=+ SSIM" \
      "backbone_wetarea_v2=+ wet area" \
      "backbone_opticalflow_v2=+ optical flow")"
  if [ -n "$s1" ]; then
    echo "  [run ] study1 figures"
    run "s1" "'$PYTHON' scripts/evaluate/make_plots.py --out '${FIGDIR}/study1' ${s1}"
  fi

  # Study 2: does it propagate through the generative stage?
  local s2; s2="$(_model_args \
      "backbone_vanilla=MSE backbone" \
      "backbone_minkowski=Minkowski backbone" \
      "fm_clean=FM clean" \
      "fm_energy=FM + energy" \
      "fm_reward=FM + reward" \
      "fm_mink_backbone=FM on Mink bb")"
  if [ -n "$s2" ]; then
    echo "  [run ] study2 figures"
    run "s2" "'$PYTHON' scripts/evaluate/make_plots.py --out '${FIGDIR}/study2' ${s2}"
  fi

  # Everything on one set, for the appendix
  local all; all="$(_model_args \
      "bicubic=Bicubic" "backbone_vanilla=MSE" "backbone_minkowski=+ Minkowski" \
      "backbone_spectral_v2=+ spectral" "backbone_ssim_v2=+ SSIM" \
      "backbone_wetarea_v2=+ wet area" "backbone_opticalflow_v2=+ optical flow" \
      "fm_clean=FM clean" "fm_energy=FM + energy" "fm_mink_backbone=FM on Mink bb")"
  if [ -n "$all" ]; then
    echo "  [run ] all-model figures"
    run "all" "'$PYTHON' scripts/evaluate/make_plots.py --out '${FIGDIR}/all' ${all}"
  fi
}

# ---------------------------------------------------------------------
# Summary table + the two checks that decide whether the numbers mean anything
# ---------------------------------------------------------------------
stage_summary() {
  [ "$DRY_RUN" != "0" ] && return 0
  say "Summary"
  "$PYTHON" - <<'PY'
import glob, os, yaml

rows = []
for f in sorted(glob.glob("eval_results/extremes/*/extremes_summary.yaml")):
    rows.append((os.path.basename(os.path.dirname(f)), yaml.safe_load(open(f))))
if not rows:
    print("  no evaluations found"); raise SystemExit

# --- guard 1: every row must share the peaks-over-threshold level -----------------
obs = {r[1].get("gpd_xi_obs") for r in rows if r[1].get("gpd_xi_obs") is not None}
if len(obs) > 1:
    print("!! rows were fitted at DIFFERENT peaks-over-threshold levels "
          f"(gpd_xi_obs takes {len(obs)} values).")
    print("   The shape-error and return-level columns are not comparable across them.\n")

# --- guard 2: is each auxiliary loss actually active? ----------------------------
mae = {}
for f in glob.glob("eval_results/backbone/*/backbone_summary.yaml"):
    mae[os.path.basename(os.path.dirname(f))] = yaml.safe_load(open(f)).get("mae_overall")
base = mae.get("vanilla")
if base:
    print("activity check — an auxiliary loss costing < ~1% MAE has not entered the")
    print("objective, and its structural columns say nothing. Minkowski pays +3.8%.")
    for lab in sorted(mae):
        m = mae[lab]
        if m is None: continue
        d = (m / base - 1) * 100
        flag = "INERT" if d < 1.0 else ("active" if d < 8.0 else "large — check images")
        print(f"   {lab:<18s} MAE {m:.5f}  {d:+5.1f}%   {flag}")
    print()

hdr = ["model", "RMSE_ext", "FSS@89", "RAPSD", "Mink", "exc.ratio", "RL bias", "aniso", "peak"]
keys = ["rmse_extreme", "fss_thr89_win5", "rapsd_log_distance", "minkowski_distance",
        "gpd_exc_count_ratio", "rl_bias_mean_abs", "anisotropy_axis_over_diag",
        "peak_ratio_median"]
print(" | ".join(f"{h:>12s}" for h in hdr))
for lab, s in rows:
    cells = [lab] + [s.get(k) for k in keys]
    print(" | ".join(f"{c:>12.3f}" if isinstance(c, float) else f"{str(c):>12s}"
                     for c in cells))

print("\nLaTeX rows:")
for lab, s in rows:
    f = lambda v, n=2: ("--" if v is None else f"{v:.{n}f}")
    print(f"    {lab:<20s} & {f(s.get('rmse_extreme'))} & {f(s.get('fss_thr89_win5'),3)} & "
          f"{f(s.get('rapsd_log_distance'))} & {f(s.get('minkowski_distance'))} & "
          f"{f(s.get('gpd_exc_count_ratio'),3)} & {f(s.get('rl_bias_mean_abs'),3)} & "
          f"{f(s.get('anisotropy_axis_over_diag'))} \\\\")

print("\nReading order: activity first (an inert run's structural numbers are")
print("uninformative), then anisotropy (vanilla ~1.87 — a structural gain paired with a")
print("much higher value is the grid-aligned filament pathology, which no headline metric")
print("detects), then the tail columns. The Minkowski distance is the training objective")
print("for Minkowski-trained rows and is not independent evidence there.")
PY
}

# ---------------------------------------------------------------------
say "Full evaluation sweep"
echo "  python   : $PYTHON"
echo "  GPU      : ${GPU} (CUDA_DEVICE_ORDER=PCI_BUS_ID)"
echo "  POT level: ${POT} mm/h — identical for every row"
echo "  figures  : ${FIGDIR}"
[ "$DRY_RUN" != "0" ] && echo "  MODE     : dry run, nothing will be executed"

case "$STAGE" in
  all)       stage_archive; stage_backbones; stage_bicubic; stage_fm; stage_figures; stage_summary ;;
  backbones) stage_archive; stage_backbones; stage_summary ;;
  bicubic)   stage_bicubic ;;
  fm)        stage_fm; stage_summary ;;
  figures)   stage_figures ;;
  summary)   stage_summary ;;
  *) echo "unknown STAGE=${STAGE}; expected all|backbones|bicubic|fm|figures|summary" >&2; exit 1 ;;
esac

say "Done."
echo "  evaluations: eval_results/"
echo "  figures    : ${FIGDIR}/{study1,study2,all}"
