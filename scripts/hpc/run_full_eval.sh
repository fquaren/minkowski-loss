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
#   STAGE=fields    bash scripts/hpc/run_full_eval.sh # re-dump + re-plot the field panels
#
# Stages: backbones | bicubic | fm | fields | figures | all (default)
#
# The field stage is cheap: it runs each model over a handful of patches rather than the
# split, so it costs seconds. Delete eval_results/fields/*.npz to force a re-dump.
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
# Shared node: GPU 1 only (EXPERIMENTS.md "Compute node"). Refuse anything else.
[[ "$GPU" == "1" ]] || { echo "GPU=$GPU is not allowed on this node: GPU 1 only." >&2; exit 1; }
POT="${POT:-31}"                      # one level for every row
SPLIT="${SPLIT:-test}"
ENSEMBLE="${ENSEMBLE:-16}"
FM_BATCHES="${FM_BATCHES:-32}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"
FIGDIR="${FIGDIR:-figures/committee}"
EXT="${PROJECT_ROOT}/eval_results/extremes"
BB="${PROJECT_ROOT}/eval_results/backbone"
FIELDS="${PROJECT_ROOT}/eval_results/fields"
# One field-panel patch per percentile of the target maximum. Spread rather than top-N:
# the patch pool contains bad radar observations, and they concentrate at the very top of
# the intensity distribution, so a top-N selection renders artefacts almost exclusively.
FIELD_PCT="${FIELD_PCT:-99.99 99.9 99.5 99 97 95 90 75 50 25}"
N_EXTREME="${N_EXTREME:-3}"           # only used when FIELD_PCT is set empty
N_MID="${N_MID:-1}"                   # plus this many from the middle of the distribution
FIELD_MODE="${FIELD_MODE:-both}"      # compare | detail | both
FIELD_NORM="${FIELD_NORM:-power}"     # colour stretch shared by the precipitation panels
FIELD_INDICES="${FIELD_INDICES:-}"    # explicit dataset indices, overriding the selection
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
  say "Stage 1/5 — deterministic backbones (full test set, POT u=${POT})"
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
  say "Stage 2/5 — bicubic reference"
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
  say "Stage 3/5 — flow matching (M=${ENSEMBLE}, heun-16, ${FM_BATCHES} batches)"
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
# Stage 4: qualitative field panels
# ---------------------------------------------------------------------
# One display label per evaluation tag, shared by the field panels and the figure stage so
# a model is named identically wherever it appears.
_disp() {
  case "$1" in
    vanilla|backbone_vanilla)               echo "MSE" ;;
    minkowski|backbone_minkowski)           echo "+ Minkowski" ;;
    spectral_v2|backbone_spectral_v2)       echo "+ spectral" ;;
    ssim_v2|backbone_ssim_v2)               echo "+ SSIM" ;;
    wetarea_v2|backbone_wetarea_v2)         echo "+ wet area" ;;
    opticalflow_v2|backbone_opticalflow_v2) echo "+ optical flow" ;;
    fm_clean)                               echo "FM clean" ;;
    fm_energy)                              echo "FM + energy" ;;
    fm_reward)                              echo "FM + reward" ;;
    fm_mink_backbone)                       echo "FM on Mink bb" ;;
    *)                                      echo "$1" ;;
  esac
}

# Dump one bundle and render it, unless the bundle is already there.
_field_bundle() {  # $1 = name, $2 = config, $3 = the --model argument string
  local name="$1" cfg="$2" models="$3"
  if [ -z "$models" ]; then
    echo "  [skip] ${name}: no checkpoints found"; return 0
  fi
  if [ -f "${FIELDS}/${name}.npz" ]; then
    echo "  [skip] ${name} dump (done; delete the npz to force a re-dump)"
  else
    echo "  [run ] ${name} dump"
    local sel="--n_extreme ${N_EXTREME} --n_mid ${N_MID}"
    [ -n "$FIELD_PCT" ] && sel="--pct ${FIELD_PCT}"
    if [ -n "$FIELD_INDICES" ]; then
      sel="--indices ${FIELD_INDICES}"
    elif [ -f "${BB}/vanilla/backbone_arrays.npz" ]; then
      # reuse the tmax array the vanilla backbone run already wrote, in dataset order,
      # instead of rescanning the split just to rank patches by intensity
      sel="${sel} --rank_from '${BB}/vanilla/backbone_arrays.npz'"
    fi
    run "$name" "${CUDA_ENV} '$PYTHON' scripts/evaluate/dump_fields.py '${cfg}' \
        ${models} ${sel} --split ${SPLIT} --output '${FIELDS}/${name}.npz' \
        > '${LOGS}/dump_fields_${name}.log' 2>&1" \
      || { echo "    ! failed, see logs/dump_fields_${name}.log"; return 0; }

    # the residual scale must come from a config, never from residual_stats.json
    if [ "$DRY_RUN" = "0" ] && [ -f "${LOGS}/dump_fields_${name}.log" ]; then
      grep -E "residual scale sigma_r|WARNING" "${LOGS}/dump_fields_${name}.log" \
        | sed "s/^/         /" || true
    fi
  fi
  [ -f "${FIELDS}/${name}.npz" ] || return 0
  echo "  [run ] ${name} figures"
  run "$name" "'$PYTHON' scripts/evaluate/make_plots.py \
      --field_bundle '${FIELDS}/${name}.npz' --out '${FIGDIR}/${name}/fields' \
      --field_mode ${FIELD_MODE} --field_norm ${FIELD_NORM} \
      >> '${LOGS}/dump_fields_${name}.log' 2>&1" \
    || echo "    ! figures failed, see logs/dump_fields_${name}.log"
}

stage_fields() {
  if [ -n "$FIELD_INDICES" ]; then
    say "Stage 4/5 — qualitative fields (patches ${FIELD_INDICES})"
  elif [ -n "$FIELD_PCT" ]; then
    say "Stage 4/5 — qualitative fields (one patch per percentile: ${FIELD_PCT})"
  else
    say "Stage 4/5 — qualitative fields (${N_EXTREME} extreme + ${N_MID} mid patches)"
  fi
  mkdir -p "$FIELDS"

  # Study 1: bicubic and every deterministic backbone, on one shared colour scale.
  local s1="--model \"Bicubic|bicubic||\""
  for entry in "${BACKBONES[@]}"; do
    local label="${entry%%|*}" run_dir="${entry#*|}"
    local ckpt="runs/sr_analytical/${run_dir}/unet_best.pth"
    [ -f "$ckpt" ] || continue
    s1="${s1} --model \"$(_disp "$label")|backbone|${ckpt}|\""
  done
  _field_bundle "study1" "config.yaml" "$s1"

  # Study 2: the two backbones the generative stage sits on, then the flow-matching models.
  # Each flow-matching entry carries its own config because that is what pins sigma_r.
  local s2=""
  [ -f "$VANILLA_BB" ] && s2="${s2} --model \"MSE backbone|backbone|${VANILLA_BB}|\""
  [ -f "$MINK_BB" ] && s2="${s2} --model \"Minkowski backbone|backbone|${MINK_BB}|\""
  for entry in "${FM_MODELS[@]}"; do
    IFS='|' read -r label cfg ckpt bbone <<< "$entry"
    [ -f "$ckpt" ] && [ -f "$cfg" ] || continue
    s2="${s2} --model \"$(_disp "$label")|fm|${ckpt}|${bbone}|${cfg}\""
  done
  _field_bundle "study2" "config.yaml" "$s2"
}


# ---------------------------------------------------------------------
# Stage 5: figures
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
  say "Stage 5/5 — figures"
  # make_plots draws the per-sample perception-distortion clouds alongside the aggregate
  # plane. The Minkowski cloud needs the gamma arrays, so it covers the backbone rows only;
  # the spectral cloud needs the per-sample spectral_dist array, which evaluations run
  # before that was added do not have, and it skips itself with a note until they are re-run.

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
      "fm_clean=FM clean" "fm_energy=FM + energy" "fm_reward=FM + reward" \
      "fm_mink_backbone=FM on Mink bb")"
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
  all)       stage_archive; stage_backbones; stage_bicubic; stage_fm; stage_fields; stage_figures; stage_summary ;;
  backbones) stage_archive; stage_backbones; stage_summary ;;
  bicubic)   stage_bicubic ;;
  fm)        stage_fm; stage_summary ;;
  fields)    stage_fields ;;
  figures)   stage_figures ;;
  summary)   stage_summary ;;
  *) echo "unknown STAGE=${STAGE}; expected all|backbones|bicubic|fm|fields|figures|summary" >&2; exit 1 ;;
esac

say "Done."
echo "  evaluations: eval_results/"
echo "  figures    : ${FIGDIR}/{study1,study2,all}"
echo "  fields     : ${FIGDIR}/{study1,study2}/fields  (from ${FIELDS}/*.npz)"
