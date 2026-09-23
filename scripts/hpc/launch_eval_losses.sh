#!/bin/bash
# =====================================================================
# Evaluate the Study-1 competing-loss backbones under one fixed protocol.
#
# Every row must share the same peaks-over-threshold level, or the GPD shape error and the
# return-level bias are not comparable between models: those columns are computed against a
# fit whose observed reference depends on that level. POT_THRESHOLD below is applied to all
# runs, including the reference backbones, so the whole table is on one footing.
#
# Runs two evaluations per checkpoint:
#   eval_extremes.py -> tab:iid row (FSS, GPD, return levels, anisotropy, peak ratio)
#   eval_backbone.py -> tab:prelim row (MAE, SAL, isoperimetric, RAPSD)
#
# The v1 runs of the four competing losses were all INERT: MAE cost only +0.6-0.8% against
# the vanilla backbone (the Minkowski loss at its working weight pays +3.8%), and the
# radially averaged spectrum got *worse* under a loss that optimises the spectrum directly.
# That is the signature of a weight too small for the loss to enter the objective, not of a
# weak loss. The v2 runs below re-train at per-loss weights; the collector therefore reports
# the MAE cost against vanilla as an ACTIVITY CHECK and flags any run still under +1%.
#
# Usage:
#   bash scripts/hpc/eval_losses.sh
#   DRY_RUN=1 bash scripts/hpc/eval_losses.sh      # print the plan only
#   ONLY=ssim_v2 bash scripts/hpc/eval_losses.sh   # single entry
# Resumable: a run whose summary already exists is skipped.
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set +u
source "${SCRIPT_DIR}/env.sh"
set -u
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config.yaml}"
GPU="${GPU:-1}"
# Shared node: GPU 1 only (EXPERIMENTS.md "Compute node"). Refuse anything else.
[[ "$GPU" == "1" ]] || { echo "GPU=$GPU is not allowed on this node: GPU 1 only." >&2; exit 1; }
POT_THRESHOLD="${POT_THRESHOLD:-31}"     # ONE level for every row in the table
SPLIT="${SPLIT:-test}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"

# "label|run directory under runs/sr_analytical/"
# References first so the collector has the vanilla MAE baseline available.
# The _v1 rows are the inert shared-weight scouting runs, kept for the record; drop them
# once the v2 rows are confirmed active.
RUNS=(
  "vanilla|UNet_Ana_20260623_144958"
  "minkowski|UNet_Ana_20260731_100128"

  # --- v2: per-loss weights ---
  "spectral_v2|UNet_Ana_20260829_033325"
  "opticalflow_v2|UNet_Ana_20260827_151319"
  "wetarea_v2|UNet_Ana_20260826_041128"
  "ssim_v2|UNet_Ana_20260824_155753"

  # --- v1: shared weight, all inert (see header) ---
  "spectral_v1|UNet_Ana_20260812_085343"
  "ssim_v1|UNet_Ana_20260817_113622"
)

echo "config : $CONFIG"
echo "POT    : ${POT_THRESHOLD} mm/h (applied to every row)"
echo "split  : ${SPLIT}, full test set (deterministic models)"
echo

for entry in "${RUNS[@]}"; do
  label="${entry%%|*}"; run="${entry#*|}"
  [ -n "$ONLY" ] && [ "$label" != "$ONLY" ] && continue
  ckpt="${PROJECT_ROOT}/runs/sr_analytical/${run}/unet_best.pth"
  if [ ! -f "$ckpt" ]; then echo "[skip] ${label}: no checkpoint at ${ckpt}"; continue; fi

  echo "[eval] ${label}  <- ${run}"
  [ "$DRY_RUN" != "0" ] && continue

  # --- tab:iid row ---
  out_ext="${PROJECT_ROOT}/eval_results/extremes"
  if [ -f "${out_ext}/backbone_${label}/extremes_summary.yaml" ]; then
    echo "  [skip] extremes already done"
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
    "$PYTHON" "${PROJECT_ROOT}/scripts/evaluate/eval_extremes.py" "$CONFIG" \
        --model backbone --checkpoint "$ckpt" \
        --pot_threshold "$POT_THRESHOLD" --split "$SPLIT" \
        --tag "backbone_${label}" --output_dir "$out_ext" \
        > "${PROJECT_ROOT}/logs/eval_extremes_${label}.log" 2>&1 \
        && echo "  extremes ok" \
        || echo "  ! extremes FAILED, see logs/eval_extremes_${label}.log"
  fi

  # --- tab:prelim row ---
  out_bb="${PROJECT_ROOT}/eval_results/backbone/${label}"
  if [ -f "${out_bb}/backbone_summary.yaml" ]; then
    echo "  [skip] backbone already done"
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
    "$PYTHON" "${PROJECT_ROOT}/scripts/evaluate/eval_backbone.py" "$CONFIG" \
        --checkpoint "$ckpt" --split "$SPLIT" --output_dir "$out_bb" \
        > "${PROJECT_ROOT}/logs/eval_backbone_${label}.log" 2>&1 \
        && echo "  backbone ok" \
        || echo "  ! backbone FAILED, see logs/eval_backbone_${label}.log"
  fi
done

# --- collect the table columns ---
echo
"$PYTHON" - "$POT_THRESHOLD" <<'PY'
import glob, os, sys, yaml

pot = sys.argv[1]

# MAE from the backbone summaries: the activity check.
mae = {}
for f in glob.glob("eval_results/backbone/*/backbone_summary.yaml"):
    lab = os.path.basename(os.path.dirname(f))
    try:
        mae[lab] = yaml.safe_load(open(f)).get("mae_overall")
    except Exception:
        pass
base = mae.get("vanilla")

rows = []
for f in sorted(glob.glob("eval_results/extremes/backbone_*/extremes_summary.yaml")):
    lab = os.path.basename(os.path.dirname(f)).replace("backbone_", "")
    rows.append((lab, yaml.safe_load(open(f))))

# --- consistency guard: gpd_xi_obs depends only on the data, so it must be identical ---
obs = {r[1].get("gpd_xi_obs") for r in rows if r[1].get("gpd_xi_obs") is not None}
if len(obs) > 1:
    print("!! WARNING: rows were fitted at DIFFERENT peaks-over-threshold levels.")
    print(f"   gpd_xi_obs takes {len(obs)} distinct values: {sorted(obs)}")
    print("   The xi-error and RL-bias columns are NOT comparable across those rows.\n")

print("=== activity check (is the auxiliary loss actually doing anything?) ===")
print(f"{'run':>16s} {'MAE':>9s} {'vs vanilla':>11s}   verdict")
for lab, _ in rows:
    m = mae.get(lab)
    if m is None or base is None:
        print(f"{lab:>16s} {'--':>9s} {'--':>11s}   no backbone_summary")
        continue
    d = (m / base - 1) * 100
    v = ("INERT (weight too low)" if d < 1.0
         else "active" if d < 8.0 else "large cost -- check images")
    print(f"{lab:>16s} {m:9.5f} {d:+10.1f}%   {v}")
print("  reference: Minkowski at its working weight pays +3.8%; below ~+1% the loss has")
print("  not entered the objective and the structural columns say nothing about it.\n")

print(f"=== tab:iid columns (POT u={pot}) ===")
hdr = ["loss", "RMSE_ext", "FSS@89", "RAPSD", "Mink", "exc.ratio", "RL bias", "xi err",
       "aniso", "peak"]
print(" | ".join(f"{h:>10s}" for h in hdr))
for lab, s in rows:
    vals = [lab, s.get("rmse_extreme"), s.get("fss_thr89_win5"), s.get("rapsd_log_distance"),
            s.get("minkowski_distance"), s.get("gpd_exc_count_ratio"),
            s.get("rl_bias_mean_abs"), s.get("gpd_xi_abs_err"),
            s.get("anisotropy_axis_over_diag"), s.get("peak_ratio_median")]
    print(" | ".join(f"{v:>10.3f}" if isinstance(v, float) else f"{str(v):>10s}" for v in vals))

print("\nLaTeX rows (exceedance ratio and RL bias are the tail columns; the shape error is a")
print("secondary diagnostic reported in the text, not here):")
for lab, s in rows:
    f = lambda v, n=2: ("--" if v is None else f"{v:.{n}f}")
    print(f"    Backbone + {lab:<14s} & {f(s.get('rmse_extreme'))} & "
          f"{f(s.get('fss_thr89_win5'),3)} & {f(s.get('rapsd_log_distance'))} & "
          f"{f(s.get('minkowski_distance'))} & {f(s.get('gpd_exc_count_ratio'),3)} & "
          f"{f(s.get('rl_bias_mean_abs'),3)} & {f(s.get('anisotropy_axis_over_diag'))} \\\\")

print("\nReading order: check the activity column FIRST -- an inert run's structural numbers")
print("are uninformative. Then anisotropy (vanilla ~1.87): a structural gain paired with a")
print("materially higher value is the grid-aligned filament pathology, which no headline")
print("metric detects. Look at the sample images before believing such a row.")
PY
