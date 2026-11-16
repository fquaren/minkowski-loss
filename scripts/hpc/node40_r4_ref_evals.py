"""Reference rows for the R4 scorecard on the FM subsets, with the current eval code (conditioned scores, twCRPS):
the clean FM model on the full list (as ev_fm in run_fm_r4_queue.sh, which evaluates the cut-list FM base as
fm_r4_clean_cut) and the vanilla v4 U-Net. New tags, so the older `fm` / `vanilla` directories stay as they are.
    scripts/hpc/node40_run.sh <gpu> r4ref scripts/hpc/node40_r4_ref_evals.py
"""
import os
import subprocess
import sys

W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
OUTE = f"{W}/eval_results/extremes"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FMB = f"{W}/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth"
BB = f"{W}/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth"

jobs = []
for sub in ("ext", "sub"):
    jobs.append((sub, "fm_r4_clean_full", ["--model", "fm", "--fm_checkpoint", FMB, "--backbone", BB, "--ensemble", "16"]))
    jobs.append((sub, "r4tab_vanilla", ["--model", "backbone", "--checkpoint", BB]))
for sub, tag, extra in jobs:
    out = f"{OUTE}/v4adv_val{sub}"
    if os.path.isdir(f"{out}/{tag}"):
        print(f"exists, skipping {tag} {sub}", flush=True)
        continue
    print(f"=== {tag} {sub}", flush=True)
    r = subprocess.run([sys.executable, "-u", f"{ROOT}/scripts/evaluate/eval_extremes.py",
                        f"{W}/configs/config_v4_adv_fm_eval{sub}.yaml", *extra, "--split", "val",
                        "--pot_threshold", "50.1", "--tag", tag, "--output_dir", out], cwd=ROOT)
    print(f"{'ok' if r.returncode == 0 else 'FAILED'} {tag} {sub}", flush=True)
print("=== REF_EVALS_DONE", flush=True)
