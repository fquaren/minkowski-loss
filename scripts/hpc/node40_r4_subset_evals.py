"""FM-queue step 2 on node40: every R4 U-Net evaluated on val_extremes1024 / val_subset4096 (u = 50.1).

Same commands and output directories as step 2 of run_fm_r4_queue.sh, so the node32 queue skips whatever is
already there and goes straight to the cut-arm FM base. Shard i of n takes every n-th (label, subset) job.
Launch through node40_run.sh (GPUs 0-3, CPUs 0-5):
    scripts/hpc/node40_run.sh <gpu> r4sub<i> scripts/hpc/node40_r4_subset_evals.py <i> <n>
"""
import os
import subprocess
import sys

W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
R = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c"
OUTE = f"{W}/eval_results/extremes"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

shard, nshards = int(sys.argv[1]), int(sys.argv[2])
jobs = []
for line in open(f"{R}/logs/r4_queue_done.txt"):
    lab, run = line.split()[:2]
    ck = f"{R}/runs/sr_analytical/{run}/unet_latest.pth"
    if not os.path.isfile(ck):
        ck = f"{W}/runs/sr_analytical/{run}/unet_latest.pth"
    for sub in ("ext", "sub"):
        jobs.append((lab, sub, ck))

for lab, sub, ck in jobs[shard::nshards]:
    out = f"{OUTE}/v4adv_val{sub}"
    if os.path.isdir(f"{out}/{lab}"):
        print(f"exists, skipping {lab} {sub}", flush=True)
        continue
    print(f"=== {lab} {sub} {ck}", flush=True)
    r = subprocess.run([sys.executable, "-u", f"{ROOT}/scripts/evaluate/eval_extremes.py",
                        f"{W}/configs/config_v4_adv_fm_eval{sub}.yaml", "--model", "backbone",
                        "--checkpoint", ck, "--split", "val", "--pot_threshold", "50.1",
                        "--tag", lab, "--output_dir", out], cwd=ROOT)
    print(f"{'ok' if r.returncode == 0 else 'FAILED'} {lab} {sub}", flush=True)
print("=== SUBSET_EVALS_DONE", flush=True)
