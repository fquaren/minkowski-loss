"""Main-table re-evals (DECISIONS §47) on val_extremes1024 / val_subset4096, u = 50.1, into new directories
v4adv_val{ext,sub}_tab (the old ones are left as they are). FM rows score all 16 members; FM cut-arm rows use the
cut backbone's sigma_r (logs/tab/configs). Shard i of n takes every n-th job; finished jobs are skipped, so the
script can be rerun to pick up runs that finish later (s44 cut, G3). Launch per GPU through node40_run.sh:
    scripts/hpc/node40_run.sh <gpu> tab<i> logs/tab/tab_evals.py <i> <n>
"""
import os
import subprocess
import sys

ROOT = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4"
W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
R = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c"
OUTE = f"{W}/eval_results/extremes"
FMR = f"{ROOT}/runs/sr_fm_minkowski"
BB_FULL = f"{W}/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth"
BB_CUT = f"{R}/runs/sr_analytical/UNet_Ana_20261105_142458/unet_latest.pth"


def unet(run):
    ck = f"{R}/runs/sr_analytical/{run}/unet_latest.pth"
    return ck if os.path.isfile(ck) else f"{W}/runs/sr_analytical/{run}/unet_latest.pth"


def fm_run(prefix):          # latest run dir of an R4 fine-tune label, or None
    ds = sorted(d for d in os.listdir(FMR) if d.startswith(prefix + "_2026"))
    return f"{FMR}/{ds[-1]}/fm_mink_latest.pth" if ds else None


DET = {  # label: (full run, cut run)
    "mse_s42": ("UNet_Ana_20261105_150530", "UNet_Ana_20261105_142458"),
    "mse_s43": ("UNet_Ana_20261109_125359", "UNet_Ana_20261107_150924"),
    "mse_s44": ("UNet_Ana_20261111_162434", "UNet_Ana_20261111_143655"),
    "wmse_s42": ("UNet_Ana_20261118_151313", "UNet_Ana_20261120_003042"),
    "quantile_s42": ("UNet_Ana_20261118_151343", "UNet_Ana_20261120_004843"),
    "fss_s42": ("UNet_Ana_20261118_151413", "UNet_Ana_20261120_071846"),
    "spectralhann0.051_s42": ("UNet_Ana_20261121_150629", "UNet_Ana_20261121_224857"),
    "area_s42": ("UNet_Ana_20261109_125328", "UNet_Ana_20261109_125213"),
    "mink_s42": ("UNet_Ana_20261105_150601", "UNet_Ana_20261105_150429"),
    "mink_s43": ("UNet_Ana_20261109_125433", "UNet_Ana_20261107_150954"),
    "mink_s44": ("UNet_Ana_20261111_170644", "UNet_Ana_20261111_152845"),
}
FM = {  # label: (full checkpoint, cut checkpoint)
    "fm_clean": (f"{W}/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth",
                 open(f"{ROOT}/logs/fm_r4/fmb_cut.txt").read().strip()),
    "fm_ctl_s42": (f"{W}/runs/sr_fm_minkowski/FMenergyctl_v4adv_20261030_141741/fm_mink_latest.pth",
                   f"{FMR}/FMr4_ctl_cut_s42_20261120_090336/fm_mink_latest.pth"),
    "fm_mink_s42": (f"{W}/runs/sr_fm_minkowski/FMenergyqm_v4adv_20261031_022957/fm_mink_latest.pth",
                    f"{FMR}/FMr4_mink_cut_s42_20261120_211914/fm_mink_latest.pth"),
}
for s in (43, 44):
    for m in ("ctl", "mink"):
        FM[f"fm_{m}_s{s}"] = (fm_run(f"FMr4_{m}_full_s{s}"), fm_run(f"FMr4_{m}_cut_s{s}"))


def done_training(ck):       # an R4 fine-tune is finished when the queue has written its label to done.txt
    if ck is None or not os.path.isfile(ck):
        return False
    if not ck.startswith(FMR):
        return True
    run = os.path.basename(os.path.dirname(ck))
    return any(run in line for line in open(f"{ROOT}/logs/fm_r4/done.txt"))


jobs = []   # (label, subset, args); FM first, so the long jobs spread over the shards
for lab, (f, c) in FM.items():
    for arm, ck, bb in (("full", f, BB_FULL), ("cut", c, BB_CUT)):
        for sub in ("ext", "sub"):
            cfg = (f"{ROOT}/logs/tab/configs/config_v4_adv_fm_eval{sub}_cut.yaml" if arm == "cut"
                   else f"{W}/configs/config_v4_adv_fm_eval{sub}.yaml")
            jobs.append((f"{lab}_{arm}", sub, ck, [cfg, "--model", "fm", "--fm_checkpoint", ck, "--backbone", bb,
                                                    "--ensemble", "16"]))
for lab, (f, c) in DET.items():
    for arm, run in (("full", f), ("cut", c)):
        for sub in ("ext", "sub"):
            a = [f"{W}/configs/config_v4_adv_fm_eval{sub}.yaml", "--model", "backbone", "--checkpoint", unet(run)]
            jobs.append((f"{lab}_{arm}", sub, unet(run), a))
            if lab == "mse_s42":
                jobs.append((f"mse_qm_s42_{arm}", sub, unet(run), a + ["--qm", f"{R}/logs/node40/qm/qm_{arm}.json"]))
for sub in ("ext", "sub"):
    a = [f"{W}/configs/config_v4_adv_fm_eval{sub}.yaml", "--model", "backbone", "--checkpoint",
         unet("UNet_Ana_20261107_151025")]
    jobs.append(("mse_randdrop_s42", sub, unet("UNet_Ana_20261107_151025"), a))

shard, nshards = int(sys.argv[1]), int(sys.argv[2])
# longest-first greedy split by a rough cost (16 members, 4x the patches on valsub), the same in every shard
cost = lambda j: (16 if "--ensemble" in j[3] else 1) * (4 if j[1] == "sub" else 1)
load, mine = [0] * nshards, []
for j in sorted(jobs, key=cost, reverse=True):
    k = load.index(min(load)); load[k] += cost(j)
    if k == shard:
        mine.append(j)
env = dict(os.environ, PYTHONPATH=ROOT)
for lab, sub, ck, a in mine:
    out = f"{OUTE}/v4adv_val{sub}_tab"
    if os.path.isfile(f"{out}/{lab}/extremes_summary.yaml"):
        print(f"exists, skipping {lab} {sub}", flush=True)
        continue
    if not done_training(ck):
        print(f"not trained yet, skipping {lab} {sub}", flush=True)
        continue
    print(f"=== {lab} {sub} {ck}", flush=True)
    r = subprocess.run([sys.executable, "-u", f"{ROOT}/scripts/evaluate/eval_extremes.py", *a, "--split", "val",
                        "--pot_threshold", "50.1", "--tag", lab, "--output_dir", out], cwd=ROOT, env=env)
    print(f"{'ok' if r.returncode == 0 else 'FAILED'} {lab} {sub}", flush=True)
print("=== TAB_EVALS_DONE", flush=True)
