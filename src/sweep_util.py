#!/usr/bin/env python
"""Helper for the reward-fine-tuning sweep. Two jobs, no repo state touched.

  patch    write a trial config = base config + explicit key overrides
  collect  append one row of an extremes_summary.yaml to a tidy CSV

Kept deliberately small: the sweep logic lives in sweep_reward.sh, and this file only
does the two things bash does badly (YAML editing and typed value parsing).
"""

import argparse
import csv
import os
import re
import sys

import yaml

# Columns recorded per trial. Ordered: identification, then the extreme-tail metrics that
# the sweep is meant to attribute, then the guard diagnostics, then the objective itself
# (recorded last and flagged, because it is what training optimises and so is circular).
COLUMNS = [
    "trial", "overrides", "checkpoint", "n_samples",
    # primary tail metrics
    "rl_bias_mean_abs", "gpd_xi_abs_err", "gpd_xi_pred", "gpd_xi_obs",
    "gpd_exc_count_ratio", "fss_thr89_win5", "fss_thr53_win5",
    "rmse_extreme", "crps_extreme", "crps_mean",
    # structure
    "rapsd_log_distance", "SAL_S",
    # guards (not optimised by the reward)
    "peak_ratio_median", "peak_ratio_extreme_median", "anisotropy_axis_over_diag",
    # objective (circular for coupled models)
    "minkowski_distance",
]


def _parse_val(v):
    """Type a CLI override value. Avoids PyYAML's float quirks (it reads '1e-5' as a str)."""
    s = v.strip()
    low = s.lower()
    if low in ("null", "none", ""):
        return None
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def cmd_patch(args):
    with open(args.base) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        if "=" not in item:
            sys.exit(f"bad override {item!r}, expected KEY=VALUE")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _parse_val(v)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(args.out)


def cmd_collect(args):
    with open(args.summary) as f:
        s = yaml.safe_load(f)
    row = {c: "" for c in COLUMNS}
    row["trial"] = args.trial
    row["overrides"] = args.overrides
    row["checkpoint"] = args.checkpoint
    for c in COLUMNS:
        if c in s:
            row[c] = s[c]
    if "SAL_median" in s and isinstance(s["SAL_median"], (list, tuple)):
        row["SAL_S"] = s["SAL_median"][0]
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended {args.trial} -> {args.csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("patch")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.set_defaults(func=cmd_patch)

    c = sub.add_parser("collect")
    c.add_argument("--summary", required=True)
    c.add_argument("--trial", required=True)
    c.add_argument("--overrides", default="")
    c.add_argument("--checkpoint", default="")
    c.add_argument("--csv", required=True)
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()