#!/usr/bin/env python
"""`radar_frame.temporal.min_wet_jump` from normal operation (DECISIONS §30, item 4).

s = min(w_t - w_{t-1}, w_t - w_{t+1}): how far a radar-frame's wet fraction (share of owned
pixels >= 1 mm/h) exceeds both neighbour frames. Over radar-frames in normal operation: 500
random days (seed 0) of the 2026-10-09 radar pass, radar-days with score < 1 and no flagged
frame, the frame and both neighbours with n_valid >= radar_frame.min_valid and consecutive.
Proposed value: the 99.9th percentile of s.

    python scripts/data_quality/wet_jump.py
"""

import glob
import os

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Q4 = "/home/fquareng/work/data/extremes/OPERA/quality_v4"


def main():
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    mv = q["radar_frame"]["min_valid"]
    files = sorted(glob.glob(f"{Q4}/radars/*.npz"))
    files = list(np.random.default_rng(0).choice(files, 500, replace=False))
    r = pd.read_csv(f"{Q4}/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "score"])
    bad = set(zip(r[r.score >= 1].radar, r[r.score >= 1].day))
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    bad |= set(zip(ff.radar, ff.timestamp.str[:8]))
    s_all = []
    for f in files:
        z = np.load(f)
        day = os.path.basename(f)[:8]
        F, keys = z["feats"], z["keys"].astype(str)
        t = pd.to_datetime(z["times"].astype(str), format="%Y%m%d%H%M%S").values
        consec = np.r_[False, np.diff(t) == np.timedelta64(15, "m")]          # t-1 -> t is 15 min
        w, n = F[..., 1], F[..., 0]
        ok = n >= mv
        for j, k in enumerate(keys):
            if (k, day) in bad:
                continue
            for i in range(1, len(t) - 1):
                if consec[i] and consec[i + 1] and ok[i - 1:i + 2, j].all():
                    s_all.append(min(w[i, j] - w[i - 1, j], w[i, j] - w[i + 1, j]))
    s = np.array(s_all)
    qs = {p: float(np.percentile(s, p)) for p in (99, 99.9, 99.99)}
    txt = (f"{len(files)} days, {len(s):,} radar-frames in normal operation; s = wet-fraction excess "
           f"over both neighbours: q99 {qs[99]:.3f}, q99.9 {qs[99.9]:.3f}, q99.99 {qs[99.99]:.3f}; "
           f"share >= 0.15 (current): {(s >= 0.15).mean():.2e}. Proposed min_wet_jump = {qs[99.9]:.3f}")
    print(txt)
    os.makedirs(f"{Q4}/calib/wet_jump", exist_ok=True)
    open(f"{Q4}/calib/wet_jump/summary.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
