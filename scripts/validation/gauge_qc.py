#!/usr/bin/env python
"""Sanity check of the 10-min gauge records, using the gauges only (never the radar, which
is what they are meant to validate).

Flags, per wet 10-min value:
  bound      > BOUND mm per 10 min: physically impossible here (the records hold 99.9 and
             149.6 mm sentinel / fault values; the largest plausible values are ~40-45 mm)
  stuck      the same value >= 1 mm repeated in >= STUCK_RUN consecutive intervals
  isolated   >= ISO_MIN mm with exactly 0 in the intervals before and after, AND no rain
             (>= 0.1 mm) at any gauge with a complete record that day within ISO_KM, over
             t +- 30 min. Either condition alone is common in real convection; both together
             are a burst nothing else saw.
A value whose neighbours all lack a complete record that day is counted as `unverifiable`
and kept.

Output: <gauge_dir>/gauge_qc.csv (network, station, t [minutes since 1970, as labelled],
rr, reason) and a printed summary. `validate_tail.py` masks the flagged values.

    python scripts/validation/gauge_qc.py
"""

import argparse
import os

import numpy as np
import pandas as pd

D = "/home/fquareng/work/data/extremes/OPERA/validation/gauges"
BOUND = 50.0
STUCK_RUN = 6
ISO_MIN = 10.0
ISO_KM = 30.0
ISO_WIN = 30           # minutes
SHIFT = np.int64(1) << 40


def load(gdir):
    st_all, sid, t, rr, avail, days = [], [], [], [], [], None
    off = 0
    for net in ("dwd", "smn"):
        st = pd.read_csv(os.path.join(gdir, f"{net}_stations.csv"), dtype={"station": str})
        w = np.load(os.path.join(gdir, f"{net}_wet.npz"))
        a = np.load(os.path.join(gdir, f"{net}_avail.npz"))
        st["network"] = net
        st_all.append(st)
        sid.append(w["sid"].astype(np.int64) + off); t.append(w["t"]); rr.append(w["rr"])
        avail.append(a["n_valid"])
        days = a["days"]
        off += len(st)
    st = pd.concat(st_all, ignore_index=True)
    return st, np.concatenate(sid), np.concatenate(t), np.concatenate(rr), np.vstack(avail), days


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gauge_dir", default=D)
    a = ap.parse_args()
    st, sid, t, rr, n_valid, days = load(a.gauge_dir)
    order = np.lexsort((t, sid))
    sid, t, rr = sid[order], t[order], rr[order]
    key = sid * SHIFT + t
    reason = np.full(len(rr), "", dtype=object)

    # bound
    reason[rr > BOUND] = "bound"

    # stuck: runs of identical values >= 1 mm in consecutive intervals of one station
    same = (sid[1:] == sid[:-1]) & (t[1:] - t[:-1] == 10) & (rr[1:] == rr[:-1]) & (rr[1:] >= 1.0)
    run_start = np.r_[True, ~same]
    run_id = np.cumsum(run_start) - 1
    run_len = np.bincount(run_id)
    stuck = run_len[run_id] >= STUCK_RUN
    reason[stuck & (reason == "")] = "stuck"

    # isolated: temporal isolation first (cheap), then spatial support
    def value_at(k):
        i = np.searchsorted(key, k)
        i = np.minimum(i, len(key) - 1)
        return np.where(key[i] == k, rr[i], 0.0)
    cand = np.nonzero((rr >= ISO_MIN) & (reason == ""))[0]
    tiso = (value_at(key[cand] - 10) == 0) & (value_at(key[cand] + 10) == 0)
    cand = cand[tiso]
    ok_xy = np.isfinite(st["row"].values)
    xy = np.c_[st["row"].values, st["col"].values]
    tree = cKDTree(np.where(ok_xy[:, None], xy, 1e9))
    day0 = days[0].astype("datetime64[m]").astype(np.int64)
    n_iso = n_unver = 0
    for i in cand:
        s, tt = sid[i], t[i]
        if not ok_xy[s]:
            continue
        nb = [j for j in tree.query_ball_point(xy[s], ISO_KM / 2.0) if j != s]
        d = int((tt - day0) // 1440)
        nb = [j for j in nb if 0 <= d < n_valid.shape[1] and n_valid[j, d] >= 140]
        if not nb:
            n_unver += 1
            continue
        wet = False
        for j in nb:
            lo, hi = np.searchsorted(key, [j * SHIFT + tt - ISO_WIN, j * SHIFT + tt + ISO_WIN + 1])
            if (rr[lo:hi] >= 0.1).any():
                wet = True
                break
        if not wet:
            reason[i] = "isolated"
            n_iso += 1

    flag = reason != ""
    out = pd.DataFrame({"network": st["network"].values[sid[flag]], "station": st["station"].values[sid[flag]],
                        "t": t[flag], "rr": rr[flag], "reason": reason[flag]})
    out.to_csv(os.path.join(a.gauge_dir, "gauge_qc.csv"), index=False)
    print(f"{len(rr):,} wet 10-min values; flagged {flag.sum():,}:")
    for r, g in out.groupby("reason"):
        print(f"  {r:9s} {len(g):7,}  (values >= 10 mm: {(g.rr >= 10).sum():,}; max {g.rr.max():.1f})")
    print(f"  temporally isolated >= {ISO_MIN:g} mm: {len(cand):,}, of which spatially unsupported "
          f"{n_iso:,}, unverifiable (no neighbour with a complete day) {n_unver:,}")
    print(f"  share of values >= 10 mm flagged: {(flag & (rr >= 10)).sum() / max((rr >= 10).sum(), 1):.1%}")


if __name__ == "__main__":
    main()
