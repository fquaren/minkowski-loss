#!/usr/bin/env python
"""Compare refetched archive days with the original (150 mm/h-capped, no-QIND) day stores.

A refetched day may replace its original only if it is the same product apart from the
cap: complete store with TOT_PREC + QIND, at least 90% of the original's time steps, the
same radar footprint (NaN mask IoU >= 0.9 at shared times), and, where the original is
between the drizzle floor and 149 mm/h, values that agree within 1% on >= 95% of pixels.
Also reports how much the refetch holds above 150 mm/h, which is the point of the swap.

Writes <out>.csv (one row per day) and prints a summary; exit code 0 if >= 95% of days pass.

    python scripts/dataset_v2/verify_refetch.py --orig_dir .../raw/OPERA \
        --new_dir .../raw/OPERA_refetch --out logs/refetch_verify
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def check_day(orig, new, var="TOT_PREC", n_times=4):
    import xarray as xr
    r = {"day": os.path.basename(orig)}
    if not os.path.exists(os.path.join(new, ".zmetadata")):
        r["reason"] = "no complete refetched store"; return r
    a = xr.open_zarr(orig, consolidated=True); b = xr.open_zarr(new, consolidated=True)
    r["t_orig"], r["t_new"] = a.sizes["time"], b.sizes["time"]
    r["has_qind"] = "QIND" in b
    shared = np.intersect1d(a.time.values, b.time.values)
    r["t_shared"] = len(shared)
    if not len(shared):
        r["reason"] = "no shared timestamps"; return r
    pick = shared[np.linspace(0, len(shared) - 1, min(n_times, len(shared))).astype(int)]
    iou, agree, n_cmp, gt150 = [], 0, 0, 0
    for t in pick:
        x = a[var].sel(time=t).values.astype(np.float64)
        y = b[var].sel(time=t).values.astype(np.float64)
        mx, my = np.isfinite(x), np.isfinite(y)
        iou.append((mx & my).sum() / max((mx | my).sum(), 1))
        m = mx & my & (x >= 0.1) & (x <= 149.0)
        if m.any():
            agree += int((np.abs(y[m] - x[m]) <= 0.01 * x[m] + 1e-3).sum()); n_cmp += int(m.sum())
    for t in b.time.values:
        y = b[var].sel(time=t).values
        gt150 += int(np.nansum(y > 150.0))
    r["iou"] = float(np.min(iou)); r["agree"] = agree / n_cmp if n_cmp else np.nan
    r["n_cmp"] = n_cmp; r["new_px_gt150"] = gt150
    r["orig_max"] = float(np.nanmax(a[var].values)); r["new_max"] = float(np.nanmax(b[var].values))
    ok = (r["has_qind"] and r["t_new"] >= 0.9 * r["t_orig"] and r["iou"] >= 0.9
          and (np.isnan(r["agree"]) or r["agree"] >= 0.95))
    r["pass"] = bool(ok); r["reason"] = "" if ok else "criteria"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--new_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    days = sorted(d for d in glob.glob(os.path.join(a.orig_dir, "20[0-9]" * 1 + "[0-9]" * 5))
                  if not os.path.isdir(os.path.join(d, "QIND")))
    rows = []
    for k, d in enumerate(days, 1):
        try:
            rows.append(check_day(d, os.path.join(a.new_dir, os.path.basename(d))))
        except Exception as e:
            rows.append({"day": os.path.basename(d), "reason": f"{type(e).__name__}: {e}"})
        if k % 50 == 0:
            print(f"  {k}/{len(days)}", flush=True)
    t = pd.DataFrame(rows)
    t["pass"] = t.get("pass", False).fillna(False).astype(bool)
    t.to_csv(a.out + ".csv", index=False)
    n, p = len(t), int(t["pass"].sum())
    print(f"[verify] {p}/{n} original days pass; failures: "
          f"{t.loc[~t['pass'], 'reason'].value_counts().to_dict()}")
    if "agree" in t:
        print(f"  value agreement below 149 mm/h: median {t['agree'].median():.4f}, "
              f"min {t['agree'].min():.4f}; footprint IoU min {t['iou'].min():.3f}")
        print(f"  refetch pixels > 150 mm/h: {int(t['new_px_gt150'].sum()):,} over {n} days "
              f"(originals: max {t['orig_max'].max():.1f})")
    sys.exit(0 if n and p >= 0.95 * n else 1)


if __name__ == "__main__":
    main()
