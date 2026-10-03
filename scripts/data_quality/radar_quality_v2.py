#!/usr/bin/env python
"""Which radars are consistently bad, per year, over the whole archive?

Extends `radar_attribution.py` (745 audited days) to every day, from two sources:

  climatology  (`clutter_climatology.npz`, all days, per year): over the pixels each radar
               owns (nearest active radar within 250 km, as in `scan_flags.py`) with
               >= 200 valid steps: hot-pixel share, and exceedance frequencies >= 31, 150 and
               500 mm/h with hot pixels excluded;
  flags        (`quality_v2/flags`, if present), joined to the tile tables: per radar-year,
               the tail tiles (cleaned max >= 31) whose maximum is in a cell with no temporal
               support, or on a range ring.

Because rain climates differ, every frequency is also scored **relative to the 5 nearest
other radars that year** (ratio to their median). A radar-year is an outlier on a signal
when its relative score is in the worst 5% of all radar-years and above zero (the flag
shares are zero for most radar-years, so the 95th percentile alone can be 0); a radar is
**consistently bad** when it is an outlier on the same signal in >= half of its years
(>= 3 years).

Radars are keyed by ODIM code, or by location where the database has none. Database rows
that describe the same site (current + archive entry, overlapping years) are merged into
one radar, so a site is never its own neighbour and its pixels are not split.

Outputs under --out_dir: radar_year.csv, radar_summary.csv, radar_quality.md. Exclusion is
NOT applied anywhere: the list is validated against gauges first (`validate_tail.py`).

    python scripts/data_quality/radar_quality_v2.py
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_v2")))

Q = "/home/fquareng/work/data/extremes/OPERA/quality_v2"
SIGNALS = ["hot_frac", "rel_f31", "rel_f150", "rel_f500", "unsup_share", "ring_share"]


_OWNERS = {}


def owners_year(year):
    """(meta, owner) for `year`: one row per radar key (ODIM code, or location if it has
    none), duplicate database entries merged; owner = (H, W) index into meta, -1 beyond
    250 km. Same nearest-radar rule as `scan_flags.py`."""
    if year not in _OWNERS:
        import scan_flags as sf
        act, owner, _ = sf._sites_year(year)
        odim = act["odim"].fillna("").astype(str).str.strip()
        key = np.where(odim != "", odim, "loc:" + act["location"].astype(str)).astype(str)
        uk, inv = np.unique(key, return_inverse=True)
        meta = (act.assign(radar=key, odim=odim).drop_duplicates("radar")
                .set_index("radar").loc[uk].reset_index())
        _OWNERS[year] = (meta, np.where(owner >= 0, inv[np.maximum(owner, 0)], -1))
    return _OWNERS[year]


def climatology_stats(clim_path):
    from src.data.cleaning import hot_mask_from_climatology
    clim = dict(np.load(clim_path))
    years = sorted(int(k[1:5]) for k in clim if k.startswith("y") and k.endswith("_n_valid"))
    rows = []
    for year in years:
        act, owner = owners_year(year)
        nv = clim[f"y{year}_n_valid"].astype(float)
        hot = hot_mask_from_climatology(clim, year)
        hot = np.zeros_like(nv, bool) if hot is None else hot
        ok = nv >= 200
        o = np.where(ok, owner, -1).ravel()
        m = o >= 0
        n_px = np.bincount(o[m], minlength=len(act))
        n_hot = np.bincount(o[m], weights=hot.ravel()[m], minlength=len(act))
        clean = (ok & ~hot).ravel()
        oc = np.where(clean, owner.ravel(), -1)
        mc = oc >= 0
        nvc = np.bincount(oc[mc], weights=nv.ravel()[mc], minlength=len(act))
        r = {"year": year, "radar": act["radar"].values, "odim": act["odim"].values, "location": act["location"].values,
             "country": act["country"].values, "n_px": n_px, "hot_frac": n_hot / np.maximum(n_px, 1)}
        for u in (31, 150, 500):
            n_u = clim[f"y{year}_n_ge{u}"].astype(float).ravel()
            r[f"f{u}"] = np.bincount(oc[mc], weights=n_u[mc], minlength=len(act)) / np.maximum(nvc, 1)
        d = pd.DataFrame(r)
        d["row"], d["col"] = act["row"].values, act["col"].values
        rows.append(d[d["n_px"] >= 500])
    t = pd.concat(rows, ignore_index=True)
    # relative to the 5 nearest other radars of the same year
    for u in (31, 150, 500):
        rel = np.full(len(t), np.nan)
        for year, g in t.groupby("year"):
            xy = g[["row", "col"]].values
            dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
            np.fill_diagonal(dd, np.inf)
            nn = np.argsort(dd, axis=1)[:, :5]
            ref = np.median(g[f"f{u}"].values[nn], axis=1)
            rel[g.index] = g[f"f{u}"].values / np.maximum(ref, 1e-9)
        t[f"rel_f{u}"] = rel
    return t


def flag_stats(flags_dir, tiles_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(flags_dir, "*.csv.gz"))):
        day = os.path.basename(f)[:8]
        tp = os.path.join(tiles_dir, f"{day}.csv.gz")
        if not os.path.exists(tp):
            continue
        fl = pd.read_csv(f, usecols=["timestamp", "row", "col", "amax_row", "amax_col",
                                     "unsup_max", "on_ring"], dtype={"timestamp": str})
        ti = pd.read_csv(tp, usecols=["timestamp", "row", "col", "max"], dtype={"timestamp": str})
        d = fl.merge(ti, on=["timestamp", "row", "col"])
        d = d[d["max"] >= 31]
        if not len(d):
            continue
        d["year"] = int(day[:4])
        # re-attribute from the argmax: the scan writes "" both for radars without an ODIM
        # code and for "no radar within 250 km"
        meta, owner = owners_year(d["year"].iat[0])
        o = owner[d["amax_row"].values, d["amax_col"].values]
        d = d[o >= 0].assign(radar=meta["radar"].values[o[o >= 0]])
        if not len(d):
            continue
        d["unsup"] = d["unsup_max"] >= 31
        parts.append(d.groupby(["year", "radar"]).agg(n_tail=("max", "size"), n_unsup=("unsup", "sum"),
                                                      n_ring=("on_ring", "sum")).reset_index())
    if not parts:
        return None
    s = pd.concat(parts).groupby(["year", "radar"]).sum().reset_index()
    s["unsup_share"] = s["n_unsup"] / s["n_tail"]
    s["ring_share"] = s["n_ring"] / s["n_tail"]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--climatology", default=f"{Q}/clutter_climatology.npz")
    ap.add_argument("--flags_dir", default=f"{Q}/flags")
    ap.add_argument("--tiles_dir", default=f"{Q}/tiles")
    ap.add_argument("--out_dir", default=f"{Q}/radars")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t = climatology_stats(a.climatology)
    fs = flag_stats(a.flags_dir, a.tiles_dir) if os.path.isdir(a.flags_dir) else None
    if fs is not None:
        t = t.merge(fs, on=["year", "radar"], how="left")
        t.loc[t["n_tail"] < 50, ["unsup_share", "ring_share"]] = np.nan   # too few tail tiles
    sig = [s for s in SIGNALS if s in t]
    for s in sig:
        t[f"out_{s}"] = (t[s] >= t[s].quantile(0.95)) & (t[s] > 0)
    t.to_csv(os.path.join(a.out_dir, "radar_year.csv"), index=False)
    g = t.groupby("radar")
    summ = g.size().rename("years").to_frame().join(g[["odim", "location", "country"]].first())
    for s in sig:
        summ[f"n_out_{s}"] = g[f"out_{s}"].sum()
    summ["consistently_bad_on"] = [
        ",".join(s for s in sig if r[f"n_out_{s}"] >= max(3, np.ceil(r["years"] / 2)))
        for _, r in summ.iterrows()]
    summ = summ.reset_index().sort_values("years", ascending=False)
    summ.to_csv(os.path.join(a.out_dir, "radar_summary.csv"), index=False)
    bad = summ[summ["consistently_bad_on"] != ""]
    L = ["# Consistently bad radars (whole archive)", "",
         f"{t['year'].nunique()} years, {t['radar'].nunique()} radars, {len(t)} radar-years. "
         f"Signals: {', '.join(sig)} (outlier = worst 5% of radar-years, and > 0).", "",
         f"**{len(bad)} radars consistently bad** (outlier on the same signal in >= half of "
         "their years, >= 3):", "",
         "| radar | location | country | years | bad on |", "|---|---|---|---|---|"]
    for _, r in bad.iterrows():
        L.append(f"| {r['odim']} | {r['location']} | {r['country']} | {r['years']} | {r['consistently_bad_on']} |")
    open(os.path.join(a.out_dir, "radar_quality.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
