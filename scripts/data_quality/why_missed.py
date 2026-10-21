#!/usr/bin/env python
"""Why the hourly screen misses a noise pattern: rule by rule, on the pixels of a given mask.

Replays the v4 hourly chain (`HourlyDay`, as `build_store_hourly.py`, no advection) on the given days
and, for every raw pixel-frame >= U mm/h inside the mask (default U = 31), reports:
  - what happened: changed by `clean_frame` (hot fill / spike), by each coded repair, or left;
  - for the pixel-frames left at >= U, which condition of each rule they fail:
      spike        z >= 10 and all 8 neighbours < 10% of z               (isolated single pixel)
      persistence  local peak (>= 1 mm/h and >= 2x its 5x5 median) in >= p1_min of the day's frames
      burst        one frame >= 100 mm/h carrying > 80% of the pixel-hour, hour's 5x5 median < 1 mm
      unsupported  cell >= 10 mm/h with no echo >= 1 mm/h within 30 km in the previous or next frame
      footprint    connected to a core > 500 mm/h
  - the noise's own structure: per-frame share of the mask >= U, connected-component sizes of >= U
    pixels, the commonest values (a reflectivity ceiling shows as one repeated value), and the rain
    around it (share of the mask's 50 km outer ring >= 1 mm/h).

    python scripts/data_quality/why_missed.py --days 20230310 20230722 --mask <mask.npy> --label rsjas_2023
      (-> notes/data_analysis_v4/why_missed_<label>.md)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
NOTES = f"{REPO}/notes/data_analysis_v4"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--mask", required=True, help=".npy bool (2200, 1900)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--u", type=float, default=31.0)
    a = ap.parse_args()
    import scan_tiles as st
    from src.data import cleaning as C
    from src.data.hourly import HourlyDay, local_peaks, median5
    from scripts.data_quality._md import md_table
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    cfg = {**q.get("hourly", {}), "advection": None}
    mask = np.load(a.mask)
    ring = ndimage.binary_dilation(mask, iterations=25) & ~mask          # 50 km outer ring
    bits = [("footprint", C.REPAIR_FOOTPRINT), ("ceiling", C.REPAIR_CEILING), ("ring", C.REPAIR_RING),
            ("ray", C.REPAIR_RAY), ("persistence", C.REPAIR_PERSIST), ("unsupported", C.REPAIR_UNSUPPORTED),
            ("burst", C.REPAIR_BURST), ("stationary", C.REPAIR_STATIONARY),
            ("refused (too large to repair, left)", C.REPAIR_REFUSED)]
    out, per_frame, comps, vals = [], [], [], []
    for day in a.days:
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(f"{D}/quality_v2/clutter_climatology.npz", y),
                       ring=lambda y: st._ring(f"{D}/quality_v2/ring_mask.npz", y), sites_rc=st._sites(), ceilings=ceil,
                       cfg=cfg).run()
        hot = st._hot(f"{D}/quality_v2/clutter_climatology.npz", int(day[:4]))
        d0 = np.datetime64(pd.Timestamp(day))
        proc = [(t, ts) for t, ts in enumerate(hd.frames) if d0 < ts <= d0 + np.timedelta64(1, "D")]
        # persistence signal of the day: local-peak counts on the chain's frames before persistence
        n_peak = hd.signals["n_peak"]
        res = {"day": day, "raw_ge_u": 0, "changed_clean_frame": 0, **{b: 0 for b, _ in bits}, "left_ge_u": 0,
               "left: spike-isolated": 0, "left: n_peak >= p1_min": 0, "left: burst share & rate": 0,
               "left: burst iso median": 0, "left: >= 10 with no echo nearby (prev/next)": 0, "left: raw > 500": 0}
        for t, ts in proc:
            r = hd.raw[t]; f = hd.fin[t]; code = hd.code[t]
            hit = mask & np.isfinite(r) & (r >= a.u)
            nh = int(hit.sum())
            wet_ring = float((np.nan_to_num(r[ring]) >= 1).mean())
            per_frame.append({"day": day, "time": pd.Timestamp(ts).strftime("%H:%M"), "share_ge_u": nh / mask.sum(),
                              "ring_wet_ge1": wet_ring})
            if not nh:
                continue
            res["raw_ge_u"] += nh
            cf = C.clean_frame(r, hot, hot_fallback=cfg.get("hot_fallback", "zero"))[0]
            res["changed_clean_frame"] += int((hit & ~np.isclose(np.nan_to_num(cf), np.nan_to_num(r))).sum())
            for b, v in bits:
                res[b] += int((hit & ((code & v) > 0)).sum())
            left = hit & (np.nan_to_num(f) >= a.u)
            res["left_ge_u"] += int(left.sum())
            z = np.nan_to_num(r)
            nb = ndimage.maximum_filter(z, footprint=C._RING, mode="constant")
            res["left: spike-isolated"] += int((left & (z >= C.SPIKE_MIN) & (nb < C.SPIKE_RATIO * z)).sum())
            res["left: n_peak >= p1_min"] += int((left & (n_peak >= cfg["p1_min"])).sum())
            fz = np.nan_to_num(f)
            res["left: raw > 500"] += int((left & (z > C.UNPHYSICAL)).sum())
            res["left: final > 500"] = res.get("left: final > 500", 0) + int((left & (fz > C.UNPHYSICAL)).sum())
            res["left: final > 500, coded refused"] = res.get("left: final > 500, coded refused", 0) + int(
                (left & (fz > C.UNPHYSICAL) & ((code & C.REPAIR_REFUSED) > 0)).sum())
            res["left: final 31-150"] = res.get("left: final 31-150", 0) + int((left & (fz < 150)).sum())
            if ceil is not None:
                cm = ceil.mask(np.where(np.isfinite(r), r, np.nan).astype(np.float32), int(day[:4]), None)
                res["left: on the ceiling mask"] = res.get("left: on the ceiling mask", 0) + int((left & cm).sum())
                res["left: on the ceiling mask, coded refused"] = res.get("left: on the ceiling mask, coded refused", 0) + int(
                    (left & cm & ((code & C.REPAIR_REFUSED) > 0)).sum())
            # temporal support, as repair_unsupported: echo >= 1 within 30 km in the previous / next frame
            sup = np.zeros(z.shape, bool)
            for tt in (t - 1, t + 1):
                if 0 <= tt < len(hd.frames):
                    sup |= ndimage.maximum_filter(np.nan_to_num(hd.raw[tt]) >= C.SUPPORT_MIN, size=2 * C.SUPPORT_RADIUS_PX + 1)
            res["left: >= 10 with no echo nearby (prev/next)"] += int((left & ~sup).sum())
            lab, n = ndimage.label(z >= a.u, structure=C._EIGHT)
            ids = np.unique(lab[hit]); ids = ids[ids > 0]
            if len(ids):
                comps += list(ndimage.sum(np.ones_like(z), lab, ids).astype(int))
            vals += list(np.round(np.nan_to_num(f)[left], 2))
        # burst conditions per hour on the left pixels
        for h in hd.hours:
            jj = h["idx"]
            if not (d0 < hd.frames[jj[-1]] <= d0 + np.timedelta64(1, "D")):
                continue
            F4 = np.nan_to_num(hd.raw[jj])
            fmax, tot = F4.max(0), F4.sum(0)
            leftany = mask & (np.nan_to_num(hd.fin[jj]).max(0) >= a.u) & (fmax >= a.u)
            cand = leftany & (fmax >= cfg["burst"]["min_rate"]) & (fmax > cfg["burst"]["min_share"] * np.maximum(tot, 1e-6))
            res["left: burst share & rate"] += int(cand.sum())
            res["left: burst iso median"] += int((cand & (median5(0.25 * tot, cand) < cfg["burst"]["iso_median"])).sum())
        out.append(res)
        del hd
        print(res, flush=True)
    t = pd.DataFrame(out)
    pf = pd.DataFrame(per_frame)
    cs = pd.Series(comps)
    vc = pd.Series(vals).value_counts().head(8)
    L = [f"# Why the screen misses it: {a.label}", "",
         f"Days {', '.join(a.days)}; mask {int(mask.sum()):,} px; pixel-frames with raw >= {a.u:g} mm/h inside it. "
         "Replay of the v4 hourly chain (`HourlyDay`, no advection). Columns 'left: ...' count the pixel-frames "
         f"still >= {a.u:g} after the chain that meet that rule's condition (pixel-hours for the burst columns).", "",
         md_table(t.set_index("day").T.reset_index().rename(columns={"index": "count"}), index=False), "",
         "## Structure of the noise", "",
         f"- Share of the mask >= {a.u:g} mm/h per frame: median {pf.share_ge_u.median():.4f}, 90th percentile "
         f"{pf.share_ge_u.quantile(.9):.4f}, frames with any: {(pf.share_ge_u > 0).mean():.0%}.",
         f"- Rain around it (50 km ring outside the mask, share >= 1 mm/h): median {pf.ring_wet_ge1.median():.3f}; "
         f"in frames where the mask has > 0.5% of pixels >= {a.u:g}: "
         f"{pf[pf.share_ge_u > 0.005].ring_wet_ge1.median() if (pf.share_ge_u > 0.005).any() else float('nan'):.3f}.",
         f"- Connected components of >= {a.u:g} px touching the mask: {len(cs):,}; size median {cs.median() if len(cs) else 0:.0f} px, "
         f"90th percentile {cs.quantile(.9) if len(cs) else 0:.0f}, single pixels {(cs == 1).mean() if len(cs) else 0:.0%}.",
         "- Commonest final values left >= the level (mm/h): " + ", ".join(f"{k:g} ({v:,})" for k, v in vc.items()), ""]
    os.makedirs(NOTES, exist_ok=True)
    open(f"{NOTES}/why_missed_{a.label}.md", "w").write("\n".join(L) + "\n")
    pf.to_csv(f"{NOTES}/why_missed_{a.label}_frames.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
