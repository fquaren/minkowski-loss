#!/usr/bin/env python
"""Shifted 128 x 128 windows for the event test sets, where the stride-128 grid has no
fully covered tile (notes/events.md, option A).

The v2 tiles sit on a fixed stride-128 grid and must be fully covered, so an event whose
grid tiles each miss a few pixels (Emilia-Romagna, May 2023: 18 and 272 permanently missing
pixels) gets no event tiles at all. For every catalogued event and every frame of its window
(not the buffer), this adds windows at any offset on a stride-16 lattice that:
  - overlap the event box (same overlap rule as `make_splits.event_rows`),
  - are fully covered at that frame and hold rain (cleaned max >= --min_max),
  - do not overlap a fully covered grid tile (those are already event tiles) or a window
    already chosen: greedy, wettest first (pixels >= 1 mm/h).
Test-only: the rows land in `events_test` / `events_nimbus`, never in training.

Frames are cleaned with the same `DayCleaner` chain as the scan, and each row has the
scan-table columns plus `event`, so `make_splits.py --event_windows` can append them.

Output: <out_dir>/event_windows/YYYYMMDD.csv.gz

    python scripts/dataset_v2/scan_event_windows.py config.yaml \
        --climatology .../quality_v2/clutter_climatology.npz \
        --ring_mask .../quality_v2/ring_mask.npz --out_dir .../quality_v2 --workers 6
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan_tiles as st  # noqa: E402  (shared mask / site caches)

STRIDE = 16


def box_pixels(bbox, H=2200, W=1900):
    """Pixel extent (r0, r1, c0, c1) of a lat/lon box (corners and edge midpoints)."""
    from src.data import geo
    la0, la1, lo0, lo1 = bbox
    lons = np.array([lo0, lo1, lo0, lo1, (lo0 + lo1) / 2, (lo0 + lo1) / 2, lo0, lo1])
    lats = np.array([la0, la0, la1, la1, la0, la1, (la0 + la1) / 2, (la0 + la1) / 2])
    r, c = geo.lonlat_to_rowcol(lons, lats)
    r, c = np.asarray(r), np.asarray(c)
    return (int(max(np.floor(r.min()), 0)), int(min(np.ceil(r.max()), H - 1)),
            int(max(np.floor(c.min()), 0)), int(min(np.ceil(c.max()), W - 1)))


def _integral(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    s[1:, 1:] = a.astype(np.int64).cumsum(0).cumsum(1)
    return s


def _win_sum(s, r, c, P):
    return s[r + P, c + P] - s[r, c + P] - s[r + P, c] + s[r, c]


def windows_day(day, raw_dir, var, events, clim_path, ring_path, patch, min_max, out_path):
    from src.data.cleaning import UNPHYSICAL, ray_flag, tile_stats
    from src.data.day_cleaner import DayCleaner
    dc = DayCleaner(raw_dir, day, var, hot=lambda y: st._hot(clim_path, y),
                    ring=lambda y: st._ring(ring_path, y), sites_rc=st._sites())
    has_q = "QIND" in dc.ds
    H, W = dc.ds.sizes["y"], dc.ds.sizes["x"]
    rows = []
    for k in range(dc.T):
        raw = dc.raw(k)
        fin = np.isfinite(raw)
        miss = _integral(~fin)
        cln = None
        q = None
        for e in events:
            r0, r1, c0, c1 = e["_px"]
            occ = np.zeros((H, W), bool)
            # fully covered grid tiles overlapping the box are event tiles already
            for g0 in range((max(r0 - patch + 1, 0) // patch) * patch, r1 + 1, patch):
                for h0 in range((max(c0 - patch + 1, 0) // patch) * patch, c1 + 1, patch):
                    if g0 + patch <= H and h0 + patch <= W and _win_sum(miss, g0, h0, patch) == 0:
                        occ[g0:g0 + patch, h0:h0 + patch] = True
            cand = []
            for y in range(max(r0 - patch + 1, 0) // STRIDE * STRIDE, min(r1, H - patch) + 1, STRIDE):
                for x in range(max(c0 - patch + 1, 0) // STRIDE * STRIDE, min(c1, W - patch) + 1, STRIDE):
                    if _win_sum(miss, y, x, patch) == 0:
                        cand.append((y, x))
            if not cand:
                continue
            if cln is None:
                cln, rcode = dc.clean(k)
                wet = _integral(np.nan_to_num(cln, nan=0.0) >= 1.0)
            cand.sort(key=lambda p: -_win_sum(wet, p[0], p[1], patch))
            ts = (np.datetime_as_string(dc.times[k], unit="s").replace("-", "").replace("T", "")
                  .replace(":", ""))
            for y, x in cand:
                if occ[y:y + patch, x:x + patch].any() or _win_sum(wet, y, x, patch) == 0:
                    continue
                ct = cln[y:y + patch, x:x + patch]
                s = tile_stats(ct)
                if s["max"] < min_max:
                    continue
                occ[y:y + patch, x:x + patch] = True
                rt = raw[y:y + patch, x:x + patch]
                zr = np.where(rt < 0.1, 0.0, rt)
                if has_q and q is None:
                    q = dc.ds["QIND"].isel(time=k).values
                qt = q[y:y + patch, x:x + patch] if has_q else None
                w_ = (ct >= 0.1) & np.isfinite(qt) if has_q else None
                s.update({"timestamp": ts, "row": y, "col": x, "raw_max": float(rt.max()),
                          "n_fixed": int((np.abs(ct - zr) > 1e-6).sum()),
                          "ray": int(s["n_ge1"] >= 30 and ray_flag(ct, y, x, st._sites())),
                          "unphysical": int(s["max"] > UNPHYSICAL),
                          "q_mean_wet": float(qt[w_].mean()) if has_q and w_.any() else np.nan,
                          "has_qind": int(has_q),
                          "n_repaired": int((rcode[y:y + patch, x:x + patch] > 0).sum()),
                          "event": e["id"]})
                rows.append(s)
    dc.close()
    cols = ["timestamp", "row", "col", "max", "mean", "wet_frac", "n_ge1", "n_ge10", "n_ge31",
            "n_ge53", "n_ge89", "n_ge150", "coarse_max", "raw_max", "n_fixed", "ray",
            "unphysical", "q_mean_wet", "has_qind", "n_repaired", "event"]
    df = pd.DataFrame(rows, columns=cols)
    tmp = out_path + ".part"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.7g")
    os.replace(tmp, out_path)
    return day, len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--events", default="configs/prominent_events.yaml")
    ap.add_argument("--climatology", required=True)
    ap.add_argument("--ring_mask", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_max", type=float, default=1.0, help="minimum cleaned window max (mm/h)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(a.config)
    raw_dir = cfg["RAW_OPERA_DATA_DIR"]
    events = [e for e in yaml.safe_load(open(a.events))["events"] if e.get("available", True)]
    by_day = {}
    for e in events:
        e["_px"] = box_pixels(e["bbox"])
        for d in pd.date_range(str(e["start"]), str(e["end"]), freq="D").strftime("%Y%m%d"):
            if os.path.exists(os.path.join(raw_dir, d, ".zmetadata")):
                by_day.setdefault(d, []).append(e)
    out = os.path.join(a.out_dir, "event_windows")
    os.makedirs(out, exist_ok=True)
    print(f"[windows] {len(events)} events over {len(by_day)} days -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(windows_day, d, raw_dir, cfg["PRECIP_VAR_NAME"], ev, a.climatology,
                          a.ring_mask, cfg["PATCH_SIZE"], a.min_max,
                          os.path.join(out, f"{d}.csv.gz")): d for d, ev in sorted(by_day.items())}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                d, n = f.result()
                print(f"  {k}/{len(futs)} {d}: {n} windows", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {futs[f]}: {type(e).__name__}: {e}", flush=True)
    print(f"[windows] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
