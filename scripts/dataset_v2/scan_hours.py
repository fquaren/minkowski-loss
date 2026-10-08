#!/usr/bin/env python
"""Phase 3 for hourly targets: run the hourly chain on every processing day and describe every
fully covered tile-hour (DECISIONS §21-§23; `src/data/hourly.py`).

A processing day d is the 24 hours ending d 01:00 .. d+1 00:00 (OPERA ACRR convention: hour H =
0.25 x frames H-45 .. H). A tile-hour is described when the 128x128 tile on the stride-128 grid
is valid in all four frames. Rejection is not decided here: the per-rule columns let
`radar_screen.decide_hourly` decide at split time, so thresholds can change without a rescan.
The repairs themselves change the stored values, so their thresholds (`hourly:` in the quality
config) are fixed before this scan runs.

Columns (per tile-hour):
  end, row, col, frames                 hour end and its four frame timestamps (';'-joined)
  max, mean, wet_frac, n_ge*, coarse_max  of the cleaned hourly sum (mm), as `tile_stats`,
                                        plus n_ge5, n_ge20, n_ge50
  raw_max, max_rate, raw_max_rate       raw hourly sum max; max cleaned / raw 15-min rate (mm/h)
  n_hot_kept, n_ceiling, n_footprint, n_ring, n_ray, n_persist, n_unsupported, n_refused
                                        pixel-frames coded by each rule over the four frames
  n_ceiling_max                         largest per-frame ceiling count (v4 rejects at >= 5)
  rep_value, rep_count, rep_excess, max_rep, rep_context
                                        repeated-value test on the worst of the four frames
  unphysical                            max_rate > 500 mm/h after the chain
  owners                                'radar:share;...' for joining radar flags at split time

Per day also <out_dir>/signals/YYYYMMDD.npz: n_peak, n_wet, n_valid (uint8) and the persistence
flags (bool), for audit and galleries.

    python scripts/dataset_v2/scan_hours.py config.yaml --quality configs/quality_v4.yaml \\
        --climatology .../quality_v2/clutter_climatology.npz --ring_mask .../quality_v2/ring_mask.npz \\
        --out_dir .../quality_v4/hourly --workers 6 --skip_existing
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

RULE_BITS = {"n_footprint": 1, "n_ray": 2, "n_ring": 4, "n_unsupported": 8, "n_ceiling": 16,
             "n_refused": 32, "n_hot_kept": 64, "n_persist": 128}


def _ts(t):
    return np.datetime_as_string(t, unit="s").replace("-", "").replace("T", "").replace(":", "")


def scan_day(day, raw_dir, var, clim_path, ring_path, quality, patch, out_dir, rules=None):
    import yaml
    import scan_tiles as st
    from src.data.cleaning import tile_stats
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import owner_shares, repeat_context, repeated_value
    q, ceil = st._quality(quality)
    cfg = dict(q.get("hourly", {}))
    if rules:
        cfg["rules"] = {**cfg.get("rules", {}), **rules}
    hd = HourlyDay(raw_dir, day, var, hot=lambda y: st._hot(clim_path, y),
                   ring=lambda y: st._ring(ring_path, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=cfg).run()
    g = st._geometry(int(day[:4]))
    owners, rows = {}, []
    for h in hd.hours:
        idx = h["idx"]
        H, W = h["sum"].shape
        frames = ";".join(_ts(hd.frames[i]) for i in idx)
        code = hd.code[idx]
        for r0 in range(0, H - patch + 1, patch):
            for c0 in range(0, W - patch + 1, patch):
                if not h["valid"][r0:r0 + patch, c0:c0 + patch].all():
                    continue
                s = h["sum"][r0:r0 + patch, c0:c0 + patch]
                st_ = tile_stats(s)
                st_.update({"end": _ts(h["end"]), "row": r0, "col": c0, "frames": frames,
                            "n_ge5": int((s >= 5).sum()), "n_ge20": int((s >= 20).sum()),
                            "n_ge50": int((s >= 50).sum()),
                            "raw_max": float(h["raw_sum"][r0:r0 + patch, c0:c0 + patch].max()),
                            "max_rate": float(h["max_rate"][r0:r0 + patch, c0:c0 + patch].max()),
                            "raw_max_rate": float(np.nanmax(hd.raw[idx][:, r0:r0 + patch, c0:c0 + patch]))})
                cd = code[:, r0:r0 + patch, c0:c0 + patch]
                for k, b in RULE_BITS.items():
                    st_[k] = int(((cd & b) > 0).sum())
                st_["n_ceiling_max"] = int(((cd & 16) > 0).sum(axis=(1, 2)).max())
                best = None
                for i in idx:
                    rv = repeated_value(hd.fin[i][r0:r0 + patch, c0:c0 + patch], q["repeat"]["u"])
                    if best is None or (rv["rep_count"], rv["rep_excess"]) > (best[0]["rep_count"], best[0]["rep_excess"]):
                        best = (rv, i)
                rv, i = best
                rv["rep_context"] = repeat_context(hd.fin[i][r0:r0 + patch, c0:c0 + patch], rv["rep_value"]) \
                    if rv["rep_count"] >= 10 else float("nan")
                st_.update(rv)
                st_["unphysical"] = int(st_["max_rate"] > 500.0)
                if (r0, c0) not in owners:
                    owners[(r0, c0)] = owner_shares(g, r0, c0, patch)
                st_["owners"] = owners[(r0, c0)]
                rows.append(st_)
    cols = ["end", "row", "col", "frames", "max", "mean", "wet_frac", "n_ge1", "n_ge5", "n_ge10",
            "n_ge20", "n_ge31", "n_ge50", "n_ge53", "n_ge89", "n_ge150", "coarse_max", "raw_max",
            "max_rate", "raw_max_rate", *RULE_BITS, "n_ceiling_max", "rep_value", "rep_count",
            "rep_excess", "max_rep", "rep_context", "unphysical", "owners"]
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.join(out_dir, "hours"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)
    out = os.path.join(out_dir, "hours", f"{day}.csv.gz")
    df.to_csv(out + ".part", index=False, compression="gzip", float_format="%.7g")
    os.replace(out + ".part", out)
    s = hd.signals
    if s:
        sp = os.path.join(out_dir, "signals", f"{day}.npz")
        np.savez_compressed(sp + ".part.npz", **{k: (np.clip(v, 0, 255).astype(np.uint8) if v.dtype != bool else v)
                                               for k, v in s.items()})
        os.replace(sp + ".part.npz", sp)
    return day, len(df), len(hd.hours)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--quality", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--climatology", required=True)
    ap.add_argument("--ring_mask", required=True)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--days_file", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    days = sorted(os.path.basename(d) for d in glob.glob(os.path.join(raw_dir, "[0-9]" * 8))
                  if os.path.exists(os.path.join(d, ".zmetadata")))
    want = set(args.days or [])
    if args.days_file:
        want |= {l.strip() for l in open(args.days_file) if l.strip()}
    if want:
        days = [d for d in days if d in want]
    todo = [d for d in days if not (args.skip_existing and os.path.exists(
        os.path.join(args.out_dir, "hours", d + ".csv.gz")))]
    print(f"[hours] {len(days)} days, {len(todo)} to scan -> {args.out_dir}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_day, d, raw_dir, cfg["PRECIP_VAR_NAME"], args.climatology,
                          args.ring_mask, args.quality, cfg["PATCH_SIZE"], args.out_dir): d for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                day, n, nh = f.result()
                if k % 10 == 0 or k == len(futs) or k <= 3:
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} days ({day}: {nh} hours, {n} tile-hours), "
                          f"{el / k:.0f} s/day, ~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)
            except Exception as e:                                         # noqa: BLE001
                print(f"  ! {futs[f]}: {type(e).__name__}: {e}", flush=True)
    print(f"[hours] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
