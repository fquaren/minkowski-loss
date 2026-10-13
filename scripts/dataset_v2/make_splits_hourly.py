#!/usr/bin/env python
"""Leak-free splits of the hourly dataset (target = 1-h accumulation, DECISIONS §21-§27), from
the tile-hour tables of `scan_hours.py`. Metadata only; `build_store_hourly.py` writes the patches.

Same design as `make_splits.py` (15-min, v2/v3), whose functions it reuses:
  - whole processing days per split (a processing day is the 24 hours ending d 01:00 .. d+1
    00:00, so the hour ending at midnight belongs to the day before); validation and test
    are whole ISO weeks drawn per calendar month across years; catalogued events (widened by
    --event_buffer days) are test; --buffer days are dropped wherever two splits meet;
  - NIMBUS days (from --nimbus_start) form their own split `nimbus`;
  - each split is sampled by intensity stratum of the cleaned tile-hour max, every row
    carrying its inclusion weight; event tile-hours of test / nimbus are all kept.

**Rejection** happens here, not in the scan, so thresholds can change without a rescan:
  - `unphysical`: a 15-min rate > 500 mm/h left after the chain;
  - `radar_screen.decide_hourly` (DECISIONS §23): >= 5 ceiling pixels in a frame, a repeated
    value outside rain, refused repairs, a flagged radar-frame owning >= the reject share
    (`frame_flags.csv.gz` of `radar_screen_v4.py frames`, rule 3 with the temporal check),
    and radar-day exclusions (`--day_excl`, rule 5).
Rejection is reported per raw-max bin and per reason (a rule that removes more of the tail
than of the bulk is removing storms, DECISIONS §17).

**Strata** are on the hourly sum (mm). The edges default to the 15-min ones and
--tail to 31 mm; the hourly tail threshold is still to be re-derived (EXPERIMENTS §5).

Outputs under --out_dir: full_{split}.txt (ts,y,x,max; ts = hour end YYYYmmddHHMMSS, in store
order), full_{split}_info.csv.gz, light_* / extremes_* / events_* subsets (5 columns, last =
store row), days.csv, rejected_by_reason.csv, report.md.

    python scripts/dataset_v2/make_splits_hourly.py config.yaml \\
        --hours_dir .../quality_v4/hourly/hours --quality configs/quality_v4.yaml \\
        --out_dir .../OPERA/v4_hourly --budget 3000000 --workers 8
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
from make_splits import (DEFAULT_SHARES, NIMBUS_START, PRIORITY, STRATA, assign_days,  # noqa: E402
                         event_rows, sample_split, write_meta)

KEEP = ["end", "row", "col", "max", "mean", "wet_frac", "n_ge5", "n_ge20", "n_ge31", "n_ge50",
        "raw_max", "max_rate", "raw_max_rate", "n_persist", "n_burst", "n_unsupported", "unphysical"]
REASONS = ["unphysical", "rej_ceiling", "rej_repeat", "rej_refused", "rej_radar_frame", "rej_radar_day"]
_W = {}


def _init(q, frame_flags, day_excl):
    _W.update(q=q, ff=frame_flags, de=day_excl)


def load_day(path):
    """One day's tile-hours with the rejection columns, reduced to what the split needs."""
    from src.data.radar_screen import decide_hourly
    t = pd.read_csv(path, dtype={"end": str, "frames": str, "owners": str})
    if not len(t):
        return None
    if "n_burst" not in t:                     # scans before 2026-10-09
        t["n_burst"] = 0
    d = decide_hourly(t, _W["q"], frame_flags=_W["ff"], day_excl=_W["de"])
    out = t[KEEP].copy()
    out["unphysical"] = out["unphysical"].astype(bool)
    for r in REASONS[1:]:
        out[r] = d[r].values
    out["rejected"] = out["unphysical"] | d["rej_v4"].values
    out["day"] = pd.Timestamp(os.path.basename(path)[:8])   # the processing day
    return out.rename(columns={"end": "timestamp"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--hours_dir", required=True, help="<scan_hours out_dir>/hours")
    ap.add_argument("--quality", default="configs/quality_v4.yaml")
    ap.add_argument("--frame_flags", default=None,
                    help="frame_flags.csv.gz (default: <quality data_dir>/frame_flags.csv.gz)")
    ap.add_argument("--day_excl", default=None, help="CSV radar,day (YYYYMMDD) of excluded radar-days")
    ap.add_argument("--events", default="configs/prominent_events.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budget", type=int, default=3_000_000)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--buffer", type=int, default=1)
    ap.add_argument("--event_buffer", type=int, default=1)
    ap.add_argument("--light_frac", type=float, default=0.25)
    ap.add_argument("--tail", type=float, default=31.0, help="mm in 1 h; provisional")
    ap.add_argument("--shares", default=None)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--nimbus_start", default=NIMBUS_START)
    ap.add_argument("--days_file", default=None, help="restrict to these processing days (tests)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    shares = dict(DEFAULT_SHARES)
    if args.shares:
        shares.update({k: float(v) for k, v in (kv.split("=") for kv in args.shares.split(","))})
    assert abs(sum(shares.values()) - 1) < 1e-6, f"shares must sum to 1: {shares}"
    os.makedirs(args.out_dir, exist_ok=True)

    q = yaml.safe_load(open(args.quality))
    ffp = args.frame_flags or os.path.join(q["data_dir"], "frame_flags.csv.gz")
    ff = pd.read_csv(ffp, dtype=str)
    frame_flags = set(zip(ff["radar"], ff["timestamp"]))
    day_excl = set()
    if args.day_excl:
        de = pd.read_csv(args.day_excl, dtype=str)
        day_excl = set(zip(de["radar"], de["day"]))
    files = sorted(glob.glob(os.path.join(args.hours_dir, "[0-9]" * 8 + ".csv.gz")))
    if args.days_file:
        want = {l.strip() for l in open(args.days_file) if l.strip()}
        files = [f for f in files if os.path.basename(f)[:8] in want]
    print(f"[splits-hourly] {len(files)} days; {len(frame_flags):,} flagged radar-frames "
          f"({ffp}); {len(day_excl):,} excluded radar-days", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(q, frame_flags, day_excl)) as ex:
        parts = [p for p in ex.map(load_day, files, chunksize=8) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    events = [e for e in yaml.safe_load(open(args.events))["events"] if e.get("available", True)]
    rep = ["# Hourly dataset report", "",
           f"{len(df):,} fully covered tile-hours over {df['day'].nunique()} processing days; "
           f"budget {args.budget:,}; shares {shares}; val/test {args.val_frac}/{args.test_frac}; "
           f"buffer {args.buffer} d; event buffer {args.event_buffer} d; tail {args.tail} mm; "
           f"seed {args.seed}", "",
           f"Rejection inputs: quality `{args.quality}`, frame flags `{ffp}` ({len(frame_flags):,}), "
           f"radar-day exclusions `{args.day_excl}` ({len(day_excl):,}).", ""]

    # ---- rejection by raw max bin and by reason (a tile-hour can have several reasons)
    bins = [0, 0.1, 1, 5, 10, 20, 31, 50, 100, 200, np.inf]
    lab = pd.cut(df["raw_max"], bins, right=False)
    g = df.assign(bin=lab).groupby("bin", observed=True)
    tab = pd.DataFrame({"tile_hours": g.size(), "rejected": g["rejected"].mean(),
                        **{r: g[r].mean() for r in REASONS}})
    tab.to_csv(os.path.join(args.out_dir, "rejected_by_reason.csv"))
    rep += ["## Rejection by raw tile-hour max (mm)", "",
            "| raw max | tile-hours | rejected | " + " | ".join(REASONS) + " |",
            "|---" * (3 + len(REASONS)) + "|"]
    rep += [f"| {b} | {int(r.tile_hours):,} | {r.rejected:.2%} | "
            + " | ".join(f"{r[x]:.2%}" for x in REASONS) + " |" for b, r in tab.iterrows()]
    rep.append("")
    df = df[~df["rejected"]].reset_index(drop=True)

    # ---- day splits and leak checks
    split, reason, is_event = assign_days(df["day"].unique(), events, args)
    pd.DataFrame({"day": split.index.strftime("%Y-%m-%d"), "split": split.values,
                  "reason": reason.values, "event_window": is_event.values}).to_csv(
        os.path.join(args.out_dir, "days.csv"), index=False)
    df["split"] = split.reindex(df["day"]).values
    rep += ["## Days", "", " | ".join(f"{k}: {v}" for k, v in split.value_counts().items()), ""]
    df["event"] = event_rows(df, events)
    kept = df[df["split"] != "drop"]
    assert (kept.groupby("timestamp")["split"].nunique() == 1).all(), "a timestamp in two splits"
    sdays = {s: np.sort(kept.loc[kept["split"] == s, "day"].unique()) for s in PRIORITY}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test"), ("train", "nimbus"),
                   ("val", "nimbus"), ("test", "nimbus")):
        if len(sdays[a_]) and len(sdays[b_]):
            gap = np.min(np.abs(sdays[a_][:, None] - sdays[b_][None, :])).astype("timedelta64[D]").astype(int)
            rep.append(f"min gap {a_} <-> {b_}: {gap} day(s)")
            assert gap >= args.buffer + 1, f"{a_} within {gap} d of {b_}"
    rep.append("")

    # ---- stratified sampling per split (strata on the cleaned hourly sum, mm)
    n_by = kept["split"].value_counts()
    out = {}
    info_cols = ["timestamp", "row", "col", "max", "raw_max", "max_rate", "raw_max_rate", "mean",
                 "wet_frac", "n_ge20", "n_ge50", "n_persist", "n_burst", "stratum", "weight",
                 "event", "era"]
    for sp in PRIORITY:
        d = kept[kept["split"] == sp].reset_index(drop=True)
        b = int(round(args.budget * n_by.get(sp, 0) / max(n_by.sum(), 1)))
        forced = (d["event"].values != "") if sp in ("test", "nimbus") else np.zeros(len(d), bool)
        idx, w, s = sample_split(d, b, shares, args.seed + PRIORITY[sp], forced)
        sel = d.iloc[idx].copy()
        sel["weight"], sel["stratum"] = w, s
        sel["era"] = np.where(sel["day"] >= pd.Timestamp(args.nimbus_start), "NIMBUS", "ODYSSEY")
        sel = sel.sort_values(["timestamp", "row", "col"]).reset_index(drop=True)
        out[sp] = sel
        write_meta(os.path.join(args.out_dir, f"full_{sp}.txt"), sel)
        sel[info_cols].to_csv(os.path.join(args.out_dir, f"full_{sp}_info.csv.gz"), index=True,
                              index_label="store_row", compression="gzip", float_format="%.6g")
        cnt = sel["stratum"].value_counts()
        rep.append(f"full_{sp}: {len(sel):,} tile-hours (pool {len(d):,}); "
                   + ", ".join(f"{k} {cnt.get(k, 0):,}" for k, _, _ in STRATA))
    rep.append("")

    # ---- subsets
    rng = np.random.default_rng(args.seed + 7)
    for sp, sel in out.items():
        rows, st = np.arange(len(sel)), sel["stratum"].values
        light = np.sort(np.concatenate([rng.choice(rows[st == k], size=int(round(args.light_frac * (st == k).sum())),
                                                   replace=False) for k, _, _ in STRATA if (st == k).any()]
                                       or [np.zeros(0, np.int64)]))
        write_meta(os.path.join(args.out_dir, f"light_{sp}.txt"), sel.iloc[light], light)
        ext = rows[sel["max"].values >= args.tail]
        write_meta(os.path.join(args.out_dir, f"extremes_{sp}.txt"), sel.iloc[ext], ext)
        rep.append(f"light_{sp}: {len(light):,} | extremes_{sp}: {len(ext):,}")
        if sp in ("test", "nimbus"):
            evr = rows[sel["event"].values != ""]
            write_meta(os.path.join(args.out_dir, f"events_{sp}.txt"), sel.iloc[evr], evr)
            rep.append(f"events_{sp}: {len(evr):,}")
    rep += ["", "## Events", "", "| event | split | tile-hours | >= tail | max (mm) |", "|---|---|---|---|---|"]
    for e in events:
        for sp in ("test", "nimbus"):
            m = out[sp]["event"].values == e["id"]
            if m.any():
                rep.append(f"| {e['id']} | {sp} | {int(m.sum()):,} | {int((out[sp]['max'].values[m] >= args.tail).sum()):,} "
                           f"| {out[sp]['max'].values[m].max():.1f} |")
    open(os.path.join(args.out_dir, "report.md"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
