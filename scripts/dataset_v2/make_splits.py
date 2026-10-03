#!/usr/bin/env python
"""Leak-free splits and the v2 dataset family, from the tile scan.

Inputs: the per-day tile tables of `scan_tiles.py` and `configs/prominent_events.yaml`.
Outputs under `--out_dir` (metadata only; `build_store.py` then writes the patches):

  full_{train,val,test}.txt        4 columns ts,y,x,max, in store order (the store groups)
  light_{train,val,test}.txt       5 columns ts,y,x,max,store_row -> rows of full_<split>
  extremes_{train,val,test}.txt    5 columns, the tail (cleaned max >= --tail) of full
  events_test.txt                  5 columns, every tile of every catalogued event
  full_<split>_info.csv.gz         per store row: stratum, sampling weight, event id, ...
  days.csv                         each day's split, or why it was dropped
  report.md                        counts, rejection by intensity, per-event coverage

**Leakage.** Splits are assigned by whole days, never by patch, so a timestamp and all its
neighbouring tiles always share a split (DECISIONS §15):
  - validation and test are whole ISO weeks, drawn per calendar month across years (so each
    split sees every season and every year) with a fixed seed;
  - every catalogued event window, widened by --event_buffer days, is test, together with
    the ISO weeks it touches;
  - wherever two different splits meet, --buffer days are dropped from the lower-priority
    side (train < val < test), so no train day is within 24 h of an evaluation day.

**Cleaning.** Tiles flagged `unphysical` (cleaned max > 500 mm/h) or `ray` (RLAN ray
pointing at a radar) are rejected before sampling; the rates of both are reported per
intensity bin, because a rule that removes more of the tail than the bulk is removing storms
(DECISIONS §17).

**Sampling.** Available tiles are far more than the budget, and 15-minute neighbours are
highly redundant, so each split is sampled by intensity stratum of the cleaned tile max,
uniformly at random within a stratum. Every row records its inclusion weight
(N_stratum / n_stratum), so distribution statistics can be reweighted to the population.
Event tiles are never subsampled: they are all kept (weight 1) and the rest of their stratum
is sampled around them.

**Eras.** Days from --nimbus_start on are the NIMBUS product and form their own split,
`nimbus` (store group of the same name): train/val/test are drawn from ODYSSEY days only, so
the main results use one product, and the NIMBUS split measures generalisation to a changed
product. It is sampled like the others, its events go to `events_nimbus.txt`, and the buffer
rule treats it as the highest-priority split, so the last ODYSSEY day before the switch is
dropped.

**Subsets.** light = a stratified --light_frac of each full split (the same strata shares,
so the same distribution, at a fraction of the cost); extremes = full rows with cleaned max
>= --tail; events_test = full test rows inside an event's box and window.

    python scripts/dataset_v2/make_splits.py config.yaml \
        --tiles_dir .../quality_v2/tiles --events configs/prominent_events.yaml \
        --out_dir .../OPERA/v2 --budget 3000000
"""

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

STRATA = [("dry", 0.0, 0.1), ("light", 0.1, 1.0), ("moderate", 1.0, 10.0),
          ("heavy", 10.0, 31.0), ("tail", 31.0, np.inf)]
DEFAULT_SHARES = {"dry": 0.08, "light": 0.12, "moderate": 0.25, "heavy": 0.25, "tail": 0.30}
PRIORITY = {"train": 0, "val": 1, "test": 2, "nimbus": 3}
# OPERA changed production chain on 2024-07-05 (archive product QIND_RATE -> RATE,
# ODYSSEY -> NIMBUS). train/val/test are ODYSSEY only; NIMBUS days form a separate
# product-shift evaluation split (option 1, 2026-09-30).
NIMBUS_START = "2024-07-05"
COLS = ["timestamp", "row", "col", "max", "wet_frac", "n_ge31", "raw_max", "n_fixed",
        "ray", "unphysical", "has_qind"]


# ----------------------------------------------------------------------------------
def load_tiles(tiles_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(tiles_dir, "*.csv.gz"))):
        d = pd.read_csv(f, usecols=COLS, dtype={"timestamp": str})
        if not len(d):
            continue
        d = d.astype({"row": np.int16, "col": np.int16, "max": np.float32,
                      "wet_frac": np.float32, "n_ge31": np.int32, "raw_max": np.float32,
                      "n_fixed": np.int32, "ray": np.int8, "unphysical": np.int8,
                      "has_qind": np.int8})
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["day"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
    return df


def stratum_of(mx):
    out = np.empty(len(mx), dtype=object)
    for name, lo, hi in STRATA:
        out[(mx >= lo) & (mx < hi)] = name
    return out


# ----------------------------------------------------------------------------------
def assign_days(days, events, args):
    """day -> 'train' / 'val' / 'test', or a drop reason."""
    days = pd.DatetimeIndex(sorted(set(days)))
    iso = days.isocalendar()
    week = pd.Series(list(zip(iso["year"], iso["week"])), index=days)
    split = pd.Series("train", index=days, dtype=object)
    reason = pd.Series("", index=days, dtype=object)

    # 1. events -> test, with their ISO weeks
    ev_days = set()
    for e in events:
        s = pd.Timestamp(e["start"]) - pd.Timedelta(days=args.event_buffer)
        t = pd.Timestamp(e["end"]) + pd.Timedelta(days=args.event_buffer)
        ev_days.update(pd.date_range(s, t, freq="D"))
    nim = days >= pd.Timestamp(args.nimbus_start)
    nim_weeks = set(week[nim].values)
    ev_weeks = set(week[week.index.isin(ev_days) & ~nim].values) - nim_weeks

    # 2. val / test weeks, per calendar month across years
    rng = np.random.default_rng(args.seed)
    wk_month = pd.DataFrame({"week": week.values, "month": days.month}, index=days)
    wk_month = wk_month.groupby("week")["month"].agg(lambda m: m.mode().iloc[0])
    test_w, val_w = set(ev_weeks), set()
    for m in range(1, 13):
        cand = [w for w in wk_month.index[wk_month.values == m]
                if w not in ev_weeks and w not in nim_weeks]
        rng.shuffle(cand)
        n = len(cand) + sum(1 for w in ev_weeks if wk_month.get(w) == m)
        n_test = max(0, int(round(args.test_frac * n)) - sum(1 for w in ev_weeks
                                                                 if wk_month.get(w) == m))
        n_val = int(round(args.val_frac * n))
        test_w.update(cand[:n_test])
        val_w.update(cand[n_test:n_test + n_val])
    split[week.isin(test_w).values] = "test"
    split[week.isin(val_w).values] = "val"
    split[nim] = "nimbus"

    # 3. buffers: a day is dropped if any calendar day within +-buffer days belongs to a
    #    higher-priority split (train < val < test). So every train day is > buffer days
    #    from every val/test day, and every val day > buffer days from every test day.
    pri = pd.Series([PRIORITY[v] for v in split.values], index=days)
    full = pd.date_range(days.min() - pd.Timedelta(days=args.buffer),
                         days.max() + pd.Timedelta(days=args.buffer), freq="D")
    pri_full = pri.reindex(full, fill_value=-1)
    neigh_max = pri_full.rolling(2 * args.buffer + 1, center=True, min_periods=1).max()
    drop = (neigh_max.reindex(days).values > pri.values)
    reason[drop] = "buffer"
    split[drop] = "drop"
    is_event = pd.Series(days.isin(ev_days), index=days)
    return split, reason, is_event


def event_rows(df, events):
    """Event id for every tile inside an event's bbox during its window ('' otherwise)."""
    from src.data import geo
    ev = np.full(len(df), "", dtype=object)
    P = 128
    # tile centre lat/lon, once per tile location
    # A tile belongs to an event if its extent OVERLAPS the event box: tiles are 256 km on a
    # 256 km grid, so a small box (Valencia's is ~130 km) may contain no tile centre at all.
    locs = df[["row", "col"]].drop_duplicates().reset_index(drop=True)
    ext = [geo.tile_location(int(r), int(c), P) for r, c in zip(locs.row, locs.col)]
    for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
        locs[k] = [e[k] for e in ext]
    m_ = df[["row", "col"]].merge(locs, on=["row", "col"], how="left")
    for e in events:
        la0, la1, lo0, lo1 = e["bbox"]
        overlap = ((m_["lat_max"].values >= la0) & (m_["lat_min"].values <= la1)
                   & (m_["lon_max"].values >= lo0) & (m_["lon_min"].values <= lo1))
        m = ((df["day"] >= pd.Timestamp(e["start"])) & (df["day"] <= pd.Timestamp(e["end"]))).values & overlap
        ev[m & (ev == "")] = e["id"]
    return ev


def sample_split(d, budget, shares, seed, forced):
    """Stratified sample of `d` (one split). Returns (index, weight, stratum)."""
    rng = np.random.default_rng(seed)
    st = stratum_of(d["max"].values)
    keep_idx, w_out, s_out = [], [], []
    for name, _, _ in STRATA:
        m = st == name
        pool = np.nonzero(m & ~forced)[0]
        must = np.nonzero(m & forced)[0]
        n_target = int(round(shares[name] * budget))
        n_free = max(0, min(len(pool), n_target - len(must)))
        pick = rng.choice(pool, size=n_free, replace=False) if n_free < len(pool) else pool
        # inclusion probability: forced rows 1, sampled rows n_free / |pool|
        p = n_free / len(pool) if len(pool) else 1.0
        keep_idx += [must, pick]
        w_out += [np.ones(len(must)), np.full(len(pick), 1.0 / p if p > 0 else 0.0)]
        s_out += [np.full(len(must) + len(pick), name, dtype=object)]
    idx = np.concatenate(keep_idx).astype(np.int64)
    return idx, np.concatenate(w_out), np.concatenate(s_out)


def write_meta(path, d, store_rows=None):
    with open(path, "w") as f:
        if store_rows is None:
            for t, y, x, m in zip(d["timestamp"], d["row"], d["col"], d["max"]):
                f.write(f"{t},{y},{x},{m:.4f}\n")
        else:
            for t, y, x, m, r in zip(d["timestamp"], d["row"], d["col"], d["max"], store_rows):
                f.write(f"{t},{y},{x},{m:.4f},{r}\n")


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--tiles_dir", required=True)
    ap.add_argument("--events", default="configs/prominent_events.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budget", type=int, default=3_000_000, help="total full-dataset patches")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--buffer", type=int, default=1, help="days dropped at each split boundary")
    ap.add_argument("--event_buffer", type=int, default=1, help="days added around each event")
    ap.add_argument("--light_frac", type=float, default=0.25)
    ap.add_argument("--tail", type=float, default=31.0)
    ap.add_argument("--shares", default=None,
                    help="stratum shares, e.g. dry=0.08,light=0.12,moderate=0.25,heavy=0.25,tail=0.30")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--nimbus_start", default=NIMBUS_START,
                    help="first day of the NIMBUS product (own split); '' disables the era split")
    args = ap.parse_args()
    shares = dict(DEFAULT_SHARES)
    if args.shares:
        shares.update({k: float(v) for k, v in (kv.split("=") for kv in args.shares.split(","))})
    assert abs(sum(shares.values()) - 1) < 1e-6, f"shares must sum to 1: {shares}"
    os.makedirs(args.out_dir, exist_ok=True)
    rep = [f"# v2 dataset report", "", f"budget {args.budget:,}; shares {shares}; "
           f"val/test {args.val_frac}/{args.test_frac}; buffer {args.buffer} d; "
           f"event buffer {args.event_buffer} d; seed {args.seed}", ""]

    df = load_tiles(args.tiles_dir)
    events = yaml.safe_load(open(args.events))["events"]
    print(f"[splits] {len(df):,} tiles over {df['day'].nunique()} days; {len(events)} events",
          flush=True)

    # ---- rejection, reported by intensity bin of the raw max
    rej = (df["unphysical"] > 0) | (df["ray"] > 0)
    bins = [0, 0.1, 1, 10, 31, 89, 150, 500, np.inf]
    lab = pd.cut(df["raw_max"], bins, right=False)
    t = pd.DataFrame({"bin": lab, "rej": rej, "unph": df["unphysical"] > 0, "ray": df["ray"] > 0})
    g = t.groupby("bin", observed=True).agg(n=("rej", "size"), rejected=("rej", "mean"),
                                            unphysical=("unph", "mean"), ray=("ray", "mean"))
    rep += ["## Rejection by raw tile max (mm/h)", "", "| raw max | tiles | rejected | unphysical | ray |",
            "|---|---|---|---|---|"]
    rep += [f"| {b} | {int(r.n):,} | {r.rejected:.2%} | {r.unphysical:.2%} | {r.ray:.2%} |"
            for b, r in g.iterrows()]
    rep.append("")
    df = df[~rej].reset_index(drop=True)

    # ---- day splits
    split, reason, is_event = assign_days(df["day"].unique(), events, args)
    days_tab = pd.DataFrame({"day": split.index.strftime("%Y-%m-%d"), "split": split.values,
                             "reason": reason.values, "event_window": is_event.values})
    days_tab.to_csv(os.path.join(args.out_dir, "days.csv"), index=False)
    df["split"] = split.reindex(df["day"]).values
    vc = days_tab["split"].value_counts()
    rep += ["## Days", "", " | ".join(f"{k}: {v}" for k, v in vc.items()), ""]
    df["event"] = event_rows(df, events)

    # ---- leak check: no timestamp in two splits, and train >= 24 h from val/test
    kept = df[df["split"] != "drop"]
    per_ts = kept.groupby("timestamp")["split"].nunique()
    assert (per_ts == 1).all(), "a timestamp appears in two splits"
    tday = np.sort(kept.loc[kept["split"] == "train", "day"].unique())
    for other in ("val", "test"):
        oday = np.sort(kept.loc[kept["split"] == other, "day"].unique())
        if len(tday) and len(oday):
            gap = np.min(np.abs(tday[:, None] - oday[None, :])).astype("timedelta64[D]").astype(int)
            rep.append(f"min gap train <-> {other}: {gap} day(s)")
            assert gap >= args.buffer + 1, f"train within {gap} d of {other}"
    nday = np.sort(kept.loc[kept["split"] == "nimbus", "day"].unique())
    for other, od in (("train", tday), ("val", np.sort(kept.loc[kept["split"] == "val", "day"].unique())),
                      ("test", np.sort(kept.loc[kept["split"] == "test", "day"].unique()))):
        if len(od) and len(nday):
            gap = np.min(np.abs(od[:, None] - nday[None, :])).astype("timedelta64[D]").astype(int)
            rep.append(f"min gap {other} <-> nimbus: {gap} day(s)")
            assert gap >= args.buffer + 1, f"{other} within {gap} d of nimbus"
    vday = np.sort(kept.loc[kept["split"] == "val", "day"].unique())
    sday = np.sort(kept.loc[kept["split"] == "test", "day"].unique())
    if len(vday) and len(sday):
        gap = np.min(np.abs(vday[:, None] - sday[None, :])).astype("timedelta64[D]").astype(int)
        rep.append(f"min gap val <-> test: {gap} day(s)")
        assert gap >= args.buffer + 1, f"val within {gap} d of test"
    rep.append("")

    # ---- sampling per split, budget proportional to available tiles of that split
    n_by = kept["split"].value_counts()
    out = {}
    for sp in ("train", "val", "test", "nimbus"):
        d = kept[kept["split"] == sp].reset_index(drop=True)
        b = int(round(args.budget * n_by.get(sp, 0) / max(n_by.sum(), 1)))
        forced = ((d["event"].values != "") if sp in ("test", "nimbus")
                  else np.zeros(len(d), bool))
        idx, w, s = sample_split(d, b, shares, args.seed + PRIORITY[sp], forced)
        sel = d.iloc[idx].copy()
        sel["weight"], sel["stratum"] = w, s
        sel["era"] = np.where(sel["day"] >= pd.Timestamp(args.nimbus_start), "NIMBUS", "ODYSSEY")
        sel = sel.sort_values(["timestamp", "row", "col"]).reset_index(drop=True)   # store order
        out[sp] = sel
        write_meta(os.path.join(args.out_dir, f"full_{sp}.txt"), sel)
        sel[["timestamp", "row", "col", "max", "raw_max", "n_fixed", "has_qind", "stratum",
             "weight", "event", "era"]].to_csv(os.path.join(args.out_dir, f"full_{sp}_info.csv.gz"),
                                        index=True, index_label="store_row",
                                        compression="gzip", float_format="%.6g")
        cnt = sel["stratum"].value_counts()
        rep.append(f"full_{sp}: {len(sel):,} patches (pool {len(d):,}); "
                   + ", ".join(f"{k} {cnt.get(k, 0):,}" for k, _, _ in STRATA))
    rep.append("")

    # ---- subsets
    rng = np.random.default_rng(args.seed + 7)
    for sp, sel in out.items():
        rows = np.arange(len(sel))
        parts_ = [rng.choice(rows[sel["stratum"].values == k],
                             size=int(round(args.light_frac * (sel["stratum"].values == k).sum())),
                             replace=False) for k, _, _ in STRATA
                  if (sel["stratum"].values == k).any()]
        light = np.concatenate(parts_) if parts_ else np.zeros(0, np.int64)
        light.sort()
        write_meta(os.path.join(args.out_dir, f"light_{sp}.txt"), sel.iloc[light], light)
        ext = rows[sel["max"].values >= args.tail]
        write_meta(os.path.join(args.out_dir, f"extremes_{sp}.txt"), sel.iloc[ext], ext)
        rep.append(f"light_{sp}: {len(light):,} | extremes_{sp}: {len(ext):,}")
        if sp in ("test", "nimbus"):
            evr = rows[sel["event"].values != ""]
            write_meta(os.path.join(args.out_dir, f"events_{sp}.txt"), sel.iloc[evr], evr)
            rep.append(f"events_{sp}: {len(evr):,}")
    rep.append("")

    # ---- per-event coverage
    rep += ["## Events", "", "| event | split | tiles | tiles >= 31 | max (mm/h) |",
            "|---|---|---|---|---|"]
    for e in events:
        for sp in ("test", "nimbus"):
            te = out[sp]; m = te["event"].values == e["id"]
            if m.any() or (sp == "test" and not (out["nimbus"]["event"].values == e["id"]).any()):
                rep.append(f"| {e['id']} | {sp} | {int(m.sum()):,} | {int((te['max'].values[m] >= 31).sum()):,} | "
                           f"{(te['max'].values[m].max() if m.any() else float('nan')):.1f} |")
    open(os.path.join(args.out_dir, "report.md"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
