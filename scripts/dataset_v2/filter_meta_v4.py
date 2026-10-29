#!/usr/bin/env python
"""Metadata-level filtering of the v4 store (DECISIONS §34). The store is unchanged.

  - excludes ownerless tile-hours (no owning radar in the hourly scan's `owners` column, so rule 5
    and rule 3 cannot act on them): writes `<meta>/<list>_owned.txt` for every list, 5 columns
    (timestamp, row, col, max, store_row), which `DeterministicSRDataset` maps to store rows;
  - flags pre-break rows (hour end before 2015-12-12, the speckle-floor step): writes
    `<meta>/full_<split>_flags.csv.gz` (store_row, ownerless, pre_break);
  - with `--biased <csv>` (columns radar, year; DECISIONS §36b), also excludes tile-hours in which a
    flagged radar-year owns >= 10% of the tile (`owners` share), and adds a `biased` flag column.

    python scripts/dataset_v2/filter_meta_v4.py [--biased <quality_v4>/biased_radar_years_tail.csv]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

D = "/home/fquareng/work/data/extremes/OPERA"
META = f"{D}/patches_v4/meta"
HOURS = f"{D}/quality_v4/hourly/hours"
BREAK = "20151212"
SPLITS = {"train": "full_train", "validation": "full_val", "test": "full_test", "nimbus": "full_nimbus"}
SUBSETS = {"train": ["light_train", "extremes_train"], "validation": ["light_val", "extremes_val"],
           "test": ["light_test", "extremes_test", "events_test"],
           "nimbus": ["light_nimbus", "extremes_nimbus", "events_nimbus"]}


def ownerless_keys(days):
    keys = set()
    for d in sorted(days):
        f = f"{HOURS}/{d}.csv.gz"
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f, usecols=["end", "row", "col", "owners"], dtype={"end": str, "owners": str})
        t = t[t.owners.fillna("").str.strip() == ""]
        keys |= set(zip(t.end.str[:12], t.row.astype(int), t.col.astype(int)))
    return keys


def biased_keys(days, flagged, min_share=0.1):
    """(end[:12], row, col) of tile-hours where a flagged (radar, year) owns >= min_share."""
    keys = set()
    years = {y for _, y in flagged}
    for d in sorted(days):
        f = f"{HOURS}/{d}.csv.gz"
        if d[:4] not in years or not os.path.exists(f):
            continue
        t = pd.read_csv(f, usecols=["end", "row", "col", "owners"], dtype={"end": str, "owners": str})
        for e, r, c, o in zip(t.end, t.row, t.col, t.owners.fillna("")):
            for part in o.split(";"):
                if ":" in part:
                    k, v = part.split(":")
                    if (k, d[:4]) in flagged and float(v) >= min_share:
                        keys.add((e[:12], int(r), int(c)))
                        break
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--biased", default=None, help="CSV of flagged radar-years (radar, year)")
    args = ap.parse_args()
    flagged = set()
    if args.biased:
        b = pd.read_csv(args.biased, dtype={"radar": str, "year": str})
        flagged = set(zip(b.radar, b.year))
    report = [f"biased radar-years: {len(flagged)} from {args.biased}"] if args.biased else []
    for split, stem in SPLITS.items():
        lines = open(f"{META}/{stem}.txt").read().split("\n")
        lines = [l for l in lines if l.strip()]
        ts = [l.split(",")[0] for l in lines]
        rc = [(int(l.split(",")[1]), int(l.split(",")[2])) for l in lines]
        T = pd.to_datetime(pd.Series(ts).str[:14], format="%Y%m%d%H%M%S")
        pdays = (T - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d").unique()
        bad = ownerless_keys(pdays[pdays < BREAK + "0"] if split != "nimbus" else [])
        own = np.array([(t[:12], r, c) in bad for t, (r, c) in zip(ts, rc)])
        pre = np.array([t[:8] < BREAK for t in ts])
        bk = biased_keys(pdays, flagged) if flagged else set()
        bia = np.array([(t[:12], r, c) in bk for t, (r, c) in zip(ts, rc)])
        keep = np.nonzero(~own & ~bia)[0]
        with open(f"{META}/{stem}_owned.txt", "w") as f:
            for i in keep:
                f.write(f"{lines[i].split(',')[0]},{lines[i].split(',')[1]},{lines[i].split(',')[2]},{lines[i].split(',')[3]},{i}\n")
        flags = {"store_row": np.arange(len(lines)), "ownerless": own, "pre_break": pre}
        if flagged:
            flags["biased"] = bia
        pd.DataFrame(flags).to_csv(f"{META}/{stem}_flags.csv.gz", index=False)
        report.append(f"{split}: {len(lines):,} rows, ownerless {own.sum():,} excluded, "
                      + (f"biased radar-year {(bia & ~own).sum():,} excluded, " if flagged else "")
                      + f"kept {len(keep):,}, pre-break {pre[keep].sum():,} of the kept")
        keep_set = set(keep.tolist())
        for sub in SUBSETS[split]:
            p = f"{META}/{sub}.txt"
            if not os.path.exists(p):
                continue
            sl = [l for l in open(p).read().split("\n") if l.strip()]
            ok = [l for l in sl if int(l.split(",")[4]) in keep_set]
            open(f"{META}/{sub}_owned.txt", "w").write("\n".join(ok) + ("\n" if ok else ""))
            report.append(f"  {sub}: {len(sl):,} -> {len(ok):,}")
    txt = "\n".join(report)
    print(txt)
    open(f"{META}/filter_report.txt", "w").write(f"filter_meta_v4.py (DECISIONS §34), break {BREAK}\n" + txt + "\n")


if __name__ == "__main__":
    main()
