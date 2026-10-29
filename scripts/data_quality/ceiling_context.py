#!/usr/bin/env python
"""Ceiling rejections >= 20 mm: capped intense cores or artefacts? (DECISIONS §29, test (iii))

For every calibration tile-hour rejected by `rej_ceiling` with hourly max >= 20 mm
(`calib/hourly_c`), every raw pixel of the tile, in the hour's four frames, equal (+-0.005) to a
ceiling value of that year (`ceilings.csv`; radars sharing a value counted once). Context = dBZ of the
median of the valid non-ceiling pixels of its 5x5 window (no rain counts as 0). Embedded =
context >= ceiling dBZ - 10 dB. Pass lines: >= 80% embedded -> capped cores; <= 20% -> artefacts.

    python scripts/data_quality/ceiling_context.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO]
from src.data.radar_screen import dbz, decide_hourly  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
HOURS = f"{Q4}/calib/hourly_c/hours"
OUT = f"{Q4}/calib/ceiling_context"
P = 128


def main():
    from src.data.day_cleaner import open_day
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    ce = pd.read_csv(f"{Q4}/ceilings.csv")
    rows = []
    for f in sorted(glob.glob(f"{HOURS}/*.csv.gz")):
        day = os.path.basename(f)[:8]
        t = pd.read_csv(f, dtype={"end": str, "frames": str, "owners": str})
        if not len(t):
            continue
        sel = t[decide_hourly(t, q)["rej_ceiling"].values & (t["max"].values >= 20)]
        if not len(sel):
            continue
        cy = ce[ce.year == int(day[:4])]
        ds = open_day(f"{D}/raw/OPERA/{day}")
        tl = {np.datetime_as_string(x, unit="s").replace("-", "").replace("T", "").replace(":", ""): i
              for i, x in enumerate(ds.time.values)}
        for _, th in sel.iterrows():
            # every ceiling value of the year (a value can come from a radar not owning the tile);
            # radars sharing a value are listed together, so each pixel is counted once
            vals = cy.groupby(["value", "dbz"]).radar.apply(lambda s: "/".join(sorted(set(s)))).reset_index()
            vals = vals[["radar", "value", "dbz"]].values
            if not len(vals):
                continue
            r0, c0 = int(th.row), int(th.col)
            for fr in str(th.frames).split(";"):
                if fr not in tl:
                    continue
                z = ds["TOT_PREC"].isel(time=tl[fr]).values.astype(np.float32)
                tile = z[r0:r0 + P, c0:c0 + P]
                ceilm = np.zeros(z.shape, bool)
                for _, v, _ in vals:
                    ceilm |= np.abs(z - v) <= 0.005
                for radar, v, vdb in vals:
                    ys, xs = np.nonzero(np.abs(tile - v) <= 0.005)
                    for y, x in zip(ys + r0, xs + c0):
                        w = z[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
                        cm = ceilm[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
                        nb = w[np.isfinite(w) & ~cm]
                        med = float(np.median(np.maximum(nb, 0))) if len(nb) else np.nan
                        ctx = float(dbz(med)) if med > 0 else -np.inf
                        rows.append({"day": day, "end": th.end, "frame": fr, "radar": radar, "value": v,
                                     "ceiling_dbz": vdb, "row": y, "col": x, "n_nb": len(nb),
                                     "nb_median": med, "ctx_dbz": ctx, "embedded": ctx >= vdb - 10})
        ds.close()
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/pixels.csv.gz", index=False)
    emb = df.embedded.mean()
    verdict = "capped cores (>= 80%)" if emb >= 0.8 else "artefacts (<= 20%)" if emb <= 0.2 else "mixed"
    L = ["# Ceiling pixels in rejected tile-hours >= 20 mm (DECISIONS §29, test (iii))", "",
         f"{df[['day', 'end']].drop_duplicates().shape[0]} tile-hours, {len(df):,} ceiling pixel-frames; "
         f"embedded (5x5 non-ceiling median within 10 dB of the ceiling): **{emb:.1%} -> {verdict}**", "",
         "| radar | value (mm/h) | dBZ | pixel-frames | embedded | context median (dBZ) | no rain around |",
         "|---|---|---|---|---|---|---|"]
    for (r, v, vd), g in df.groupby(["radar", "value", "ceiling_dbz"]):
        fin = g.ctx_dbz[np.isfinite(g.ctx_dbz)]
        L.append(f"| {r} | {v} | {vd} | {len(g):,} | {g.embedded.mean():.1%} | "
                 f"{fin.median():.1f} | {np.isinf(g.ctx_dbz).mean():.1%} |")
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
