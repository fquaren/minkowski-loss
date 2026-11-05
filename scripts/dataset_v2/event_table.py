#!/usr/bin/env python
"""
The appendix event table and the frozen showcases (DECISIONS §43, todo A4).

One row per event of both catalogues (`configs/prominent_events.yaml`, `configs/mch_events.csv`, rain,
mixed and precip-without-warning events), with the v4 split, the ladder rungs it enters at a cut u_c,
and statistics of the target tile-hours over it, all from the corrected lists (no model output):
    tiles     prominent events: tiles intersecting the event box; MCH events: the two Swiss tiles
    window    prominent: start day 00 UTC .. end day + 1 d; MCH: start .. end + 1 h (UTC)
Rungs: ODYSSEY test -> R1, and R4 if the event's tile maximum >= u_c; NIMBUS in the paired period
(2024-07-05 .. 10-30) -> R2, otherwise R3; NIMBUS and >= u_c -> R5. Train / val events are not scored.

Usage: python scripts/dataset_v2/event_table.py 50.1 notes/event_table.csv
"""

import sys

import numpy as np
import pandas as pd
import yaml

from src.data.geo import tile_location

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta"
PAIRED = (pd.Timestamp("2024-07-05"), pd.Timestamp("2024-10-31"))
CH_TILES = {(640, 768), (640, 896)}
SHOWCASES = {"misox_20240621": "C1", "boris_20240913": "C2", "mch_36": "C2", "mch_64": "C3"}


def lists():
    out = []
    for sp in ("train", "val", "test", "nimbus"):
        m = pd.read_csv(f"{V4}/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        m["t"] = pd.to_datetime(m.t.astype(str), format="%Y%m%d%H%M%S")
        out.append(m.assign(split=sp))
    return pd.concat(out, ignore_index=True)


def summarise(w, u_c):
    if not len(w):
        return {"split": "none", "tile_hours": 0}
    sp = w.split.value_counts().index[0]
    mx = w.mx.max()
    t0 = w.t.min()
    if sp == "test":
        rungs = ["R1"] + (["R4"] if mx >= u_c else [])
    elif sp == "nimbus":
        rungs = ["R2" if PAIRED[0] <= t0 < PAIRED[1] else "R3"] + (["R5"] if mx >= u_c else [])
    else:
        rungs = []
    return {"split": "+".join(sorted(w.split.unique())), "tile_hours": len(w), "tile_max_mm": round(mx, 1),
            f"tile_hours_ge_u_c": int((w.mx >= u_c).sum()), "rungs": "|".join(rungs) or "not scored"}


if __name__ == "__main__":
    u_c, dst = float(sys.argv[1]), sys.argv[2]
    m = lists()
    tiles = m[["y", "x"]].drop_duplicates()
    box = {k: tile_location(*k) for k in map(tuple, tiles.values)}
    rows = []
    for e in yaml.safe_load(open("configs/prominent_events.yaml"))["events"]:
        if not e.get("available", True):
            continue
        la0, la1, lo0, lo1 = e["bbox"]
        keep = {k for k, b in box.items() if b["lat_min"] <= la1 and b["lat_max"] >= la0
                and b["lon_min"] <= lo1 and b["lon_max"] >= lo0}
        t0, t1 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]) + pd.Timedelta(days=1)
        w = m[(m.t > t0).values & (m.t <= t1).values & np.array([k in keep for k in zip(m.y, m.x)])]
        rows.append({"catalogue": "ours", "id": e["id"], "name": e["name"], "start": t0, "end": t1,
                     "type": "|".join(e.get("type", [])), **summarise(w, u_c)})
    mch = pd.read_csv("configs/mch_events.csv", parse_dates=["start", "end"])
    mch = mch[mch.precip_kind.isin(["rain", "mixed", "precip_no_warning"])]
    ch = m[[k in CH_TILES for k in zip(m.y, m.x)]]
    for e in mch.itertuples():
        w = ch[(ch.t > e.start) & (ch.t <= e.end + pd.Timedelta(hours=1))]
        rows.append({"catalogue": "MCH", "id": f"mch_{e._1}", "name": e.name, "start": e.start, "end": e.end,
                     "type": e.type_codes if isinstance(e.type_codes, str) else "", **summarise(w, u_c),
                     "same_as": e.prominent_event if isinstance(e.prominent_event, str) else ""})
    t = pd.DataFrame(rows)
    t["showcase"] = t.id.map(SHOWCASES).fillna("")
    t.sort_values(["rungs", "start"]).to_csv(dst, index=False)
    pd.set_option("display.width", 250)
    print(t[["catalogue", "id", "start", "split", "tile_hours", "tile_max_mm", "tile_hours_ge_u_c", "rungs", "showcase"]]
          .sort_values(["rungs", "start"]).to_string(index=False))
    print(t.groupby(["catalogue", "rungs"]).size().unstack(fill_value=0))
