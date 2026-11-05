#!/usr/bin/env python
"""
Clean the MeteoSwiss event list (Lionel Moret, received 2026-11-04) and map it onto the v4 splits.

Input: the `events_database.xlsx` sheet as delivered (one row per event, 12 columns). Read with
the standard library, since openpyxl is not in dl-stable. Output: one CSV row per source row,
same `_id`, the source columns cleaned, plus derived columns:

    start, end          UTC. The sheet is in Zurich local time (CET/CEST; Lionel Moret, 2026-11-04);
                        start_local / end_local keep it as given. Our hourly timestamps are UTC.
    duration_h
    precip_kind         rain / snow / mixed from the warnings; "precip_no_warning" if TOT_PREC
                        is listed without a rain/snow/thunderstorm warning; "none" otherwise
    cleaning_note       every change made to the row
    v4_split            v4 lists holding tile-hours of the two tiles covering Switzerland,
                        (640, 768) west and (640, 896) east, in (start, end + 1 h]
    v4_tile_hours       tile-hours of those two tiles in the window, all lists together
    v4_tile_max_mm      largest hourly tile maximum among them (plain sum, list column 4). Each
                        tile is 256 km wide and covers parts of FR/DE/IT/AT, so this is not the
                        event's own maximum.
    prominent_event     id of an overlapping event in configs/prominent_events.yaml (same days,
                        box intersecting Switzerland), if any

Usage: python scripts/dataset_v2/import_mch_events.py <events_database.xlsx> configs/mch_events.csv
"""

import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import yaml

META = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta"
CH_TILES = {(640, 768), (640, 896)}   # src.data.geo.tile_location: the tiles touching CH's bbox
CH_BOX = (45.82, 47.81, 5.95, 10.49)  # lat_min, lat_max, lon_min, lon_max
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def read_sheet(path):
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{NS['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", NS):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                col = col * 26 + ord(ch) - 64
            v = c.find("m:v", NS)
            vals[col - 1] = ss[int(v.text)] if c.get("t") == "s" and v is not None else (v.text if v is not None else "")
        rows.append([vals.get(i, "") for i in range(max(vals) + 1)])
    return pd.DataFrame(rows[1:], columns=rows[0])


def clean(d):
    notes = {i: [] for i in d.index}
    for i, r in d.iterrows():
        name = re.sub(r"\s+", " ", r["name"] or "").strip()
        if name != (r["name"] or ""):
            notes[i].append("name whitespace")
        if not name:
            notes[i].append("no name in source")
        d.at[i, "name"] = name
        for c in ("start_date", "start_time", "end_date", "end_time"):
            if r[c] != r[c].strip():
                r[c] = r[c].strip()
                d.at[i, c] = r[c]
                notes[i].append(f"{c} whitespace")
        for c in ("start_time", "end_time"):
            if re.fullmatch(r"\d\d:\d\d", r[c]):
                d.at[i, c] = r[c] + ":00"
                notes[i].append(f"{c} HH:MM -> HH:MM:SS")
        if r["end_date"].startswith("2923-"):
            d.at[i, "end_date"] = "2023" + r["end_date"][4:]
            notes[i].append("end_date year 2923 -> 2023 (typo)")
        if "thuderstorms" in r["warnings_codes"]:
            d.at[i, "warnings_codes"] = r["warnings_codes"].replace("thuderstorms", "thunderstorms")
            notes[i].append("warning 'thuderstorms' spelling")
        d.at[i, "station_codes"] = "|".join(s.strip() for s in r["station_codes"].split(",") if s.strip())
    d["start_local"] = pd.to_datetime(d.start_date + " " + d.start_time)
    d["end_local"] = pd.to_datetime(d.end_date + " " + d.end_time)
    for c in ("start", "end"):  # raise on DST-ambiguous or nonexistent local times rather than guess
        d[c] = (d[c + "_local"].dt.tz_localize("Europe/Zurich", ambiguous="raise", nonexistent="raise")
                .dt.tz_convert("UTC").dt.tz_localize(None))
    d["duration_h"] = (d.end - d.start).dt.total_seconds() / 3600
    assert (d.duration_h > 0).all(), d[d.duration_h <= 0]
    for (i, j) in [(a, b) for a in d.index for b in d.index if a < b]:
        if d.at[i, "start"] == d.at[j, "start"] and d.at[i, "end"] == d.at[j, "end"]:
            notes[i].append(f"same window as _id {d.at[j, '_id']} (different stations)")
            notes[j].append(f"same window as _id {d.at[i, '_id']} (different stations)")

    def kind(r):
        w = set(r.warnings_codes.split("|")) - {""}
        rain, snow = bool(w & {"rain", "thunderstorms"}), "snow" in w
        if rain and snow:
            return "mixed"
        if rain or snow:
            return "rain" if rain else "snow"
        return "precip_no_warning" if "TOT_PREC" in r.variable_codes else "none"
    d["precip_kind"] = d.apply(kind, axis=1)
    d["cleaning_note"] = ["; ".join(notes[i]) for i in d.index]
    return d


def v4_overlap(d):
    lists = {}
    for sp in ("train", "val", "test", "nimbus"):
        m = pd.read_csv(f"{META}/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        m = m[[k in CH_TILES for k in zip(m.y, m.x)]]
        m["t"] = pd.to_datetime(m.t.astype(str), format="%Y%m%d%H%M%S")  # end of the hour
        lists[sp] = m
    split, n, mx = [], [], []
    for _, ev in d.iterrows():
        hit = {sp: m[(m.t > ev.start) & (m.t <= ev.end + pd.Timedelta(hours=1))] for sp, m in lists.items()}
        split.append("+".join(sp for sp, w in hit.items() if len(w)) or "none")
        n.append(sum(len(w) for w in hit.values()))
        mx.append(max((w.mx.max() for w in hit.values() if len(w)), default=np.nan))
    d["v4_split"], d["v4_tile_hours"], d["v4_tile_max_mm"] = split, n, mx


def prominent(d):
    evs = yaml.safe_load(open("configs/prominent_events.yaml"))["events"]
    la0, la1, lo0, lo1 = CH_BOX
    out = []
    for _, r in d.iterrows():
        hits = [e["id"] for e in evs
                if e.get("available", True)
                and e["bbox"][0] <= la1 and e["bbox"][1] >= la0 and e["bbox"][2] <= lo1 and e["bbox"][3] >= lo0
                and pd.Timestamp(e["start"]) <= r.end.normalize() and pd.Timestamp(e["end"]) >= r.start.normalize()]
        out.append("|".join(hits))
    d["prominent_event"] = out


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    d = clean(read_sheet(src))
    v4_overlap(d)
    prominent(d)
    cols = ["_id", "name", "start", "end", "start_local", "end_local", "duration_h", "precip_kind", "type_codes", "variable_codes",
            "warnings_codes", "region_codes", "station_codes", "v4_split", "v4_tile_hours", "v4_tile_max_mm",
            "prominent_event", "cleaning_note", "description"]
    d[cols].to_csv(dst, index=False, float_format="%.4g")
    print(f"{len(d)} events -> {dst}")
    print(d.groupby(["precip_kind", "v4_split"]).size().unstack(fill_value=0))
    print(d[d.cleaning_note != ""][["_id", "cleaning_note"]].to_string(index=False))
    print(d[d.prominent_event != ""][["_id", "name", "prominent_event", "v4_split"]].to_string(index=False))
