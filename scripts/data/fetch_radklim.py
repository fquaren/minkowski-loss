#!/usr/bin/env python
"""Fetch DWD RADKLIM RW (v2017.002, hourly, 1 km, Germany) for given days and put it on the OPERA
grid, paired with the hourly target's hours (DECISIONS §21, §24; `src/data/radklim.py`).

Monthly archives from opendata.dwd.de (`.../grids_germany/hourly/radolan/reproc/2017_002/bin/YYYY/
RW2017.002_YYYYMM.tar.gz`, ~30 MB) are downloaded once to <root>/tar. For each processing day d
(hours ending d 01:00 .. d+1 00:00, ACRR convention) it writes <root>/hourly/YYYYMMDD.npz:
  K      (24, h, w) float32 mm, NaN = no data; hour i ends at d 01:00 + i h
  box    (y0, y1, x0, x1) of the OPERA grid covered by RADKLIM
Pairing: OPERA hour H <-> the RADKLIM file labelled H-10 min (sum H-70 .. H-10 min). On
2021-07-14 the hour ending 16:00 correlates 0.83 with RADKLIM 14:50-15:50, against 0.67 and
0.47 for the neighbouring hours.

    python scripts/data/fetch_radklim.py --days_file .../quality_v4/calib/days.txt
"""

import argparse
import datetime as dt
import os
import sys
import tarfile
import tempfile
import urllib.request

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data import radklim as RK  # noqa: E402

BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/reproc/2017_002/bin"


def month_tar(root, ym):
    os.makedirs(os.path.join(root, "tar"), exist_ok=True)
    p = os.path.join(root, "tar", f"RW2017.002_{ym}.tar.gz")
    if not os.path.exists(p):
        url = f"{BASE}/{ym[:4]}/RW2017.002_{ym}.tar.gz"
        try:
            urllib.request.urlretrieve(url, p + ".part")
            os.replace(p + ".part", p)
        except Exception as e:                                           # noqa: BLE001
            print(f"  ! {ym}: {e}", flush=True)
            return None
    return p


def box():
    m = np.load(RK.MAP) if os.path.exists(RK.MAP) else np.load(RK.build_map())
    return int(m["row"].min()), int(m["row"].max()) + 1, int(m["col"].min()), int(m["col"].max()) + 1


def needed_names(day):
    d0 = dt.datetime.strptime(day, "%Y%m%d")
    return [f"raa01-rw2017.002_10000-{d0 + dt.timedelta(hours=i + 1, minutes=-10):%y%m%d%H%M}-dwd---bin"
            for i in range(24)]


def read_members(path, names):
    """{name: bytes} for the wanted members, reading the .tar.gz once, sequentially (random
    access would decompress the ~1.5 GB archive from the start for every member)."""
    out = {}
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if m.name in names:
                out[m.name] = tf.extractfile(m).read()
    return out


def build_day(root, day, blobs):
    y0, y1, x0, x1 = box()
    K = np.full((24, y1 - y0, x1 - x0), np.nan, np.float32)
    n = 0
    for i, name in enumerate(needed_names(day)):
        if name not in blobs:
            continue
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(blobs[name]); tmp.flush()
            field, _ = RK.read_rw(tmp.name)
        K[i] = RK.radklim_to_opera(field)[y0:y1, x0:x1]
        n += 1
    os.makedirs(os.path.join(root, "hourly"), exist_ok=True)
    np.savez_compressed(os.path.join(root, "hourly", f"{day}.npz"), K=K, box=np.array([y0, y1, x0, x1]))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--days_file", default=None)
    ap.add_argument("--root", default=RK.ROOT)
    a = ap.parse_args()
    days = list(a.days)
    if a.days_file:
        days += [l.strip() for l in open(a.days_file) if l.strip()]
    days = sorted({d.replace("-", "") for d in days})
    todo = [d for d in days if not os.path.exists(os.path.join(a.root, "hourly", f"{d}.npz"))]
    by_month = {}
    for d in todo:
        t = dt.datetime.strptime(d, "%Y%m%d")
        by_month.setdefault(t.strftime("%Y%m"), set()).add(d)   # all 24 labels (00:50 .. 23:50) are on day d
    print(f"[radklim] {len(todo)} days, {len(by_month)} months", flush=True)
    blobs, done = {}, set()
    for ym in sorted(by_month):
        p = month_tar(a.root, ym)
        want = {n for d in by_month[ym] for n in needed_names(d) if n[23:27] == ym[2:]}
        if p:
            blobs.update(read_members(p, want))
        for d in sorted(by_month[ym]):
            n = build_day(a.root, d, blobs)
            done.add(d)
            print(f"  {d}: {n}/24 hours", flush=True)
        blobs = {}
    print(f"[radklim] done: {len(done)} days", flush=True)


if __name__ == "__main__":
    main()
