#!/usr/bin/env python
"""Download open 10-min rain-gauge records as independent truth for the radar tail.

Networks (both open, CC-BY / DWD terms "Frei"):
  dwd   DWD Climate Data Center, 10-min precipitation (RWS_10, mm per 10 min), ~1,000
        stations; `historical/` (to the end of last year) + `recent/` (~last 500 days).
  smn   MeteoSwiss open data, SwissMetNet automatic stations (`ogd-smn`, 2010 ->) and
        automatic precipitation stations (`ogd-smn-precip`, 2020 ->), parameter rre150z0
        (mm per 10 min).

Timestamps are kept exactly as delivered (UTC, end-of-interval labelling is NOT assumed);
`gauge_vs_radar.py` estimates the alignment empirically from the lag correlation.

Outputs under --out_dir (default OPERA/validation/gauges):
  <net>_stations.csv   station, name, lat, lon, elev, row, col (radar grid; NaN outside)
  <net>_wet.npz        sid (index into stations), t (minutes since 1970-01-01, as labelled),
                       rr (mm per 10 min) for every interval with rr > 0
  <net>_avail.npz      n_valid[station, day] (valid 10-min values per day), days
  raw/<net>/           the downloaded files (kept, so a re-run parses without downloading)

    python scripts/validation/fetch_gauges.py --net dwd smn --start 2012-09-01 --workers 2
"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DWD = ("https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
       "10_minutes/precipitation")
STAC = "https://data.geo.admin.ch/api/stac/v1/collections"
SMN_COLLECTIONS = ("ch.meteoschweiz.ogd-smn", "ch.meteoschweiz.ogd-smn-precip")


def _get(url, path=None, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if path:
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
            return data
        except Exception as e:                       # noqa: BLE001 - retry any network error
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (k + 1))


def _download_all(jobs, workers):
    """jobs: [(url, path)]; skips files already on disk."""
    todo = [(u, p) for u, p in jobs if not os.path.exists(p)]
    print(f"  {len(jobs)} files, {len(todo)} to download", flush=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get, u, p): u for u, p in todo}
        for k, f in enumerate(cf.as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:                   # noqa: BLE001
                print(f"  ! {e}", flush=True)
            if k % 100 == 0:
                print(f"  {k}/{len(todo)} downloaded", flush=True)


# --------------------------------------------------------------------------- DWD
def dwd_files(start):
    out = []
    for sub in ("historical", "recent"):
        html = _get(f"{DWD}/{sub}/").decode("latin1")
        for name in sorted(set(re.findall(r'href="(10minutenwerte_nieder_[^"]+\.zip)"', html))):
            m = re.match(r"10minutenwerte_nieder_(\d{5})_(\d{8})_(\d{8})_hist\.zip", name)
            if m and m.group(3) < start.replace("-", ""):
                continue                              # historical file ending before start
            out.append((f"{DWD}/{sub}/{name}", sub, name))
    return out


def dwd_stations(raw):
    p = os.path.join(raw, "zehn_min_rr_Beschreibung_Stationen.txt")
    if not os.path.exists(p):
        _get(f"{DWD}/historical/zehn_min_rr_Beschreibung_Stationen.txt", p)
    rows = []
    for line in open(p, encoding="latin1").read().splitlines()[2:]:
        f = line.split()
        if len(f) < 7 or not f[0].isdigit():
            continue
        rows.append({"station": f[0], "elev": float(f[3]), "lat": float(f[4]),
                     "lon": float(f[5]), "name": " ".join(f[6:-2])})
    return pd.DataFrame(rows).drop_duplicates("station")


def dwd_parse(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.startswith("produkt"))
        d = pd.read_csv(z.open(name), sep=";", usecols=["STATIONS_ID", "MESS_DATUM", "RWS_10"],
                        dtype={"STATIONS_ID": str, "MESS_DATUM": str}, skipinitialspace=True)
    d["station"] = d["STATIONS_ID"].str.strip().str.zfill(5)
    d["time"] = pd.to_datetime(d["MESS_DATUM"].str.strip(), format="%Y%m%d%H%M")
    d["rr"] = pd.to_numeric(d["RWS_10"], errors="coerce")
    d.loc[d["rr"] < 0, "rr"] = np.nan                  # -999 = missing
    return d[["station", "time", "rr"]]


def fetch_dwd(out_dir, start, end, workers):
    raw = os.path.join(out_dir, "raw", "dwd")
    os.makedirs(raw, exist_ok=True)
    files = dwd_files(start)
    jobs = [(u, os.path.join(raw, f"{sub}_{n}")) for u, sub, n in files]
    _download_all(jobs, workers)
    st = dwd_stations(raw)
    acc = Accumulator(st, start, end)
    for k, (_, p) in enumerate(jobs, 1):
        if os.path.exists(p):
            try:
                acc.add(dwd_parse(p))
            except Exception as e:                   # noqa: BLE001
                print(f"  ! parse {os.path.basename(p)}: {e}", flush=True)
        if k % 200 == 0:
            print(f"  parsed {k}/{len(jobs)}", flush=True)
    return acc


# --------------------------------------------------------------------------- SMN
def smn_items(collection):
    url, items = f"{STAC}/{collection}/items?limit=100", []
    while url:
        d = json.loads(_get(url))
        items += d["features"]
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
    return items


def fetch_smn(out_dir, start, end, workers):
    raw = os.path.join(out_dir, "raw", "smn")
    os.makedirs(raw, exist_ok=True)
    st_parts, jobs = [], []
    y0 = int(start[:4])
    for coll in SMN_COLLECTIONS:
        short = coll.split(".")[-1]
        meta = os.path.join(raw, f"{short}_meta_stations.csv")
        if not os.path.exists(meta):
            _get(f"https://data.geo.admin.ch/{coll}/{short}_meta_stations.csv", meta)
        m = pd.read_csv(meta, sep=";", encoding="latin1")
        st_parts.append(pd.DataFrame({
            "station": m["station_abbr"].str.upper(), "name": m["station_name"],
            "lat": m["station_coordinates_wgs84_lat"], "lon": m["station_coordinates_wgs84_lon"],
            "elev": m["station_height_masl"]}))
        for it in smn_items(coll):
            for key, a in it["assets"].items():
                m_ = re.search(r"_t_(historical_(\d{4})-(\d{4})|recent)\.csv$", key)
                if not m_ or (m_.group(3) and int(m_.group(3)) < y0):
                    continue
                jobs.append((a["href"], os.path.join(raw, key)))
    _download_all(jobs, workers)
    st = pd.concat(st_parts).drop_duplicates("station").reset_index(drop=True)
    acc = Accumulator(st, start, end)
    for _, p in jobs:
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, sep=";", encoding="latin1", dtype={"station_abbr": str},
                        usecols=lambda c: c in ("station_abbr", "reference_timestamp", "rre150z0"))
        if "rre150z0" not in d:
            continue
        acc.add(pd.DataFrame({
            "station": d["station_abbr"].str.upper(),
            "time": pd.to_datetime(d["reference_timestamp"], format="%d.%m.%Y %H:%M"),
            "rr": pd.to_numeric(d["rre150z0"], errors="coerce")}))
    return acc


# --------------------------------------------------------------------------- common
class Accumulator:
    """Keeps only what validation needs, file by file: the wet intervals, and the number
    of valid 10-min values per (station, day). Files that overlap (DWD historical vs
    recent) are reconciled by dropping duplicate (station, time) wet rows and by taking
    the per-day maximum of the valid counts."""

    def __init__(self, stations, start, end):
        self.st = stations.reset_index(drop=True)
        self.idx = {k: i for i, k in enumerate(self.st["station"])}
        self.days = pd.date_range(start, end, freq="D")
        self.t0, self.t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        self.n_valid = np.zeros((len(self.st), len(self.days)), np.uint8)
        self.sid, self.t, self.rr = [], [], []

    def add(self, d):
        d = d[(d["time"] >= self.t0) & (d["time"] < self.t1) & d["station"].isin(self.idx)]
        if not len(d):
            return
        s = d["station"].map(self.idx).values.astype(np.int32)
        t = d["time"].values.astype("datetime64[m]").astype(np.int64)
        rr = d["rr"].values.astype(np.float32)
        ok = np.isfinite(rr)
        day = ((t // 1440) - (self.t0.value // 60_000_000_000 // 1440)).astype(np.int64)
        cnt = np.zeros_like(self.n_valid, dtype=np.int32)
        np.add.at(cnt, (s[ok], day[ok]), 1)
        np.maximum(self.n_valid, np.minimum(cnt, 255).astype(np.uint8), out=self.n_valid)
        wet = ok & (rr > 0)
        self.sid.append(s[wet]); self.t.append(t[wet]); self.rr.append(rr[wet])

    def write(self, net, out_dir):
        from src.data import geo
        sid, t, rr = (np.concatenate(x) if x else np.array([]) for x in (self.sid, self.t, self.rr))
        key = sid.astype(np.int64) * (1 << 40) + t
        _, first = np.unique(key, return_index=True)
        sid, t, rr = sid[first], t[first], rr[first]
        st = self.st.copy()
        r, c = geo.lonlat_to_rowcol(st["lon"].values, st["lat"].values)
        r, c = np.asarray(r, float), np.asarray(c, float)
        inside = (r >= 0) & (r < 2200) & (c >= 0) & (c < 1900)
        st["row"] = np.where(inside, np.round(r), np.nan)
        st["col"] = np.where(inside, np.round(c), np.nan)
        st["n_valid_days"] = (self.n_valid >= 120).sum(1)    # >= 120 of 144 values
        st.to_csv(os.path.join(out_dir, f"{net}_stations.csv"), index=False)
        np.savez_compressed(os.path.join(out_dir, f"{net}_wet.npz"), sid=sid, t=t, rr=rr)
        np.savez_compressed(os.path.join(out_dir, f"{net}_avail.npz"), n_valid=self.n_valid,
                            days=self.days.values.astype("datetime64[D]"))
        print(f"[{net}] {int((st.n_valid_days > 0).sum())} stations with data "
              f"({int(inside.sum())} of {len(st)} on the radar grid), "
              f"{int(self.n_valid.sum()):,} valid 10-min values, {len(rr):,} wet; "
              f"max {float(rr.max()) if len(rr) else float('nan'):.1f} mm/10 min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", nargs="+", default=["dwd", "smn"], choices=["dwd", "smn"])
    ap.add_argument("--start", default="2012-09-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out_dir", default="/home/fquareng/work/data/extremes/OPERA/validation/gauges")
    ap.add_argument("--workers", type=int, default=2, help="download threads (2-core fetch budget)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for net in a.net:
        print(f"=== {net} ({time.ctime()}) ===", flush=True)
        acc = (fetch_dwd if net == "dwd" else fetch_smn)(a.out_dir, a.start, a.end, a.workers)
        acc.write(net, a.out_dir)
    print(f"=== done ({time.ctime()}) ===", flush=True)


if __name__ == "__main__":
    main()
