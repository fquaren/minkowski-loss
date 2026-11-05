#!/usr/bin/env python
"""
Fetch SwissMetNet precipitation from MeteoSwiss Open Government Data (DECISIONS §43, todo B2).

Both open networks: `ch.meteoschweiz.ogd-smn` (automatic weather stations) and
`ch.meteoschweiz.ogd-smn-precip` (automatic precipitation stations). For every station, the 10-min
(`t`) and hourly (`h`) files from 2010 on (decade files `historical_2010-2019`, `historical_2020-2029`,
and `recent`), keeping only `rre150z0` (10-min sum) and `rre150h0` (hourly sum). Timestamps are UTC
(opendatadocs.meteoswiss.ch, a1). Licence: free use, "Source: MeteoSwiss".

Output, under <out>: stations.csv (both networks' metadata), smn_<abbr>.csv.gz with columns
time, network, res ('10min' | '1h'), precip_mm. Already fetched stations are skipped.

Usage: python scripts/data_quality/fetch_smn.py /home/fquareng/work/data/extremes/MCH/smn --workers 2
"""

import argparse
import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = "https://data.geo.admin.ch"
NETS = {"smn": "ogd-smn", "precip": "ogd-smn-precip"}
PARAM = {"t": ("rre150z0", "10min"), "h": ("rre150h0", "1h")}
PERIODS = ["historical_2010-2019", "historical_2020-2029", "recent"]


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def stations():
    out = []
    for net, coll in NETS.items():
        m = pd.read_csv(io.BytesIO(get(f"{BASE}/ch.meteoschweiz.{coll}/{coll}_meta_stations.csv")),
                        sep=";", encoding="latin-1")
        out.append(m.assign(network=net))
    return pd.concat(out, ignore_index=True)


def fetch(abbr, net, out):
    path = os.path.join(out, f"smn_{abbr.upper()}.csv.gz")
    if os.path.exists(path):
        return abbr, "skipped"
    coll, parts = NETS[net], []
    for res, (col, label) in PARAM.items():
        for per in PERIODS:
            url = f"{BASE}/ch.meteoschweiz.{coll}/{abbr.lower()}/{coll}_{abbr.lower()}_{res}_{per}.csv"
            try:
                d = pd.read_csv(io.BytesIO(get(url)), sep=";", usecols=lambda c: c in ("reference_timestamp", col))
            except Exception:
                continue  # not every station has every period or resolution
            if col not in d:
                continue
            d = d.rename(columns={"reference_timestamp": "time", col: "precip_mm"})
            d["time"] = pd.to_datetime(d.time, format="%d.%m.%Y %H:%M")
            parts.append(d.assign(res=label, network=net))
    if not parts:
        return abbr, "no precipitation files"
    d = pd.concat(parts).drop_duplicates(["time", "res"], keep="last").sort_values(["res", "time"])
    d.to_csv(path, index=False)
    return abbr, f"{len(d)} rows"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = stations()
    st.to_csv(os.path.join(a.out, "stations.csv"), index=False)
    jobs = st[["station_abbr", "network"]].drop_duplicates("station_abbr").itertuples(index=False)
    with ThreadPoolExecutor(a.workers) as ex:
        for abbr, msg in ex.map(lambda j: fetch(j[0], j[1], a.out), jobs):
            print(abbr, msg, flush=True)
    print("FETCH_SMN_DONE", flush=True)
