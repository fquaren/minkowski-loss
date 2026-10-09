#!/usr/bin/env python
"""Which radars actually contribute to the OPERA composite, month by month (2026-10-09).

The OPERA radar database lists every national radar of the member countries, whether or not it
delivers data to the composite: no Italian radar is among the contributing nodes of the 2018
(ODYSSEY) or 2025 (NIMBUS) files, and the archive has no data over most of Italy (DECISIONS
§19 status). The ownership geometry of the radar screen (`radar_screen.RadarGeometry`) must
therefore use the contributing radars only.

Each composite records its contributors in the HDF5 attribute `how/nodes` ("NOD:dedrs,NOD:..."
= ODIM codes; NIMBUS writes "'behel','bejab',..."). This script downloads one composite per month
(the first day of the month with a file, at 12:00 UTC or the nearest listed step) from the openradar archive and writes
<out>/opera_nodes.csv: month, timestamp, product, n_nodes, nodes (';'-joined ODIM codes).

    python scripts/data/opera_nodes.py --start 2012-09 --end 2026-10
"""

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

BASE = "https://s3.waw3-1.cloudferro.com/openradar-archive"
OUT = "/home/fquareng/work/data/extremes/OPERA/validation/opera_nodes"
PRODUCTS = ("QIND_RATE", "RATE", "ACRR_QIND", "ACRR")


def nodes_of(day, hhmm_list=("1200", "1215", "1230", "0000", "0600", "1800")):
    import h5py
    for hhmm in hhmm_list:
        for p in PRODUCTS:
            url = f"{BASE}/{day:%Y/%m/%d}/OPERA/COMP/OPERA@{day:%Y%m%d}T{hhmm}@0@{p}.h5"
            try:
                with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
                    urllib.request.urlretrieve(url, tmp.name)
                    with h5py.File(tmp.name, "r") as f:
                        n = f["how"].attrs.get("nodes", b"")
                        n = n.decode() if isinstance(n, bytes) else str(n)
                # ODYSSEY: "NOD:dedrs,NOD:..."; NIMBUS: "'behel','bejab',..."
                codes = {t.strip().strip("'\"").replace("NOD:", "") for t in n.split(",")}
                return f"{day:%Y%m%d}{hhmm}", p, sorted(c for c in codes if re.fullmatch(r"[a-z]{2}[a-z0-9]{3}", c))
            except Exception:                                            # noqa: BLE001
                continue
    return None, None, []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2012-09")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    y, m = map(int, a.start.split("-"))
    y1, m1 = map(int, a.end.split("-"))
    rows = []
    while (y, m) <= (y1, m1):
        ts, p, nodes = None, None, []
        for dd in range(1, 8):                     # the archive starts 2012-09-04; take the first day with a file
            ts, p, nodes = nodes_of(dt.date(y, m, dd))
            if nodes:
                break
        rows.append(f"{y:04d}-{m:02d},{ts or ''},{p or ''},{len(nodes)},{';'.join(nodes)}")
        print(f"  {y:04d}-{m:02d}: {len(nodes)} nodes ({p})", flush=True)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    with open(os.path.join(a.out, "opera_nodes.csv"), "w") as f:
        f.write("month,timestamp,product,n_nodes,nodes\n" + "\n".join(rows) + "\n")
    print(f"[opera_nodes] {len(rows)} months -> {a.out}/opera_nodes.csv")


if __name__ == "__main__":
    main()
