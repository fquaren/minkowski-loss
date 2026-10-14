#!/usr/bin/env python
"""Why OPERA reads below the gauges (methods draft §6.5): gauge, RADKLIM and screened OPERA on the
same DWD gauge-hours (`calib/gauge_gate`, days with RADKLIM). RADKLIM is gauge-adjusted and averaged
to the same 2 km pixel, so where it also falls below the gauges the gap is mostly point-vs-area
support; OPERA's further gap below RADKLIM is radar measurement bias.

    python scripts/data_quality/support_gap.py      -> calib/support_gap/summary.txt
"""
import contextlib
import glob
import io
import os

import numpy as np
import pandas as pd

D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
st = pd.read_csv(f"{D}/validation/gauges/dwd_stations.csv", dtype={"station": str})[["station", "row", "col"]].dropna()
rows = []
for f in sorted(glob.glob(f"{CAL}/gauge_gate/g_*.csv.gz")):
    day = os.path.basename(f)[2:10]
    rk = f"{D}/validation/radklim/hourly/{day}.npz"
    if not os.path.exists(rk):
        continue
    z = np.load(rk); K, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K).any():
        continue
    g = pd.read_csv(f, dtype={"day": str, "end": str, "station": str})
    g = g[g.net == "dwd"].merge(st, on="station")
    i = ((pd.to_datetime(g.end, format="%Y%m%d%H%M") - (pd.Timestamp(day) + pd.Timedelta(hours=1))) / pd.Timedelta(hours=1)).astype(int).values
    r_, c_ = g.row.astype(int).values, g.col.astype(int).values
    ok = (i >= 0) & (i < 24) & (r_ >= y0) & (r_ < y1) & (c_ >= x0) & (c_ < x1)
    k = np.full(len(g), np.nan); k[ok] = K[i[ok], r_[ok] - y0, c_[ok] - x0]
    rows.append(g.assign(K=k)[["g_shift", "K", "full", "raw"]])
x = pd.concat(rows).dropna()
buf = io.StringIO()
_ctx = contextlib.redirect_stdout(buf)
_ctx.__enter__()
print(f"{len(x):,} DWD gauge-hours with gauge, RADKLIM and OPERA")
print("\nmean (mm/h): gauge {:.4f}, RADKLIM {:.4f}, OPERA screened {:.4f}".format(x.g_shift.mean(), x.K.mean(), x.full.mean()))
print("\nquantile   gauge  RADKLIM  OPERA-scr   RADKLIM/gauge  OPERA/gauge  OPERA/RADKLIM")
for p in (0.99, 0.999, 0.9999, 0.99999):
    g_, k_, o_ = (np.quantile(x[c], p) for c in ("g_shift", "K", "full"))
    print(f"q{p * 100:<8g} {g_:6.1f} {k_:8.1f} {o_:9.1f}   {k_ / g_:13.2f} {o_ / g_:11.2f} {o_ / k_:13.2f}")
print("\nexceedance count  gauge  RADKLIM  OPERA-scr")
for u in (1, 5, 10, 20, 30):
    print(f">= {u:<3} mm  {int((x.g_shift >= u).sum()):>10,} {int((x.K >= u).sum()):>8,} {int((x.full >= u).sum()):>9,}")
# conditional: on hours where the gauge is heavy, what do the area products read?
h = x[x.g_shift >= 20]
print(f"\ngauge >= 20 mm ({len(h)} hours): median RADKLIM/gauge {np.median(h.K / h.g_shift):.2f}, "
      f"median OPERA/gauge {np.median(h.full / h.g_shift):.2f}")
_ctx.__exit__(None, None, None)
os.makedirs(f"{CAL}/support_gap", exist_ok=True)
open(f"{CAL}/support_gap/summary.txt", "w").write(buf.getvalue())
print(buf.getvalue())
