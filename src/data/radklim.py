"""DWD RADKLIM RW (v2017.002): hourly, gauge-adjusted, climatologically corrected radar
precipitation over Germany on the 1 km RADOLAN 1100x900 grid. Read the binary product and map it
onto the OPERA 2 km LAEA grid (DECISIONS §22, §24: an independent reference for the screen).

Format (one file per hour, `raa01-rw2017.002_10000-YYMMDDHH50-dwd---bin`): an ASCII header ending
with ETX (0x03), e.g. `RW141550...PR E-01INT  60U0GP1100x 900...VR2017.002MS ...`, then
1100 x 900 little-endian uint16, first row = southern edge (the coordinate file of DWD numbers
rows 0..1099 from the south). Bits 0-11 hold the value in units of the `PR` precision (0.1 mm);
bit 13 (0x2000) marks no data, bit 15 (0x8000) clutter, bit 14 (0x4000) a negative value, bit 12
(0x1000) an interpolated value. The file labelled HH:50 holds the sum over HH-1:50 .. HH:50 UTC.

Mapping: every 1 km pixel centre (DWD's coordinate file, `radolan-coords_center_1100x900.txt`)
is assigned to the OPERA pixel that contains it, and an OPERA pixel takes the mean of its ~4
RADKLIM pixels when all of them are valid (`radklim_to_opera`).
"""

from __future__ import annotations

import os

import numpy as np

NROWS, NCOLS = 1100, 900
ROOT = "/home/fquareng/work/data/extremes/OPERA/validation/radklim"
COORDS = os.path.join(ROOT, "grid", "x", "radolan-coords_center_1100x900.txt")
MAP = os.path.join(ROOT, "grid", "radklim_to_opera.npz")


def read_rw(path: str):
    """(field mm per hour, NaN = no data or clutter; header str)."""
    b = open(path, "rb").read()
    i = b.index(b"\x03")
    hdr = b[:i].decode("latin-1")
    prec = 0.1
    if "PR E" in hdr:
        prec = 10.0 ** int(hdr.split("PR E")[1][:3])
    raw = np.frombuffer(b[i + 1:i + 1 + 2 * NROWS * NCOLS], "<u2").reshape(NROWS, NCOLS)
    val = (raw & 0x0FFF).astype(np.float32) * prec
    val[(raw & 0x4000) > 0] *= -1
    val[((raw & 0x2000) > 0) | ((raw & 0x8000) > 0)] = np.nan
    return val, hdr


def build_map(out: str = MAP):
    """Index map RADKLIM pixel -> OPERA (row, col), from DWD's pixel-centre coordinates."""
    import pandas as pd
    from src.data import geo
    c = pd.read_csv(COORDS, dtype={"y_x": str})
    yx = c["y_x"].str.split("_", expand=True).astype(int).values
    r, k = geo.lonlat_to_rowcol(c["LON"].values, c["LAT"].values)       # containing pixel
    src = yx[:, 0] * NCOLS + yx[:, 1]
    np.savez_compressed(out, src=src.astype(np.int32), row=r.astype(np.int32), col=k.astype(np.int32))
    return out


_M = None


def radklim_to_opera(field: np.ndarray, shape=(2200, 1900), min_n: int = 3):
    """Mean of the RADKLIM pixels whose centres fall in each OPERA pixel; NaN where fewer than
    `min_n` valid ones (an OPERA pixel holds ~4) or outside the domain."""
    global _M
    if _M is None:
        if not os.path.exists(MAP):
            build_map()
        _M = dict(np.load(MAP))
    r, c, s = _M["row"], _M["col"], _M["src"]
    v = field.ravel()[s]
    ok = np.isfinite(v) & (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    idx = r[ok] * shape[1] + c[ok]
    n = np.bincount(idx, minlength=shape[0] * shape[1])
    tot = np.bincount(idx, weights=v[ok], minlength=shape[0] * shape[1])
    out = np.full(shape[0] * shape[1], np.nan, np.float32)
    good = n >= min_n
    out[good] = tot[good] / n[good]
    return out.reshape(shape)
