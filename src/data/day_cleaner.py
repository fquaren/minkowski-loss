"""Frame-by-frame cleaning of one day store, with the neighbour frames the repairs need.

One object per day, shared by the scan (`scan_tiles.py`), the store build (`build_store.py`)
and the gauge validation (`gauge_vs_radar.py`), so the three see identical fields.

Chain per frame k (`cleaning`): `clean_frame` (drizzle, static clutter, spikes) ->
`repair_static` (footprints of > 500 mm/h cores, rays, range rings) -> `repair_unsupported`
(cells with no echo within 30 km at k +- 1, judged on the neighbours' `repair_static` output).
The neighbours of the first and last frames come from the adjacent day stores when they
exist and are exactly 15 min away.

`repair=False` stops after `clean_frame` (the v2 field), for comparisons.

v4 (DECISIONS §19), both off by default so the v3 field is reproduced exactly:
`ceilings` (a `radar_screen.CeilingTable`) repairs ceiling pixels right after `clean_frame`
and marks them REPAIR_CEILING; `max_size` guards every ring-median repair (components larger
than this are left as they are and marked REPAIR_REFUSED).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data import cleaning

STEP = np.timedelta64(15, "m")


def open_day(path):
    import xarray as xr
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:                                    # noqa: BLE001
        return xr.open_zarr(path, consolidated=False)


class DayCleaner:
    """`hot(year)` / `ring(year)`: callables returning that year's full-grid masks (or None).
    `sites_rc`: (N, 2) radar positions for the ray repair. `crop`: (y0, y1, x0, x1) to work on
    a window of the grid (all masks are sliced to it)."""

    def __init__(self, raw_dir, day, var, hot, ring=None, sites_rc=None, repair=True,
                 crop=None, keep=4, ceilings=None, max_size=None):
        self.raw_dir, self.day, self.var = raw_dir, str(day), var     # str: may arrive as numpy.str_
        self.hot_fn, self.ring_fn, self.sites = hot, ring, sites_rc
        self.repair, self.keep = repair, keep
        self.ceilings, self.max_size = ceilings, max_size
        self.ds = open_day(os.path.join(raw_dir, self.day))
        self.times = self.ds.time.values
        self.T = len(self.times)
        H, W = self.ds.sizes["y"], self.ds.sizes["x"]
        self.crop = crop or (0, H, 0, W)
        self.year = int(self.day[:4])
        self._raw, self._s1, self._fin, self._maps = {}, {}, {}, {}
        self._edge = {}

    # ------------------------------------------------------------------ helpers
    def _slice(self, a):
        if a is None:
            return None
        y0, y1, x0, x1 = self.crop
        return a[y0:y1, x0:x1]

    def _read(self, ds, k):
        y0, y1, x0, x1 = self.crop
        return ds[self.var].isel(time=k, y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)

    def _stage1(self, raw, year):
        c = cleaning.clean_frame(raw, self._slice(self.hot_fn(year)))[0]
        if not self.repair:
            return c, np.zeros(c.shape, np.uint8)
        ceil = None
        if self.ceilings is not None:
            from src.data.radar_screen import repair_ceiling
            ceil = self.ceilings.mask(c, year, self.crop)
            repair_ceiling(c, ceil)
        ring = self._slice(self.ring_fn(year)) if self.ring_fn else None
        z, code = cleaning.repair_static(c, ring, self.sites, row0=self.crop[0], col0=self.crop[2],
                                         max_size=self.max_size)
        if ceil is not None:
            code[ceil] |= cleaning.REPAIR_CEILING
        return z, code

    def _edge_frame(self, which):
        """Stage-1 frame of the previous day's last (which=-1) or next day's first (which=+1)
        step, if it exists and is exactly 15 min away; else None."""
        if which in self._edge:
            return self._edge[which]
        d = pd.Timestamp(self.day) + pd.Timedelta(days=which)
        p = os.path.join(self.raw_dir, d.strftime("%Y%m%d"))
        out = None
        if os.path.exists(os.path.join(p, ".zmetadata")):
            ds = open_day(p)
            k = ds.sizes["time"] - 1 if which < 0 else 0
            t = ds.time.values[k]
            gap = (self.times[0] - t) if which < 0 else (t - self.times[-1])
            if gap == STEP:
                out = self._stage1(self._read(ds, k), d.year)[0]
            ds.close()
        self._edge[which] = out
        return out

    def _evict(self, k):
        for cache in (self._raw, self._s1, self._fin, self._maps):
            for j in [j for j in cache if j < k - self.keep or j > k + self.keep]:
                del cache[j]

    # ------------------------------------------------------------------ public
    def raw(self, k):
        if k not in self._raw:
            self._raw[k] = self._read(self.ds, k)
        return self._raw[k]

    def stage1(self, k):
        """(frame, code) after clean_frame + repair_static; k may be -1 or T (neighbour days)."""
        if k < 0 or k >= self.T:
            f = self._edge_frame(-1 if k < 0 else 1)
            return (f, None) if f is not None else (None, None)
        if k not in self._s1:
            self._s1[k] = self._stage1(self.raw(k), self.year)
        return self._s1[k]

    def _support(self, k):
        if k not in self._maps:
            f = self.stage1(k)[0]
            self._maps[k] = None if f is None else cleaning.support_maps(f)
        return self._maps[k]

    def clean(self, k):
        """(final cleaned frame, repair code map) for frame k of this day."""
        if k in self._fin:
            return self._fin[k]
        self._evict(k)
        s1, code = self.stage1(k)
        if not self.repair:
            self._fin[k] = (s1, code)
            return self._fin[k]
        t = self.times[k]
        tp = self.times[k - 1] if k > 0 else t - STEP
        tn = self.times[k + 1] if k + 1 < self.T else t + STEP
        prev = self._support(k - 1) if t - tp == STEP else None
        nxt = self._support(k + 1) if tn - t == STEP else None
        self._fin[k] = cleaning.repair_unsupported(s1, prev, nxt, code, max_size=self.max_size)
        return self._fin[k]

    def close(self):
        self.ds.close()
