"""Geography of the OPERA radar grid: orientation, coordinates, DEM, radar sites.

The grid is the 2 km OPERA composite grid, 2200 rows x 1900 columns, in

    +proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84

with pixel edges x in [0, 3_800_000] m and y in [-4_400_000, 0] m.

**Orientation.** The precipitation stores (`raw/OPERA/YYYYMMDD/`, and every patch cut from
them) have **row 0 at the south** (y ascending). This was checked against geography, not only
against the coordinate labels: the radar-coverage footprint overlaps the union of the OPERA
radars' ranges with IoU 0.878 in this orientation and 0.554 flipped (2026-09-28).

The DEM GeoTIFF (`STATIC_DEM_PATH`) is the opposite: **row 0 at the north**, as GeoTIFFs
usually are (Mont Blanc reads 4164 m only in that orientation). Slicing both arrays with the
same `(y_start, x_start)` therefore pairs each precipitation patch with the DEM of the
N-S-mirrored place. That is what `src/data/preprocessing.py` does, so the DEM channel of the
current patch store is geographically wrong (see EXPERIMENTS §5). Use `load_dem_on_radar_grid`
here, which returns the DEM in the precipitation orientation.

Also note that the precipitation stores label their axes with a linspace over the extent (a
nominal 2001.05 m spacing), not with true pixel centres. Coordinates are therefore computed
here from the row/column index on the true 2 km grid, never from the store's `x`/`y` arrays.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import numpy as np

PROJ = ("+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 "
        "+units=m +ellps=WGS84")
H, W = 2200, 1900
RES = 2000.0
X0 = 0.0             # west edge
Y0 = -4_400_000.0    # south edge (row 0 is the southernmost row)


# ----------------------------------------------------------------------------------
# index <-> projected <-> geographic
# ----------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _transformers():
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    inv = Transformer.from_crs(PROJ, "EPSG:4326", always_xy=True)
    return fwd, inv


def rowcol_to_xy(row, col):
    """Projected coordinates (m) of pixel centres; rows count from the south."""
    row, col = np.asarray(row, dtype=float), np.asarray(col, dtype=float)
    return X0 + (col + 0.5) * RES, Y0 + (row + 0.5) * RES


def xy_to_rowcol(x, y):
    """Pixel index containing a projected point (may fall outside the grid)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (np.floor((y - Y0) / RES).astype(int), np.floor((x - X0) / RES).astype(int))


def rowcol_to_lonlat(row, col):
    x, y = rowcol_to_xy(row, col)
    return _transformers()[1].transform(x, y)


def lonlat_to_rowcol(lon, lat):
    x, y = _transformers()[0].transform(lon, lat)
    return xy_to_rowcol(x, y)


def tile_location(row: int, col: int, patch: int = 128) -> dict:
    """Centre and corner coordinates of a tile whose south-west pixel is (row, col)."""
    lon_c, lat_c = rowcol_to_lonlat(row + patch / 2 - 0.5, col + patch / 2 - 0.5)
    corners = rowcol_to_lonlat(np.array([row, row, row + patch - 1, row + patch - 1]),
                               np.array([col, col + patch - 1, col, col + patch - 1]))
    lons, lats = np.asarray(corners[0]), np.asarray(corners[1])
    return {"lat": float(lat_c), "lon": float(lon_c),
            "lat_min": float(lats.min()), "lat_max": float(lats.max()),
            "lon_min": float(lons.min()), "lon_max": float(lons.max())}


def format_latlon(lat: float, lon: float) -> str:
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'} "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


# ----------------------------------------------------------------------------------
# DEM, in the precipitation orientation
# ----------------------------------------------------------------------------------
def load_dem_on_radar_grid(path: str, verbose: bool = False) -> np.ndarray:
    """The DEM as a (2200, 1900) float32 array with **row 0 at the south**.

    Orientation is decided from the file's own y coordinates, so it is right whichever way the
    GeoTIFF is written. The residual offset between the DEM pixel centres and the radar grid
    is checked to be well under half a pixel (it is ~0.26 px in x and ~0.07 px in y for
    `europe_dem_laea.tif`), which makes an index-for-index pairing correct.
    """
    import xarray as xr
    with xr.open_dataset(path, engine="rasterio") as ds:
        da = ds["band_data"].isel(band=0)
        dem = da.values.astype(np.float32)
        ys, xs = da["y"].values.astype(float), da["x"].values.astype(float)
    if dem.shape != (H, W):
        raise ValueError(f"DEM shape {dem.shape} != radar grid {(H, W)}")
    if ys[0] > ys[-1]:                       # north first: flip to south first
        dem, ys = dem[::-1].copy(), ys[::-1]
    dx = float(xs[0] - (X0 + RES / 2))
    dy = float(ys[0] - (Y0 + RES / 2))
    if max(abs(dx), abs(dy)) > 0.5 * RES:
        raise ValueError(f"DEM is offset from the radar grid by ({dx:.0f}, {dy:.0f}) m, "
                         f"more than half a pixel; resample it instead of slicing")
    if verbose:
        print(f"[geo] DEM oriented south-first; centre offset vs radar grid "
              f"dx={dx:+.0f} m, dy={dy:+.0f} m")
    return dem


# ----------------------------------------------------------------------------------
# OPERA radar sites
# ----------------------------------------------------------------------------------
DEFAULT_RADAR_DB = "/home/fquareng/work/data/extremes/OPERA/meta/OPERA_RADARS_DB.json"
# Same file as config.yaml STATIC_DEM_PATH; the default for the dataset classes.
DEFAULT_DEM_PATH = "/home/fquareng/work/data/extremes/OPERA/europe_dem_laea.tif"


@lru_cache(maxsize=4)
def dem_cached(path: str = DEFAULT_DEM_PATH) -> np.ndarray:
    """`load_dem_on_radar_grid`, read once per process (17 MB). Treat as read-only."""
    dem = load_dem_on_radar_grid(path)
    dem.setflags(write=False)
    return dem


def dem_patch(row: int, col: int, patch: int = 128, path: str = DEFAULT_DEM_PATH) -> np.ndarray:
    """The DEM under the tile whose south-west pixel is (row, col), in metres."""
    return dem_cached(path)[row:row + patch, col:col + patch]


def load_radar_sites(path: Optional[str] = None, include_archive: bool = True):
    """OPERA radar database as a DataFrame with grid positions.

    Source: the OPERA radar database (`OPERA_RADARS_DB.json`, and the archive of retired
    sites `OPERA_RADARS_ARH_DB.json` next to it), downloaded from
    https://www.eumetnet.eu/.../opera/database/OPERA_Database/ . Retired sites matter for
    the older years of the archive.
    """
    import pandas as pd
    path = path or DEFAULT_RADAR_DB
    rows = []
    files = [(path, "current")]
    arh = path.replace("OPERA_RADARS_DB.json", "OPERA_RADARS_ARH_DB.json")
    if include_archive and arh != path and os.path.exists(arh):
        files.append((arh, "archive"))
    for f, source in files:
        for r in json.load(open(f)):
            try:
                lat, lon = float(r["latitude"]), float(r["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            rng = r.get("maxrange")
            try:
                rng = float(rng)
            except (TypeError, ValueError):
                rng = np.nan
            rows.append({
                "odim": (r.get("odimcode") or "").strip(),
                "location": (r.get("location") or "").strip(),
                "country": (r.get("country") or "").strip(),
                "lat": lat, "lon": lon,
                "band": r.get("band"), "polarization": r.get("polarization"),
                "maxrange_km": rng, "status": r.get("status"),
                "startyear": r.get("startyear"), "finishyear": r.get("finishyear"),
                "source": source,
            })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["odim", "lat", "lon"]).reset_index(drop=True)
    x, y = _transformers()[0].transform(df["lon"].values, df["lat"].values)
    df["x"], df["y"] = x, y
    df["row"], df["col"] = xy_to_rowcol(x, y)
    return df


def nearest_radar(x, y, sites, max_km: float = 250.0):
    """Index into `sites` of the nearest radar within `max_km`, and its distance in km.

    Returns (-1, nan) beyond `max_km`. Nearest-radar attribution is an approximation: the
    composite takes each pixel from the radar(s) chosen by quality or elevation, which is
    usually but not always the nearest one.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[sites["x"].values, sites["y"].values])
    d, i = tree.query(np.c_[np.ravel(x), np.ravel(y)])
    d = d / 1000.0
    i = np.where(d <= max_km, i, -1)
    d = np.where(d <= max_km, d, np.nan)
    return i.reshape(np.shape(x)), d.reshape(np.shape(x))


def describe_location(row: int, col: int, patch: int = 128, sites=None) -> str:
    """One-line label: tile centre in lat/lon, plus the nearest radar if sites are given."""
    loc = tile_location(row, col, patch)
    s = format_latlon(loc["lat"], loc["lon"])
    if sites is not None and len(sites):
        x, y = rowcol_to_xy(row + patch / 2 - 0.5, col + patch / 2 - 0.5)
        i, d = nearest_radar(np.array([x]), np.array([y]), sites, max_km=1e9)
        r = sites.iloc[int(i[0])]
        s += f" · nearest radar {r['odim'] or r['location']} ({r['country']}) {d[0]:.0f} km"
    return s
