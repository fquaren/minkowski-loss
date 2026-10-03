"""Orientation of the DEM against the precipitation grid (the 2026-09-28 bug).

The DEM GeoTIFF is north-first and the radar stores are south-first. These tests pin down
that `src.data.geo` and the preprocessing both hand out the DEM in the precipitation
orientation. They need the real DEM file and are skipped where it is absent.
"""

import os

import numpy as np
import pytest

from src.data import geo

DEM = "/home/fquareng/work/data/extremes/OPERA/europe_dem_laea.tif"
needs_dem = pytest.mark.skipif(not os.path.exists(DEM), reason="DEM file not available")


def test_rowcol_lonlat_roundtrip():
    r, c = geo.lonlat_to_rowcol(8.68, 50.11)          # Frankfurt
    lon, lat = geo.rowcol_to_lonlat(r, c)
    assert abs(float(lon) - 8.68) < 0.03 and abs(float(lat) - 50.11) < 0.03


def test_rows_count_from_the_south():
    _, lat_s = geo.rowcol_to_lonlat(100, 950)
    _, lat_n = geo.rowcol_to_lonlat(2000, 950)
    assert lat_n > lat_s


@needs_dem
def test_dem_is_south_first():
    dem = geo.load_dem_on_radar_grid(DEM)
    for lon, lat, min_m in [(6.865, 45.833, 3000), (14.993, 37.751, 2000)]:   # Mont Blanc, Etna
        r, c = geo.lonlat_to_rowcol(lon, lat)
        assert dem[int(r), int(c)] > min_m
    r, c = geo.lonlat_to_rowcol(3.0, 56.0)                                    # North Sea
    assert not dem[int(r), int(c)] > 0


@needs_dem
def test_preprocessing_uses_oriented_dem():
    from src.data import preprocessing
    src = open(preprocessing.__file__).read()
    assert "load_dem_on_radar_grid" in src, "preprocessing must take the DEM from src.data.geo"
