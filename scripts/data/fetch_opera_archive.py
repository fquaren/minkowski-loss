#!/usr/bin/env python
"""Fetch OPERA composites from the EUMETNET open archive onto the existing patch grid.

The archive bucket is public and unsigned: no API key, no credentials, no boto3. The
MeteoGate EDR API needs a key but only covers a 24-hour rolling window and returns links
rather than data, so it is useful for discovery only and is not used here.

    bucket  openradar-archive          2012 .. present  (openradar-24h holds the last 24 h)
    layout  <bucket>/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@<PRODUCT>.h5

`QIND_RATE` carries two ODIM datasets: the surface rain rate in mm/h and the OPERA total
quality index in [0,1]. Both are kept. The quality index is the point of using this product:
on a test composite 21% of valid pixels scored below 0.8, but 42% of pixels at or above
31 mm/h did, so the flag is roughly twice as common exactly where the tail statistics are
computed. Screening on it is what the archive makes possible and the current patch set
cannot do, because the existing store keeps the rate field alone.

Two modes:

    --list      print what the archive holds over the range and stop. Nothing is downloaded.
    (default)   download, reproject onto the reference grid, and write one zarr store per day.

The output matches the layout the preprocessing already reads -- `<out>/YYYYMMDD/` holding
`TOT_PREC(time, y, x)` -- with `QIND(time, y, x)` beside it, so `process_batch` can slice
patches from it unchanged.

Reprojection is grid-aware rather than unconditional. Modern composites are already on the
2 km LAEA grid the patches use (2200x1900, `+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000
+y_0=-2100000`), so they pass through untouched; the OPERA domain has changed over the life
of the archive, and older years are warped onto the reference. Nearest-neighbour is the
default because bilinear smooths precipitation peaks, and peaks are the object of study.

Note for whoever rebuilds the patch set: `src/data/preprocessing.py::filter_precip_bounds`
does not clip at `DECLUTTER_THRESHOLD`, it sets everything above it to **zero**. A genuine
200 mm/h cell currently becomes a dry pixel. This script deliberately stores the rate field
unaltered and leaves that decision to preprocessing.

Usage:
  # what does the archive hold for this week?
  python scripts/data/fetch_opera_archive.py --list --start 2018-06-10 --end 2018-06-16

  # fetch it onto the patch grid, every 30 minutes
  python scripts/data/fetch_opera_archive.py --start 2018-06-10 --end 2018-06-16 \
      --every 30 --out /home/fquareng/work/data/extremes/OPERA/raw/OPERA_archive
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

S3_HOST = "https://s3.waw3-1.cloudferro.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
KEY_RE = re.compile(r"OPERA@(\d{8})T(\d{4})@\d+@(?P<product>[A-Z_]+)\.h5$")

# The grid the existing patches are cut from. Used only when no --reference is given.
REF_PROJ = ("+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 "
            "+units=m +ellps=WGS84")
REF_SHAPE = (2200, 1900)          # (y, x)
REF_RES = 2000.0                  # m


# ----------------------------------------------------------------------------------
# S3: listing and download, over plain HTTPS
# ----------------------------------------------------------------------------------
def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """GET with linear backoff. The bucket is public, so no signing is involved."""
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                raise                      # a missing key will not appear on a retry
            if attempt < retries - 1:
                import time
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last})")


def s3_list(bucket: str, prefix: str) -> list:
    """Every key under a prefix, following continuation tokens. Returns (key, size)."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{S3_HOST}/{bucket}?{urllib.parse.urlencode(q)}"
        root = ET.fromstring(_get(url))
        for c in root.findall("s3:Contents", S3_NS):
            key = c.findtext("s3:Key", default="", namespaces=S3_NS)
            size = int(c.findtext("s3:Size", default="0", namespaces=S3_NS))
            out.append((key, size))
        if root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=S3_NS)
        if not token:
            return out


def download(bucket: str, key: str, dest: str) -> str:
    """Download one key, writing through a temporary file so a partial file is never left."""
    url = f"{S3_HOST}/{bucket}/{urllib.parse.quote(key)}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(_get(url, timeout=180))
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


# ----------------------------------------------------------------------------------
# ODIM HDF5
# ----------------------------------------------------------------------------------
def _attr(group, name, default=None):
    v = group.attrs.get(name, default)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.item()
    return v


def _decode(raw: np.ndarray, what) -> np.ndarray:
    """Apply ODIM gain/offset and turn the two sentinels into their physical meanings.

    ``undetect`` is a real measurement of no echo and becomes 0.0; ``nodata`` is an absence
    of coverage and becomes NaN. Collapsing nodata to zero would turn every gap in the radar
    network into a confident report of dry weather.
    """
    gain = float(_attr(what, "gain", 1.0)) if what is not None else 1.0
    off = float(_attr(what, "offset", 0.0)) if what is not None else 0.0
    arr = raw * gain + off
    if what is not None:
        if "nodata" in what.attrs:
            arr[np.isclose(raw, float(_attr(what, "nodata")))] = np.nan
        if "undetect" in what.attrs:
            arr[np.isclose(raw, float(_attr(what, "undetect")))] = 0.0
    return arr.astype(np.float32)


def read_odim(path: str):
    """Physical RATE and QIND arrays plus the source grid, from one ODIM composite.

    Returns ``(fields, grid)`` where ``fields`` maps a quantity name to a float32 array and
    ``grid`` is ``(proj4, shape, (west, south, east, north))``.

    ODIM encodes two sentinels that mean different things and must not be conflated:
    ``undetect`` is a real measurement of no echo and becomes 0.0, while ``nodata`` is an
    absence of coverage and becomes NaN. Collapsing nodata to zero would turn every gap in
    the radar network into a confident report of dry weather.
    """
    import h5py                                # imported late: see _require_h5py()
    from pyproj import CRS, Transformer

    fields = {}
    with h5py.File(path, "r") as f:
        where = f["where"]
        proj4 = _attr(where, "projdef")
        xsize, ysize = int(_attr(where, "xsize")), int(_attr(where, "ysize"))

        crs = CRS.from_proj4(proj4)
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        ll = tf.transform(float(_attr(where, "LL_lon")), float(_attr(where, "LL_lat")))
        ur = tf.transform(float(_attr(where, "UR_lon")), float(_attr(where, "UR_lat")))
        bounds = (ll[0], ll[1], ur[0], ur[1])

        for dname in sorted(k for k in f if k.startswith("dataset")):
            dgrp = f[dname]
            for sub in sorted(k for k in dgrp if k.startswith("data")):
                node = dgrp[sub]
                if "data" not in node:
                    continue
                # ODIM puts `what` at the data level in the live products but at the
                # dataset level in the archive ones, where `datasetN/data1/what` is absent
                # altogether. Resolve both: reading gain/offset from only one of the two
                # leaves the -9999000 / -8888000 sentinels in the array as though they were
                # rain, which silently produces a field with no nodata and a five-figure
                # maximum.
                what = node.get("what")
                if what is None or "nodata" not in what.attrs:
                    what = dgrp.get("what", what)

                quantity = _attr(what, "quantity") if what is not None else None
                if quantity is None:
                    task = _attr(node["how"], "task", "") if "how" in node else ""
                    quantity = "QIND" if "qi" in str(task).lower() else dname

                raw = np.asarray(node["data"][:], dtype=np.float64)
                fields[str(quantity)] = _decode(raw, what)

                # the quality layer hangs off the rate dataset rather than standing alone
                for q in sorted(k for k in node if k.startswith("quality")):
                    qn = node[q]
                    if "data" not in qn:
                        continue
                    fields.setdefault(
                        "QIND",
                        _decode(np.asarray(qn["data"][:], dtype=np.float64),
                                qn.get("what")))

    return fields, (proj4, (ysize, xsize), bounds)


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError:
        sys.exit(
            "h5py is required to read ODIM composites and is not installed in this "
            "environment.\n"
            "    micromamba install -n dl-stable -c conda-forge h5py\n"
            "Listing (--list) works without it.")


# ----------------------------------------------------------------------------------
# reference grid and reprojection
# ----------------------------------------------------------------------------------
def reference_grid(path: str = None):
    """The grid patches are cut from: ``(crs, transform, shape)``.

    Taking it from an existing store rather than hard-coding it is what guarantees a new day
    lines up with the old ones: patch metadata is ``(timestamp, y_start, x_start)``, so a
    half-pixel disagreement silently shifts every patch against its DEM.
    """
    from rasterio.crs import CRS as RCRS
    from rasterio.transform import from_origin

    if path:
        if os.path.isdir(path):                                   # an existing zarr day
            import xarray as xr
            ds = xr.open_zarr(path, consolidated=False)
            x, y = ds["x"].values, ds["y"].values
            res_x = float(abs(x[1] - x[0]))
            res_y = float(abs(y[1] - y[0]))
            # zarr stores y ascending; rasterio transforms are north-up
            transform = from_origin(float(x.min()) - res_x / 2,
                                    float(y.max()) + res_y / 2, res_x, res_y)
            shape = (y.size, x.size)
            crs = RCRS.from_proj4(REF_PROJ)
            ds.close()
            return crs, transform, shape
        import rioxarray  # noqa: F401
        import xarray as xr
        ds = xr.open_dataset(path, engine="rasterio")             # the DEM GeoTIFF
        da = ds["band_data"].isel(band=0)
        crs = RCRS.from_wkt(ds.rio.crs.to_wkt())
        return crs, da.rio.transform(), (da.sizes["y"], da.sizes["x"])

    crs = RCRS.from_proj4(REF_PROJ)
    transform = from_origin(0.0, 0.0, REF_RES, REF_RES)
    return crs, transform, REF_SHAPE


def _src_transform(grid):
    from rasterio.transform import from_bounds
    _, (ny, nx), (w, s, e, n) = grid
    return from_bounds(w, s, e, n, nx, ny)


def same_grid(grid, ref, tol_px: float = 1.0) -> bool:
    """True when the composite is the reference grid and should be copied, not resampled.

    The test is CRS, shape, and origin agreeing to within ``tol_px`` pixels -- deliberately
    not exact transform equality. The existing stores label their axes with a linspace across
    the full domain (1900 points spanning 0..3_800_000 m, so a nominal spacing of 2001.05 m)
    rather than with true 2 km pixel centres, so a strict comparison never matches and every
    day would be resampled onto a grid that is only an imprecise labelling of itself.

    Measured on 2018-06-15T12:00, that needless warp altered 0.09% of pixels -- a one-pixel
    duplication at the edges where the spacing mismatch accumulates -- leaving the wet-pixel
    count and the maximum unchanged. Harmless in isolation, but the existing 436 days were
    built by index-for-index correspondence (patches are cut by ``y_start``/``x_start`` from
    precipitation and DEM alike, so index is what alignment means here), and a fetched day
    must not sit half a pixel off from them. Passing through preserves that exactly, and is
    much faster over five thousand days.

    Genuine grid changes in the older archive come with a change of shape, which still warps.
    """
    from rasterio.crs import CRS as RCRS
    ref_crs, ref_tf, ref_shape = ref
    if grid[1] != ref_shape:
        return False
    try:
        if not RCRS.from_proj4(grid[0]) == ref_crs:
            return False
    except Exception:
        return False
    src = _src_transform(grid)
    px = max(abs(ref_tf.a), abs(ref_tf.e))
    return (abs(src.c - ref_tf.c) <= tol_px * px
            and abs(src.f - ref_tf.f) <= tol_px * px)


def reproject_to_reference(arr, grid, ref, resampling: str = "nearest"):
    """Warp one field onto the reference grid, north-up, with NaN outside coverage."""
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS as RCRS

    ref_crs, ref_tf, ref_shape = ref
    dst = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(arr),
        destination=dst,
        src_transform=_src_transform(grid),
        src_crs=RCRS.from_proj4(grid[0]),
        dst_transform=ref_tf,
        dst_crs=ref_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=getattr(Resampling, resampling),
    )
    return dst


# ----------------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------------
def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def resolve_product(available, priority):
    """Pick the rate product this day actually carries, honouring the priority order.

    The archive renamed its products partway through: days up to the end of 2024 hold
    `QIND_RATE`, and from 2025-01-01 the same content is published as plain `RATE`, with the
    quality index moved into the rate file as an ODIM `quality` subgroup rather than being
    named in the filename. Both yield RATE + QIND once read, so resolving per day is what
    lets one command span the whole archive.
    """
    for p in priority:
        if p in available:
            return p
    return None


def plan(bucket: str, start: dt.date, end: dt.date, product, every: int, priority):
    """(day -> [(timestamp, key, size)]) for the range, filtered to the requested cadence.

    ``product`` may be None, meaning resolve it per day from ``priority``.
    """
    per_day, chosen = {}, {}
    for day in daterange(start, end):
        prefix = f"{day:%Y/%m/%d}/OPERA/COMP/"
        try:
            keys = s3_list(bucket, prefix)
        except Exception as e:
            print(f"  [warn] {day}: listing failed ({e})")
            per_day[day] = []
            continue

        parsed = []
        for key, size in keys:
            m = KEY_RE.search(key)
            if m:
                parsed.append((m.group("product"), m, key, size))

        want = product or resolve_product({p for p, _, _, _ in parsed}, priority)
        chosen[day] = want
        rows = []
        for pname, m, key, size in parsed:
            if pname != want:
                continue
            stamp = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
            if every and (stamp.hour * 60 + stamp.minute) % every:
                continue
            rows.append((stamp, key, size))
        per_day[day] = sorted(rows)
    return per_day, chosen


def products_seen(bucket: str, day: dt.date):
    """Distinct product names present on a day, for --list and for diagnosing an empty plan."""
    names = {}
    for key, size in s3_list(bucket, f"{day:%Y/%m/%d}/OPERA/COMP/"):
        m = KEY_RE.search(key)
        if m:
            p = m.group("product")
            names[p] = names.get(p, 0) + 1
    return names


# ----------------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------------
# The encoding of the existing raw stores, reproduced exactly so a fetched day is
# indistinguishable from one already there. Verified against
# raw/OPERA/20230801/{TOT_PREC,x,y,time}/.zarray: blosc-lz4 clevel 5 with byte shuffle,
# float64, one chunk per time step, NaN fill, consolidated metadata, and a time axis stored
# as int64 minutes since midnight of that day.
STORE_DTYPE = "float64"
STORE_CNAME = "lz4"
STORE_CLEVEL = 5


def store_complete(path: str) -> bool:
    """True when a day store exists and was fully written.

    Zarr writes `.zmetadata` at the end of `to_zarr(consolidated=True)`, so a directory
    holding the data arrays but no `.zmetadata` is the remains of an interrupted run.
    """
    return os.path.exists(os.path.join(path, ".zmetadata"))


def write_day(day, stamps, arrays, ref, out_dir, dtype=STORE_DTYPE, compress=True,
              overwrite=False):
    """One zarr store per day, byte-for-byte in the layout the existing stores use."""
    import xarray as xr
    from rasterio.transform import xy

    ref_crs, ref_tf, (ny, nx) = ref
    xs = np.array([xy(ref_tf, 0, i, offset="center")[0] for i in range(nx)])
    ys = np.array([xy(ref_tf, j, 0, offset="center")[1] for j in range(ny)])

    # the existing stores index y ascending; flip so patch offsets mean the same thing
    flip = ys[0] > ys[-1]
    if flip:
        ys = ys[::-1]

    data = {}
    for name, stack in arrays.items():
        a = np.stack(stack).astype(dtype)
        if flip:
            a = a[:, ::-1, :]
        var = "TOT_PREC" if name == "RATE" else name
        data[var] = (("time", "y", "x"), a)

    ds = xr.Dataset(data, coords={"time": np.array(stamps, dtype="datetime64[ns]"),
                                  "y": ys.astype("float64"), "x": xs.astype("float64")})
    # The existing stores carry no group-level attributes; keep it that way.
    ds["TOT_PREC"].attrs.update({
        "long_name": "total precipitation at the surface (liquid water equivalent)",
        "standard_name": "lwe_precipitation_rate",
        "units": "mm h-1",
        "spatial_resolution": "2000 m",
        "temporal_resolution": "15 min",
        "source": "EUMETNET OPERA composite, openradar-archive, CC-BY-4.0",
    })
    if "QIND" in ds:
        ds["QIND"].attrs.update({
            "long_name": "OPERA total quality index",
            "units": "1",
            "valid_range": "0..1",
            "source": "pl.imgw.quality.qi_total",
        })

    from numcodecs import Blosc
    comp = Blosc(cname=STORE_CNAME, clevel=STORE_CLEVEL,
                 shuffle=Blosc.SHUFFLE) if compress else None
    enc = {v: {"compressor": comp, "chunks": (1, ny, nx), "dtype": dtype,
               "_FillValue": np.nan} for v in data}
    for c in ("x", "y"):
        enc[c] = {"compressor": comp, "dtype": "float64"}
    # int64 minutes since midnight, as the existing stores encode it
    enc["time"] = {"compressor": comp, "dtype": "int64",
                   "units": f"minutes since {day:%Y-%m-%d} 00:00:00",
                   "calendar": "proleptic_gregorian"}

    dest = os.path.join(out_dir, f"{day:%Y%m%d}")
    if os.path.exists(dest):
        if not overwrite:
            raise FileExistsError(
                f"{dest} already exists; pass --overwrite to replace it, or "
                f"--skip_existing to leave it alone")
        shutil.rmtree(dest)
    ds.to_zarr(dest, mode="w", encoding=enc, consolidated=True)
    return dest


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Fetch OPERA composites from the public archive onto the patch grid.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--bucket", default="openradar-archive",
                    help="openradar-archive (2012-) or openradar-24h (last 24 h)")
    ap.add_argument("--product", default="auto",
                    help="'auto' (default) resolves the product per day from "
                         "--product_priority, which is what lets one run span the 2024/2025 "
                         "rename. Give an explicit name (QIND_RATE, RATE, DBZH_QIND, "
                         "ACRR_QIND) to pin it. --list shows what a day holds")
    ap.add_argument("--product_priority", nargs="+", default=["QIND_RATE", "RATE"],
                    help="preference order used when --product auto")
    ap.add_argument("--every", type=int, default=0,
                    help="keep only timestamps on this minute cadence (e.g. 30); 0 = all")
    ap.add_argument("--out", default=None, help="output directory of per-day zarr stores")
    ap.add_argument("--reference", default=None,
                    help="an existing raw zarr day, or the DEM GeoTIFF, defining the target "
                         "grid. Strongly recommended: it is what keeps new days aligned "
                         "with the existing patch metadata")
    ap.add_argument("--resampling", default="nearest",
                    choices=["nearest", "bilinear", "average"],
                    help="only used when a composite is not already on the reference grid; "
                         "nearest preserves precipitation peaks, bilinear smooths them")
    ap.add_argument("--cache", default=None,
                    help="directory for the downloaded .h5 files (default: a temporary "
                         "directory, removed afterwards)")
    ap.add_argument("--keep_h5", action="store_true", help="do not delete the raw downloads")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--skip_existing", action="store_true",
                    help="leave days whose zarr store is already present. Use this when "
                         "writing into a directory that already holds data")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing day store. Without it a collision is an "
                         "error, so a run cannot quietly destroy days already fetched")
    ap.add_argument("--list", dest="do_list", action="store_true",
                    help="report what the archive holds over the range and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="plan and report, download nothing")
    args = ap.parse_args()

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        sys.exit("--end precedes --start")

    product = None if args.product == "auto" else args.product
    print(f"[archive] {S3_HOST}/{args.bucket}  {start} .. {end}  "
          + (f"product=auto({'/'.join(args.product_priority)})" if product is None
             else f"product={product}")
          + (f"  every {args.every} min" if args.every else ""))

    # ---- listing ------------------------------------------------------------------
    if args.do_list:
        grand, total_bytes = 0, 0
        for day in daterange(start, end):
            try:
                names = products_seen(args.bucket, day)
            except Exception as e:
                print(f"  {day}  [listing failed: {e}]")
                continue
            if not names:
                print(f"  {day}  -")
                continue
            want = (args.product if args.product != "auto"
                    else resolve_product(set(names), args.product_priority))
            got = names.get(want, 0)
            grand += got
            summary = ", ".join(f"{k}x{v}" for k, v in sorted(names.items()))
            print(f"  {day}  {summary}")
        print(f"\n{grand} files over {(end - start).days + 1} day(s) "
              f"(product {args.product})")
        print("At roughly 1 MB per composite that is about "
              f"{grand / 1024:.1f} GB; 96 composites a day is a full 15-minute cadence.")
        return

    _require_h5py()
    if not args.out:
        sys.exit("--out is required unless --list is given")
    os.makedirs(args.out, exist_ok=True)

    ref = reference_grid(args.reference)
    print(f"[grid] reference {ref[2][0]}x{ref[2][1]}"
          + (f" from {args.reference}" if args.reference else " from built-in constants"))
    if not args.reference:
        print("       consider --reference <an existing raw zarr day> so the new stores "
              "are guaranteed to align with the existing patch metadata")

    print("[plan] listing the archive ...")
    per_day, chosen = plan(args.bucket, start, end, product, args.every,
                           args.product_priority)
    picks = sorted({p for p in chosen.values() if p})
    if product is None and picks:
        print(f"[plan] products resolved: {', '.join(picks)}")
    n_files = sum(len(v) for v in per_day.values())
    n_bytes = sum(s for v in per_day.values() for _, _, s in v)
    print(f"[plan] {n_files} files, {n_bytes / 1e9:.2f} GB, over "
          f"{sum(1 for v in per_day.values() if v)} day(s) with data")
    if n_files == 0:
        for day in list(per_day)[:1]:
            try:
                print(f"       products present on {day}: {products_seen(args.bucket, day)}")
            except Exception:
                pass
        sys.exit("nothing to fetch: check --product and the date range")
    if args.dry_run:
        for day, rows in per_day.items():
            if rows:
                print(f"    {day}: {len(rows):>3} files, "
                      f"{rows[0][0]:%H:%M} .. {rows[-1][0]:%H:%M}")
        print("[dry run] nothing downloaded")
        return

    cache = args.cache or tempfile.mkdtemp(prefix="opera_archive_")
    os.makedirs(cache, exist_ok=True)
    made = []
    try:
        for day, rows in per_day.items():
            if not rows:
                continue
            dest = os.path.join(args.out, f"{day:%Y%m%d}")
            if args.skip_existing and store_complete(dest):
                print(f"  [skip] {day} (store exists)")
                continue
            if os.path.exists(dest):
                # A store left behind by an interrupted run: zarr writes consolidated
                # metadata last, so its absence marks a day that never finished. Without
                # this, --skip_existing would skip the wreckage forever on every resume.
                print(f"  [redo] {day}: previous store is incomplete, rewriting")
                shutil.rmtree(dest)

            print(f"  [get ] {day}: {len(rows)} files", flush=True)
            paths = {}
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(download, args.bucket, key,
                                  os.path.join(cache, os.path.basename(key))): stamp
                        for stamp, key, _ in rows}
                for fut in cf.as_completed(futs):
                    stamp = futs[fut]
                    try:
                        paths[stamp] = fut.result()
                    except Exception as e:
                        print(f"    ! {stamp:%H:%M} download failed: {e}")

            stamps, arrays, warned = [], {}, False
            for stamp in sorted(paths):
                try:
                    fields, grid = read_odim(paths[stamp])
                except Exception as e:
                    print(f"    ! {stamp:%H:%M} unreadable: {e}")
                    continue
                passthrough = same_grid(grid, ref)
                if not passthrough and not warned:
                    print(f"    [warp] source grid {grid[1]} differs from the reference; "
                          f"reprojecting with {args.resampling}")
                    warned = True
                stamps.append(np.datetime64(stamp))
                for name, arr in fields.items():
                    if name not in ("RATE", "QIND"):
                        continue
                    out = arr if passthrough else reproject_to_reference(
                        arr, grid, ref, args.resampling)
                    arrays.setdefault(name, []).append(out)

            if not stamps:
                print(f"    ! {day}: nothing readable, no store written")
                continue
            missing = [k for k in ("RATE", "QIND") if k not in arrays]
            if missing:
                print(f"    [note] {day}: {', '.join(missing)} absent from this product")
            if any(len(v) != len(stamps) for v in arrays.values()):
                print(f"    ! {day}: field/timestamp count mismatch, skipping")
                continue

            made.append(write_day(day, stamps, arrays, ref, args.out,
                                  overwrite=args.overwrite))
            print(f"  [wrote] {made[-1]}  ({len(stamps)} times)")

            if not args.keep_h5:
                for p in paths.values():
                    if os.path.exists(p):
                        os.unlink(p)
    finally:
        if not args.cache and not args.keep_h5 and os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)

    print(f"\n[done] {len(made)} day store(s) under {args.out}")
    if made:
        print("Next: extend the timestamp map and patch metadata so preprocessing sees "
              "these days, then rebuild the patch set.")


if __name__ == "__main__":
    main()
