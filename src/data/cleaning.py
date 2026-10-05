"""Cleaning of OPERA rain-rate frames for the v2 datasets (clear errors only).

Policy: DECISIONS §17 (reject clear errors, keep imperfections) and the provisional
> 150 mm/h rule recorded in EXPERIMENTS §5. The same functions are used by the tile scan
(`scripts/dataset_v2/scan_tiles.py`) and by the store build, so what is selected and what is
stored cannot drift apart.

Frame level, pixel repairs (point-like clear errors). Nothing is ever set to zero to hide a
value; a repaired pixel takes the value its neighbourhood supports:

  drizzle        < 0.1 mm/h -> 0 (sensor noise floor, unchanged from v1)
  static clutter pixels that reach >= 31 mm/h in > 1% of the valid steps of their year
                 (~100x climatology; from `clutter_climatology.py`) take the median of the
                 non-clutter valid pixels of their 5x5 neighbourhood, if that is lower
                 (as EURADCLIM does with interpolation from neighbours)
  spikes         a pixel >= 10 mm/h whose brightest 8-neighbour is below 10% of it has no
                 spatial support at 2 km, and is lowered to that neighbour value

Tile level, flags (extended clear errors; the tile is rejected, not repaired):

  unphysical     cleaned max > 500 mm/h (~67 dBZ with Marshall-Palmer)
  ray            a thin straight component >= 80 km long pointing at an OPERA radar within
                 250 km (RLAN / emitter interference). Alignment with the radar is what
                 separates a ray from a squall line, which is also long and thin.

Values between 150 and 500 mm/h are kept as measured: no declutter zeroing in v2.

Audit-only checks (2026-10-02; computed by `scripts/dataset_v2/scan_flags.py`, not yet
applied by the scan or the store build, pending validation against gauges):

  temporal support  a cell >= 10 mm/h with no echo >= 1 mm/h within 30 km in either
                    neighbouring frame (t +- 15 min) has no precursor and no successor
  ring              a thin arc centred on an OPERA radar within 250 km (range ring), told
                    apart from a straight band by fitting both a circle about the radar and
                    a line

Repairs (v3, 2026-10-03; `repair_static` + `repair_unsupported`, decided from the gauge
validation in `notes/data_quality_assessment`). Each lowers a clear-error footprint to the
median of the valid pixels on its outer 1-px ring, so the rain around it stays and the tile
is kept instead of rejected:

  footprint    every connected region >= 150 mm/h that contains a pixel > 500 mm/h (the
               pixels around a > 500 core are artefact too: 10% gauge-corroborated at
               150-500 mm/h, against 74% at 31-89 mm/h elsewhere in the same tiles)
  ray          the ray components themselves (the `ray_flag` criterion, per stride-128
               tile), instead of rejecting the tile (73% of the other pixels were rain)
  ring         pixels >= 89 mm/h on a climatological range ring of that year
               (`ring_climatology.py`; dry under every gauge at >= 89 mm/h, but real
               rain under 38% of ring pixels at 31-89, so those are kept)
  unsupported  cells with no temporal support (gauge median and 90th percentile 0 mm/h)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DRIZZLE = 0.1
HOT_U = 31.0
HOT_FREQ = 0.01
SPIKE_MIN = 10.0
SPIKE_RATIO = 0.10
UNPHYSICAL = 500.0
RAY_MIN_LEN_PX = 40          # 80 km
RAY_MAX_ANGLE_DEG = 10.0
RAY_MAX_DIST_KM = 250.0
SUPPORT_U = 10.0             # cells tested for temporal support (mm/h)
SUPPORT_MIN = 1.0            # echo that counts as support in a neighbour frame (mm/h)
SUPPORT_RADIUS_PX = 15       # 30 km: 15 min at 33 m/s
RING_MIN_ARC_PX = 40         # 80 km of arc
RING_MAX_RADIAL_STD_PX = 1.5 # thin in range
RING_CURVATURE_RATIO = 0.8   # circle residual must beat the straight-line residual by 20%
REPAIR_RING_MIN = 89.0       # ring pixels repaired from this rate (mm/h); at 31-89 the gauges
                             # saw >= 10 mm/h under 38% of ring pixels (2026-10-04)
FOOTPRINT_MIN = 150.0        # region around a > UNPHYSICAL core that is repaired (mm/h)
REPAIR_FOOTPRINT, REPAIR_RAY, REPAIR_RING, REPAIR_UNSUPPORTED = 1, 2, 4, 8   # bit codes

_RING = np.ones((3, 3), dtype=bool)
_RING[1, 1] = False


def hot_mask_from_climatology(clim, year: int, min_valid: int = 200,
                              u: float = HOT_U, freq: float = HOT_FREQ):
    """Static-clutter mask for one year from a `clutter_climatology.npz` mapping.

    Falls back to the whole-period counts when the year is absent. Returns None if neither
    is available.
    """
    key = f"y{year}_"
    tag = f"n_ge{u:g}"
    if key + "n_valid" in clim and key + tag in clim:
        nv, n = clim[key + "n_valid"].astype(float), clim[key + tag].astype(float)
    elif "n_valid" in clim and tag in clim:
        nv, n = clim["n_valid"].astype(float), clim[tag].astype(float)
    else:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(nv >= min_valid, n / np.maximum(nv, 1), 0.0)
    return f > freq


def clean_frame(frame: np.ndarray, hot: np.ndarray | None = None):
    """Repair one full frame. Returns (cleaned, info).

    `frame`: rain rate (mm/h), NaN where there is no radar coverage (kept NaN).
    `hot`: boolean static-clutter mask of the same shape, or None.
    """
    valid = np.isfinite(frame)
    z = np.where(valid, frame, 0.0).astype(np.float32)
    z[z < DRIZZLE] = 0.0
    info = {"n_hot_fixed": 0, "n_spikes_fixed": 0}

    if hot is not None:
        cand = hot & valid & (z > 0)
        if cand.any():
            src = np.where(hot | ~valid, np.nan, z)
            H, W = z.shape
            for y, x in zip(*np.nonzero(cand)):
                win = src[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
                fin = win[np.isfinite(win)]
                rep = float(np.median(fin)) if fin.size else 0.0
                if rep < z[y, x]:
                    z[y, x] = rep
                    info["n_hot_fixed"] += 1

    nbmax = ndimage.maximum_filter(z, footprint=_RING, mode="constant", cval=0.0)
    spike = (z >= SPIKE_MIN) & (nbmax < SPIKE_RATIO * z)
    if spike.any():
        z[spike] = nbmax[spike]
        info["n_spikes_fixed"] = int(spike.sum())

    return np.where(valid, z, np.nan).astype(np.float32), info


def _components(z, u=1.0, min_size=30):
    """Label thin-candidate components; yield (centroid_y, centroid_x, length_px, angle_rad)."""
    lab, n = ndimage.label(z >= u, structure=np.ones((3, 3), bool))
    if not n:
        return
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(z), lab, idx)
    keep = size >= min_size
    if not keep.any():
        return
    yy, xx = np.indices(z.shape, dtype=float)
    idx, size = idx[keep], size[keep]
    my = ndimage.sum(yy, lab, idx) / size
    mx = ndimage.sum(xx, lab, idx) / size
    cyy = ndimage.sum(yy * yy, lab, idx) / size - my ** 2
    cxx = ndimage.sum(xx * xx, lab, idx) / size - mx ** 2
    cxy = ndimage.sum(xx * yy, lab, idx) / size - mx * my
    tr, det = cyy + cxx, cyy * cxx - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    elong = 1 - np.sqrt(lmin / np.maximum(lmax, 1e-12))
    length = np.sqrt(12 * lmax)
    # principal-axis direction (in the (x, y) index plane)
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    for k in range(len(idx)):
        if elong[k] > 0.9 and length[k] >= RAY_MIN_LEN_PX:
            yield my[k], mx[k], length[k], ang[k]


def ray_flag(tile: np.ndarray, row0: int, col0: int, sites_rc: np.ndarray) -> bool:
    """True if the tile holds a thin straight component that points at a nearby radar.

    `sites_rc`: (N, 2) array of radar (row, col) on the grid (rows count from the south).
    """
    if sites_rc is None or not len(sites_rc):
        return False
    z = np.nan_to_num(tile, nan=0.0)
    max_px = RAY_MAX_DIST_KM / 2.0
    for cy, cx, _, ang in _components(z):
        gy, gx = row0 + cy, col0 + cx
        d = sites_rc - np.array([gy, gx])
        dist = np.hypot(d[:, 0], d[:, 1])
        near = dist <= max_px
        if not near.any():
            continue
        # angle between the component axis and the direction to each nearby radar
        dirs = np.arctan2(d[near, 0], d[near, 1])          # (dy, dx) -> angle of (x, y)
        diff = np.abs(((dirs - ang) + np.pi / 2) % np.pi - np.pi / 2)
        if np.degrees(diff.min()) <= RAY_MAX_ANGLE_DEG:
            return True
    return False


def tile_stats(t: np.ndarray, factor: float = 12.5) -> dict:
    """Intensity summary of one cleaned tile (no NaN expected: tiles are fully covered)."""
    from src.data.quality import adaptive_block_means
    z = np.nan_to_num(t, nan=0.0)
    out = {"max": float(z.max()), "mean": float(z.mean()),
           "wet_frac": float((z >= DRIZZLE).mean())}
    for u in (1.0, 10.0, 31.0, 53.0, 89.0, 150.0):
        out[f"n_ge{u:g}"] = int((z >= u).sum())
    out["coarse_max"] = float(adaptive_block_means(z, int(z.shape[0] / factor)).max())
    return out


def support_maps(frame: np.ndarray, u_support: float = SUPPORT_MIN,
                 radius: int = SUPPORT_RADIUS_PX):
    """(support, coverage) maps of one neighbour frame for `temporal_support`: echo
    >= u_support anywhere within `radius` px (square window), and full coverage of that
    window. Computing them once per frame lets a scan reuse them for t - 15 and t + 15."""
    size = 2 * radius + 1
    sup = ndimage.maximum_filter((np.nan_to_num(frame, nan=0.0) >= u_support).astype(np.uint8), size=size)
    cov = ndimage.minimum_filter(np.isfinite(frame).astype(np.uint8), size=size)
    return sup, cov


def temporal_support(cur: np.ndarray, prev, nxt, u: float = SUPPORT_U,
                     u_support: float = SUPPORT_MIN, radius: int = SUPPORT_RADIUS_PX) -> np.ndarray:
    """Cells of `cur` with neither a precursor nor a successor.

    `cur`: full cleaned frame (mm/h, NaN = no coverage) at t. `prev` / `nxt`: the frames at
    t - 15 min and t + 15 min (same kind of array), or their `support_maps` tuples, or None
    when that frame does not exist.
    Returns int8, same shape: 1 = pixel of a cell (>= u, 8-connected) with no echo >= u_support
    within `radius` px (square window) in either neighbour; 0 = supported, or below u;
    -1 = undecidable (a neighbour missing, or not fully covered around the cell).
    """
    z = np.nan_to_num(cur, nan=0.0)
    lab, n = ndimage.label(z >= u, structure=np.ones((3, 3), bool))
    out = np.zeros(z.shape, np.int8)
    if not n:
        return out
    if prev is None or nxt is None:
        out[lab > 0] = -1
        return out
    m = lab > 0
    lm = lab[m]
    sup, cov = [], []
    for f in (prev, nxt):
        s_, c_ = f if isinstance(f, tuple) else support_maps(f, u_support, radius)
        sup.append(np.bincount(lm, weights=s_[m], minlength=n + 1)[1:] > 0)
        cov.append(np.bincount(lm, weights=1 - c_[m].astype(np.int8), minlength=n + 1)[1:] == 0)
    decided = cov[0] & cov[1]
    unsupported = decided & ~sup[0] & ~sup[1]
    code = np.zeros(n + 1, np.int8)
    code[1:][unsupported] = 1
    code[1:][~decided] = -1
    return code[lab]


def ring_flag(tile: np.ndarray, row0: int, col0: int, sites_rc: np.ndarray,
              u: float = 1.0, min_size: int = 30) -> bool:
    """True if the tile holds a thin arc centred on a radar within RAY_MAX_DIST_KM.

    A component is a ring about radar s if, in polar coordinates about s, it spans at least
    RING_MIN_ARC_PX of arc, its range spread is at most RING_MAX_RADIAL_STD_PX, and that
    spread is clearly smaller than its spread about its own best straight line. The last
    condition keeps a straight band that happens to lie tangential to a radar.
    """
    if sites_rc is None or not len(sites_rc):
        return False
    z = np.nan_to_num(tile, nan=0.0)
    lab, n = ndimage.label(z >= u, structure=np.ones((3, 3), bool))
    if not n:
        return False
    max_px = RAY_MAX_DIST_KM / 2.0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        if ys.size < min_size:
            continue
        ys = ys + sl[0].start + row0
        xs = xs + sl[1].start + col0
        cy, cx = ys.mean(), xs.mean()
        c = np.cov(np.vstack([xs, ys]))
        line_rms = float(np.sqrt(max(np.linalg.eigvalsh(c)[0], 0.0)))
        d = np.hypot(sites_rc[:, 0] - cy, sites_rc[:, 1] - cx)
        for sy, sx in sites_rc[d <= max_px]:
            r = np.hypot(ys - sy, xs - sx)
            if r.min() < 5:                            # the radar sits inside the component
                continue
            r_std = float(r.std())
            if r_std > RING_MAX_RADIAL_STD_PX or r_std > RING_CURVATURE_RATIO * line_rms:
                continue
            th = np.sort(np.arctan2(ys - sy, xs - sx))
            gaps = np.diff(np.r_[th, th[0] + 2 * np.pi])
            span = 2 * np.pi - gaps.max()
            if span * r.mean() >= RING_MIN_ARC_PX:
                return True
    return False


_EIGHT = np.ones((3, 3), bool)


def boundary_fill(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lower each 8-connected component of `mask` to the median of the valid pixels on its
    outer 1-px ring (pixels already lower keep their value). In place; returns the boolean
    map of pixels that changed. NaN (no coverage) stays NaN and never enters a median.
    Works per component on its bounding box (components are few and small)."""
    mask = mask & np.isfinite(z)
    changed = np.zeros(z.shape, bool)
    if not mask.any():
        return changed
    lab, n = ndimage.label(mask, structure=_EIGHT)
    H, W = z.shape
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys = slice(max(sl[0].start - 1, 0), min(sl[0].stop + 1, H))
        xs = slice(max(sl[1].start - 1, 0), min(sl[1].stop + 1, W))
        comp = lab[ys, xs] == k
        zs = z[ys, xs]
        ring = ndimage.binary_dilation(comp, structure=_EIGHT) & ~mask[ys, xs] & np.isfinite(zs)
        target = float(np.median(zs[ring])) if ring.any() else 0.0
        low = comp & (zs > target)
        zs[low] = target
        changed[ys, xs] |= low
    return changed


def _ray_components_mask(tile: np.ndarray, row0: int, col0: int, sites_rc: np.ndarray) -> np.ndarray:
    """Pixels of the components `ray_flag` would fire on (same criterion: >= 1 mm/h,
    >= 30 px, elongation > 0.9, length >= RAY_MIN_LEN_PX, axis within RAY_MAX_ANGLE_DEG of
    the direction to a radar within RAY_MAX_DIST_KM), as a mask."""
    out = np.zeros(tile.shape, bool)
    if sites_rc is None or not len(sites_rc):
        return out
    z = np.nan_to_num(tile, nan=0.0)
    lab, n = ndimage.label(z >= 1.0, structure=_EIGHT)
    max_px = RAY_MAX_DIST_KM / 2.0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == k
        yy, xx = np.nonzero(comp)
        if yy.size < 30:
            continue
        yy = yy + sl[0].start
        xx = xx + sl[1].start
        my, mx = yy.mean(), xx.mean()
        cyy, cxx = (yy * yy).mean() - my ** 2, (xx * xx).mean() - mx ** 2
        cxy = (xx * yy).mean() - mx * my
        tr, det = cyy + cxx, cyy * cxx - cxy ** 2
        disc = np.sqrt(max(tr ** 2 / 4 - det, 0.0))
        lmax, lmin = tr / 2 + disc, max(tr / 2 - disc, 0.0)
        if not (1 - np.sqrt(lmin / max(lmax, 1e-12)) > 0.9 and np.sqrt(12 * lmax) >= RAY_MIN_LEN_PX):
            continue
        ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        d = sites_rc - np.array([row0 + my, col0 + mx])
        near = np.hypot(d[:, 0], d[:, 1]) <= max_px
        if not near.any():
            continue
        dirs = np.arctan2(d[near, 0], d[near, 1])
        diff = np.abs(((dirs - ang) + np.pi / 2) % np.pi - np.pi / 2)
        if np.degrees(diff.min()) <= RAY_MAX_ANGLE_DEG:
            out[sl] |= comp
    return out


def repair_static(z: np.ndarray, ring_mask: np.ndarray | None = None,
                  sites_rc: np.ndarray | None = None, row0: int = 0, col0: int = 0,
                  patch: int = 128):
    """Footprint, ray and ring repairs of one `clean_frame` output (or a crop of it whose
    south-west pixel is (row0, col0) on the full grid). Returns (repaired copy, code map),
    code = OR of the REPAIR_* bits of the repairs that lowered each pixel. Rays are searched
    per stride-`patch` tile of the FULL grid that lies inside the array, as in the scan."""
    z = z.copy()
    code = np.zeros(z.shape, np.uint8)
    # 1. footprints of > UNPHYSICAL cores
    core = np.nan_to_num(z, nan=0.0) > UNPHYSICAL
    if core.any():
        lab, n = ndimage.label(np.nan_to_num(z, nan=0.0) >= FOOTPRINT_MIN, structure=_EIGHT)
        hit = np.unique(lab[core])
        fp = np.isin(lab, hit[hit > 0])
        code[boundary_fill(z, fp)] |= REPAIR_FOOTPRINT
    # 2. rays, per tile of the global stride grid
    if sites_rc is not None and len(sites_rc):
        H, W = z.shape
        rmask = np.zeros(z.shape, bool)
        z0 = np.nan_to_num(z, nan=0.0)
        for g0 in range(-(-row0 // patch) * patch, row0 + H - patch + 1, patch):
            for h0 in range(-(-col0 // patch) * patch, col0 + W - patch + 1, patch):
                y, x = g0 - row0, h0 - col0
                t = z0[y:y + patch, x:x + patch]
                if (t >= 1.0).sum() < 30:
                    continue
                rmask[y:y + patch, x:x + patch] |= _ray_components_mask(t, g0, h0, sites_rc)
        if rmask.any():
            code[boundary_fill(z, rmask)] |= REPAIR_RAY
    # 3. climatological range rings
    if ring_mask is not None:
        rm = ring_mask & (np.nan_to_num(z, nan=0.0) >= REPAIR_RING_MIN)
        if rm.any():
            code[boundary_fill(z, rm)] |= REPAIR_RING
    return z, code


def repair_unsupported(z: np.ndarray, prev, nxt, code: np.ndarray | None = None):
    """Lower cells with no temporal support (`temporal_support` == 1). `prev` / `nxt` as in
    `temporal_support` (frames, `support_maps` tuples or None). Returns (repaired copy, code)."""
    z = z.copy()
    code = np.zeros(z.shape, np.uint8) if code is None else code.copy()
    un = temporal_support(z, prev, nxt) == 1
    if un.any():
        code[boundary_fill(z, un)] |= REPAIR_UNSUPPORTED
    return z, code
