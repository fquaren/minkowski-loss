"""
Per-tile data-quality features for OPERA radar precipitation.

Each function takes one 128x128 tile of the *raw* rain-rate field (mm/h, NaN outside radar
coverage, no drizzle/declutter filtering applied) and returns plain floats, so a tile can be
described by one flat row of a table. The features are candidate artefact signatures, not a
screen: which of them separates artefacts from storms is what the data-quality scripts are
meant to find out (see `scripts/data_quality/README.md`).

Signatures targeted, and the feature that reads each one:

    isolated speckle / single-pixel clutter   isolated_frac_*, n_spikes, peak_nbr_ratio
    unphysical spatial gradients              max_abs_grad
    concentration of mass in a few pixels     conc, peak_block_ratio
    sea clutter                               argmax_over_sea, wet_over_sea_frac
    spokes / rays (RLAN interference)         spoke_len, max_elong
    single-frame flicker                      persist_prev, persist_next, corr_prev
    what our declutter step would delete      n_over_declutter, n_over_declutter_in_cell
    OPERA's own quality flag                  q_* (only when a QIND field is present)

Static, recurring clutter (the same pixel hot in many time steps) cannot be seen from one
tile; `scripts/data_quality/clutter_climatology.py` builds the per-pixel frequency map and
`clim_*` features are joined from it.
"""

import warnings

import numpy as np
from scipy import ndimage


DRIZZLE = 0.1
DECLUTTER = 150.0
COUNT_LEVELS = (1.0, 31.0, 53.0, 89.0, 150.0, 500.0)
EIGHT = np.ones((3, 3), dtype=bool)


def _neighbour_stack(r: np.ndarray) -> np.ndarray:
    """The 8 neighbours of every pixel, shape (8, H, W); NaN past the tile edge."""
    p = np.pad(r, 1, mode="constant", constant_values=np.nan)
    H, W = r.shape
    views = [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    return np.stack(views)


def adaptive_block_means(a: np.ndarray, out: int) -> np.ndarray:
    """Block means with the same bin edges as `torch.nn.functional.adaptive_avg_pool2d`.

    Reproduces the coarse input the model actually sees (128 / 12.5 -> 10x10, overlapping
    bins), without importing torch in worker processes.
    """
    H, W = a.shape
    ys = [(int(np.floor(i * H / out)), int(np.ceil((i + 1) * H / out))) for i in range(out)]
    xs = [(int(np.floor(j * W / out)), int(np.ceil((j + 1) * W / out))) for j in range(out)]
    return np.array([[a[y0:y1, x0:x1].mean() for x0, x1 in xs] for y0, y1 in ys])


def _block_of(idx: int, n: int, out: int) -> int:
    """Index of the first adaptive-pool bin containing pixel `idx`."""
    for i in range(out):
        if int(np.floor(i * n / out)) <= idx < int(np.ceil((i + 1) * n / out)):
            return i
    return out - 1


def pipeline_filter(r: np.ndarray, drizzle=DRIZZLE, declutter=DECLUTTER) -> np.ndarray:
    """What `preprocessing.filter_precip_bounds` does today: NaN and out-of-bounds -> 0."""
    f = np.nan_to_num(r, nan=0.0).copy()
    f[(f < drizzle) | (f > declutter)] = 0.0
    return f


def intensity_features(r: np.ndarray) -> dict:
    fin = np.isfinite(r)
    out = {"coverage": float(fin.mean())}
    if not fin.any():
        return out
    v = r[fin]
    rmax = float(v.max())
    out.update(
        max=rmax,
        mean=float(v.mean()),
        p99=float(np.percentile(v, 99)),
        wet_frac=float((v >= DRIZZLE).mean()),
        conc=float(v.mean() / rmax) if rmax > 0 else np.nan,
    )
    for u in COUNT_LEVELS:
        out[f"n_ge{u:g}"] = int((v >= u).sum())
    return out


def spatial_features(r: np.ndarray) -> dict:
    """Speckle, spikes, gradients and component structure."""
    z = np.nan_to_num(r, nan=0.0)
    nb = _neighbour_stack(np.where(np.isfinite(r), r, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nb_max = np.nanmax(nb, axis=0)
        nb_mean = np.nanmean(nb, axis=0)
        nb_med = np.nanmedian(nb, axis=0)
        nb_min = np.nanmin(nb, axis=0)
    nb_max = np.nan_to_num(nb_max, nan=0.0)
    nb_mean = np.nan_to_num(nb_mean, nan=0.0)
    nb_med = np.nan_to_num(nb_med, nan=0.0)
    nb_min = np.nan_to_num(nb_min, nan=0.0)

    out = {}
    for u in (DRIZZLE, 1.0, 10.0):
        wet = z >= u
        n_wet = int(wet.sum())
        isolated = wet & (nb_max < u)
        out[f"isolated_frac_ge{u:g}"] = float(isolated.sum() / n_wet) if n_wet else np.nan

    # A spike: an intense pixel whose brightest neighbour is under a quarter of it.
    spikes = (z >= 10.0) & (nb_max < 0.25 * z)
    out["n_spikes"] = int(spikes.sum())
    tot = float(z.sum())
    out["spike_mass_frac"] = float(z[spikes].sum() / tot) if tot > 0 else 0.0

    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    out["argmax_y"], out["argmax_x"] = int(iy), int(ix)
    out["peak_nbr_ratio"] = float(z[iy, ix] / (nb_mean[iy, ix] + DRIZZLE))

    with np.errstate(invalid="ignore"):
        gy = np.abs(np.diff(r, axis=0))
        gx = np.abs(np.diff(r, axis=1))
    g = np.concatenate([gy[np.isfinite(gy)], gx[np.isfinite(gx)]])
    out["max_abs_grad"] = float(g.max()) if g.size else np.nan

    # Declutter: what the current pipeline zeroes, split by whether it sits inside a cell.
    over = z > DECLUTTER
    out["n_over_declutter"] = int(over.sum())
    out["n_over_declutter_in_cell"] = int((over & (nb_med >= 10.0)).sum())
    # Holes already present in the raw field: dry pixel ringed by heavy rain.
    out["n_holes"] = int(((z < DRIZZLE) & (nb_min >= 10.0)).sum())

    for u in (1.0, 31.0):
        lab, n = ndimage.label(z >= u, structure=EIGHT)
        out[f"n_comp_ge{u:g}"] = int(n)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            out[f"largest_comp_ge{u:g}"] = int(sizes.max())
        else:
            out[f"largest_comp_ge{u:g}"] = 0

    out.update(_elongation(z))
    return out


def _elongation(z: np.ndarray, u: float = 1.0, min_size: int = 30) -> dict:
    """Longest thin, straight component: the spoke/ray signature.

    Second moments of each component's pixel coordinates give its principal axes;
    elongation = 1 - sqrt(lambda_min / lambda_max) is 0 for a disc and -> 1 for a line, and
    sqrt(12 lambda_max) is the length of a uniform segment with that variance.
    """
    lab, n = ndimage.label(z >= u, structure=EIGHT)
    res = {"max_elong": 0.0, "spoke_len": 0.0}
    if not n:
        return res
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(z), lab, idx)
    keep = size >= min_size
    if not keep.any():
        return res
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
    res["max_elong"] = float(elong.max())
    thin = elong > 0.9
    res["spoke_len"] = float(length[thin].max()) if thin.any() else 0.0
    return res


def coarse_features(r: np.ndarray, factor: float = 12.5) -> dict:
    """How the peak relates to the coarse input the model is conditioned on."""
    f = pipeline_filter(r)
    out_n = int(r.shape[0] / factor)
    coarse = adaptive_block_means(f, out_n)
    z = np.nan_to_num(r, nan=0.0)
    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    blk = coarse[_block_of(iy, r.shape[0], out_n), _block_of(ix, r.shape[1], out_n)]
    return {
        "coarse_max": float(coarse.max()),
        "peak_block_mean": float(blk),
        # a lone spike in a ~13x13 block reads ~160; a convective core reads single digits
        "peak_block_ratio": float(z[iy, ix] / (blk + DRIZZLE)),
        "filtered_max": float(f.max()),
    }


def dem_features(r: np.ndarray, dem: np.ndarray) -> dict:
    z = np.nan_to_num(r, nan=0.0)
    sea = ~np.isfinite(dem) | (dem <= 0)
    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    wet = z >= 1.0
    return {
        "sea_frac": float(sea.mean()),
        "dem_mean": float(np.nanmean(dem)) if np.isfinite(dem).any() else np.nan,
        "argmax_over_sea": int(sea[iy, ix]),
        "wet_over_sea_frac": float((wet & sea).sum() / wet.sum()) if wet.any() else np.nan,
    }


def qind_features(r: np.ndarray, q: np.ndarray) -> dict:
    """Summary of OPERA's quality index (in [0,1]) over the tile."""
    fr = np.isfinite(r)
    fq = np.isfinite(q) & fr
    out = {"q_cov": float(fq.sum() / fr.sum()) if fr.any() else np.nan}
    if not fq.any():
        return out
    z = np.nan_to_num(r, nan=0.0)
    out["q_mean"] = float(q[fq].mean())
    for name, sel in (("wet", z >= 1.0), ("ge31", z >= 31.0)):
        m = sel & fq
        out[f"q_mean_{name}"] = float(q[m].mean()) if m.any() else np.nan
    m = (z >= 31.0) & fq
    out["q_frac_low_ge31"] = float((q[m] < 0.5).mean()) if m.any() else np.nan
    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    out["q_at_max"] = float(q[iy, ix]) if np.isfinite(q[iy, ix]) else np.nan
    return out


def temporal_features(r: np.ndarray, prev=None, nxt=None, radius: int = 6) -> dict:
    """Does the peak persist into the neighbouring 15-minute frames?

    The window around the argmax (+-6 px = +-12 km) allows ~50 km/h advection between
    frames. A storm keeps a comparable maximum nearby; a one-frame artefact does not.
    """
    z = np.nan_to_num(r, nan=0.0)
    rmax = float(z.max())
    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    sl = (slice(max(iy - radius, 0), iy + radius + 1),
          slice(max(ix - radius, 0), ix + radius + 1))
    out = {}
    for name, other in (("prev", prev), ("next", nxt)):
        if other is None or not np.isfinite(other).any():
            out[f"persist_{name}"] = np.nan
            continue
        loc = np.nan_to_num(other[sl], nan=0.0).max()
        out[f"persist_{name}"] = float(loc / rmax) if rmax > 0 else np.nan
    if prev is not None:
        m = np.isfinite(r) & np.isfinite(prev)
        a, b = np.log1p(np.maximum(r[m], 0)), np.log1p(np.maximum(prev[m], 0))
        out["corr_prev"] = (float(np.corrcoef(a, b)[0, 1])
                            if m.sum() > 10 and a.std() > 0 and b.std() > 0 else np.nan)
    else:
        out["corr_prev"] = np.nan
    return out


def tile_features(r, dem=None, q=None, prev=None, nxt=None, factor: float = 12.5) -> dict:
    """All features for one raw tile. `dem`, `q`, `prev`, `nxt` are optional."""
    out = intensity_features(r)
    if not np.isfinite(r).any():
        return out
    out.update(spatial_features(r))
    out.update(coarse_features(r, factor))
    if dem is not None:
        out.update(dem_features(r, dem))
    if q is not None:
        out.update(qind_features(r, q))
    out.update(temporal_features(r, prev, nxt))
    return out
