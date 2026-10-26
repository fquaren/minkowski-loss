"""Striping measures: grid-aligned or diagonal artefacts in a predicted field, against its own target.

The spectral anisotropy (`tail_metrics.directional_anisotropy`) averages a per-patch ratio whose spread on
real targets is larger than the effect it should detect, and it compares models with vanilla, which is itself
anisotropic on hourly data (DECISIONS §37). These measures pair each prediction with its target instead.

edge_orientation_shares(field)
    Among the strongest edges of log(1 + field) (Sobel; pixels >= `wet` mm, gradient magnitude above the
    `q`-quantile of the wet pixels), the share whose edge direction lies within ±`tol_deg` of the grid axes
    (0°, 90°) and of the diagonals (45°, 135°). Uniform directions give 2 x 2 tol / 180 each (11.1% at 5°).
    The Sobel stencil itself prefers some directions slightly; pairing with the target cancels that.
striping_excess(pred, target)
    Per patch, axis share and diagonal share of the prediction minus those of the target.
"""

import numpy as np
from scipy import ndimage


def edge_orientation_shares(field: np.ndarray, wet: float = 0.1, q: float = 0.9, tol_deg: float = 5.0):
    """(axis share, diagonal share, n edges) for one 2-D field; NaN shares if fewer than 20 edges."""
    z = np.log1p(np.clip(np.nan_to_num(field), 0, None).astype(np.float64))
    gx = ndimage.sobel(z, axis=1)
    gy = ndimage.sobel(z, axis=0)
    mag = np.hypot(gx, gy)
    m = (np.nan_to_num(field) >= wet) & (mag > 0)
    if m.sum() < 20:
        return np.nan, np.nan, 0
    thr = np.quantile(mag[m], q)
    e = m & (mag >= thr)
    n = int(e.sum())
    if n < 20:
        return np.nan, np.nan, n
    # direction of the gradient, folded to [0, 180); an edge is perpendicular to it, which maps the
    # axis set {0, 90} and the diagonal set {45, 135} onto themselves, so the gradient angle can be used.
    ang = np.degrees(np.arctan2(gy[e], gx[e])) % 180.0
    d_axis = np.minimum.reduce([np.abs(ang - a) for a in (0.0, 90.0, 180.0)])
    d_diag = np.minimum.reduce([np.abs(ang - a) for a in (45.0, 135.0)])
    return float((d_axis <= tol_deg).mean()), float((d_diag <= tol_deg).mean()), n


def highpass_orientation_shares(field: np.ndarray, wet: float = 0.1, sigma: float = 2.0, erode: int = 3,
                                q: float = 0.75, tol_deg: float = 5.0):
    """As `edge_orientation_shares`, on the small-scale texture inside the rain: r = log(1+f) minus its
    Gaussian smoothing (sigma px), on wet pixels at least `erode` px from the wet edge, strongest (1-q) of |grad r|.
    The first version (raw gradients) is dominated by wet/dry boundaries and misses texture (DECISIONS §38)."""
    f = np.clip(np.nan_to_num(field), 0, None).astype(np.float64)
    z = np.log1p(f)
    r = z - ndimage.gaussian_filter(z, sigma)
    gx, gy = ndimage.sobel(r, axis=1), ndimage.sobel(r, axis=0)
    mag = np.hypot(gx, gy)
    inner = ndimage.binary_erosion(f >= wet, iterations=erode)
    m = inner & (mag > 0)
    if m.sum() < 50:
        return np.nan, np.nan, 0
    e = m & (mag >= np.quantile(mag[m], q))
    ang = np.degrees(np.arctan2(gy[e], gx[e])) % 180.0
    d_axis = np.minimum.reduce([np.abs(ang - a) for a in (0.0, 90.0, 180.0)])
    d_diag = np.minimum.reduce([np.abs(ang - a) for a in (45.0, 135.0)])
    return float((d_axis <= tol_deg).mean()), float((d_diag <= tol_deg).mean()), int(e.sum())


def spectral_line_excess(field: np.ndarray, k_min_frac: float = 0.15, width: float = 1.0):
    """(axis, diagonal) log ratio of mean power on narrow lines through the spectrum (|ky| <= width or
    |kx| <= width bins; |kx - ky| or |kx + ky| <= width bins) to the mean power at the same |k| over all
    directions, for |k| above k_min_frac of Nyquist; Hann-windowed log(1 + f). Periodic stripes put their
    energy on such lines; an isotropic field gives ~0. The exact kx = 0 / ky = 0 lines are kept out of the
    axis lines' neighbours only by the window, so the target pairing is needed to cancel residual leakage."""
    f = np.log1p(np.clip(np.nan_to_num(field), 0, None).astype(np.float64))
    H, W = f.shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2((f - f.mean()) * win))) ** 2
    ky, kx = np.mgrid[-(H // 2):H - H // 2, -(W // 2):W - W // 2]
    kr = np.hypot(kx, ky)
    ring = kr >= k_min_frac * min(H, W) / 2
    axis = ring & ((np.abs(kx) <= width) | (np.abs(ky) <= width)) & (kx != 0) & (ky != 0)
    diag = ring & ((np.abs(kx - ky) <= width) | (np.abs(kx + ky) <= width))
    # reference: mean power in each integer |k| ring, evaluated at every pixel
    rb = np.rint(kr).astype(int)
    ref = np.bincount(rb.ravel(), P.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    R = P / np.maximum(ref[rb], 1e-30)
    if not axis.any() or not diag.any():
        return np.nan, np.nan
    return float(np.log(R[axis].mean())), float(np.log(R[diag].mean()))


def striping_excess(pred: np.ndarray, target: np.ndarray, **kw):
    """Per-patch (axis excess, diagonal excess) = prediction share - target share. Inputs [B, H, W]."""
    out = np.full((len(pred), 2), np.nan)
    for i in range(len(pred)):
        pa, pd_, _ = edge_orientation_shares(pred[i], **kw)
        ta, td, _ = edge_orientation_shares(target[i], **kw)
        out[i] = (pa - ta, pd_ - td)
    return out
