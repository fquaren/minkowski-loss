"""Tail-aware evaluation metrics for the extreme-precipitation protocol.

Complements src/evaluation/extreme_metrics.py (SAL, CRPS) with the quantities in the
in-distribution / out-of-distribution extreme tables:

  * fractions skill score (FSS), Roberts & Lean (2008), doi:10.1175/2007MWR2123.1
  * peaks-over-threshold generalised-Pareto fit and return levels, Coles (2001),
    doi:10.1007/978-1-4471-3675-0
  * a directional anisotropy diagnostic, which the radially averaged spectrum cannot
    see by construction (it integrates over direction).

FSS and the GPD fit are dataset-level statistics, not per-patch averages: FSS must be
accumulated as summed numerator/denominator over all patches before the ratio is taken,
and the GPD is fitted once to the pooled exceedances. The accumulator classes below
enforce that.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import genpareto


# --------------------------------------------------------------------------------------
# Fractions skill score
# --------------------------------------------------------------------------------------
class FSSAccumulator:
    """Accumulate the FSS numerator and denominator over a dataset.

    For threshold q and neighbourhood width n, the fractional coverage fields are
    obtained by a uniform (box) filter over the binary exceedance masks. Following
    Roberts & Lean (2008),

        FSS = 1 - sum (P_f - P_o)^2 / [ sum P_f^2 + sum P_o^2 ],

    with the sums taken over every pixel of every patch. Averaging per-patch FSS values
    instead would weight near-dry patches equally with extreme ones and is not the
    published definition.
    """

    def __init__(self, thresholds_mmph, windows):
        self.thresholds = [float(t) for t in thresholds_mmph]
        self.windows = [int(w) for w in windows]
        self._num = {(t, w): 0.0 for t in self.thresholds for w in self.windows}
        self._den = {(t, w): 0.0 for t in self.thresholds for w in self.windows}

    def update(self, pred: np.ndarray, obs: np.ndarray) -> None:
        """pred, obs: [B, H, W] physical fields (mm/h)."""
        pred = np.asarray(pred, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)
        for t in self.thresholds:
            bp = (pred >= t).astype(np.float64)
            bo = (obs >= t).astype(np.float64)
            for w in self.windows:
                # uniform_filter over the last two axes only (per patch)
                pf = ndimage.uniform_filter(bp, size=(1, w, w), mode="constant")
                po = ndimage.uniform_filter(bo, size=(1, w, w), mode="constant")
                self._num[(t, w)] += float(((pf - po) ** 2).sum())
                self._den[(t, w)] += float((pf**2).sum() + (po**2).sum())

    def result(self) -> dict:
        out = {}
        for key in self._num:
            t, w = key
            den = self._den[key]
            out[f"fss_thr{t:g}_win{w}"] = float(1.0 - self._num[key] / den) if den > 0 else float("nan")
        return out


# --------------------------------------------------------------------------------------
# Peaks over threshold: generalised-Pareto tail
# --------------------------------------------------------------------------------------
class POTAccumulator:
    """Collect exceedances over a fixed physical threshold for a later GPD fit.

    Exceedances are rare, so storing them is cheap; the total pixel count is tracked
    separately so the exceedance rate (needed for return levels) is exact. If the number
    of exceedances exceeds ``max_store`` they are uniformly subsampled, and the rate is
    still computed from the exact counts.
    """

    def __init__(self, threshold_mmph: float, max_store: int = 2_000_000, seed: int = 0):
        self.u = float(threshold_mmph)
        self.max_store = int(max_store)
        self._exc = []
        self._n_exc = 0
        self._n_total = 0
        self._rng = np.random.default_rng(seed)

    def update(self, field: np.ndarray) -> None:
        f = np.asarray(field, dtype=np.float64).ravel()
        self._n_total += f.size
        e = f[f > self.u] - self.u
        self._n_exc += e.size
        if e.size:
            self._exc.append(e.astype(np.float32))

    def _pooled(self) -> np.ndarray:
        if not self._exc:
            return np.empty(0, dtype=np.float64)
        e = np.concatenate(self._exc).astype(np.float64)
        if e.size > self.max_store:
            idx = self._rng.choice(e.size, self.max_store, replace=False)
            e = e[idx]
        return e

    def fit(self, min_exceedances: int = 200) -> dict:
        """Fit GPD(xi, sigma) to the pooled exceedances with the location fixed at 0."""
        e = self._pooled()
        rate = self._n_exc / self._n_total if self._n_total else float("nan")
        if e.size < min_exceedances:
            return {"xi": float("nan"), "sigma": float("nan"), "rate": rate,
                    "n_exceedances": int(self._n_exc), "threshold": self.u}
        xi, _, sigma = genpareto.fit(e, floc=0.0)
        return {"xi": float(xi), "sigma": float(sigma), "rate": float(rate),
                "n_exceedances": int(self._n_exc), "threshold": self.u}


def return_level(xi: float, sigma: float, u: float, rate: float, m: float) -> float:
    """m-observation return level from a POT/GPD fit (Coles 2001, ch. 4).

    x_m = u + (sigma/xi) [ (m * rate)^xi - 1 ],  with the xi -> 0 limit u + sigma*log(m*rate).
    Returns NaN when the fit is undefined or the level is not exceeded in expectation.
    """
    if not np.isfinite(xi) or not np.isfinite(sigma) or not np.isfinite(rate) or rate <= 0:
        return float("nan")
    lam = m * rate
    if lam <= 1.0:
        return float("nan")  # level not reachable at this return period
    if abs(xi) < 1e-8:
        return float(u + sigma * np.log(lam))
    return float(u + (sigma / xi) * (lam**xi - 1.0))


def gpd_comparison(pred_fit: dict, obs_fit: dict, lam_targets=(10.0, 100.0, 1000.0)) -> dict:
    """Shape-parameter error and relative return-level bias, predicted versus observed.

    Return levels are specified by the *expected number of exceedances* lam rather than a
    raw observation count m, because the level exists only when m * rate > 1 and the
    pixel-level exceedance rate is tiny (order 1e-6), so fixed m values are almost always
    unreachable. Given lam the return period is m = lam / rate_obs, and the *same* m is used
    for both fits so the return levels are comparable.

    A predicted return level of NaN where the observed one is finite is itself the finding:
    the fitted tail of the prediction cannot reach that level at all.
    """
    out = {
        "gpd_xi_pred": pred_fit["xi"], "gpd_xi_obs": obs_fit["xi"],
        "gpd_sigma_pred": pred_fit["sigma"], "gpd_sigma_obs": obs_fit["sigma"],
        "gpd_xi_abs_err": float(abs(pred_fit["xi"] - obs_fit["xi"])),
        "gpd_n_exc_pred": pred_fit["n_exceedances"], "gpd_n_exc_obs": obs_fit["n_exceedances"],
        "gpd_exc_count_ratio": (float(pred_fit["n_exceedances"] / obs_fit["n_exceedances"])
                                if obs_fit["n_exceedances"] else float("nan")),
    }
    # xi < 0 means a bounded tail; the implied upper endpoint is often the more
    # interpretable number when the sample itself is capped.
    for tag, fit in (("pred", pred_fit), ("obs", obs_fit)):
        xi, sg, u = fit["xi"], fit["sigma"], fit["threshold"]
        out[f"gpd_upper_endpoint_{tag}"] = (float(u - sg / xi)
                                            if np.isfinite(xi) and xi < 0 else float("inf"))
    rate_o = obs_fit["rate"]
    biases = []
    for lam in lam_targets:
        if not np.isfinite(rate_o) or rate_o <= 0:
            continue
        m = lam / rate_o
        rl_p = return_level(pred_fit["xi"], pred_fit["sigma"], pred_fit["threshold"],
                            pred_fit["rate"], m)
        rl_o = return_level(obs_fit["xi"], obs_fit["sigma"], obs_fit["threshold"],
                            obs_fit["rate"], m)
        out[f"rl_pred_lam{lam:g}"] = rl_p
        out[f"rl_obs_lam{lam:g}"] = rl_o
        if np.isfinite(rl_p) and np.isfinite(rl_o) and rl_o > 0:
            b = (rl_p - rl_o) / rl_o
            out[f"rl_relbias_lam{lam:g}"] = float(b)
            biases.append(abs(b))
    out["rl_bias_mean_abs"] = float(np.mean(biases)) if biases else float("nan")
    return out


# --------------------------------------------------------------------------------------
def directional_anisotropy(field: np.ndarray, wedge_deg: float = 22.5,
                           k_min_frac: float = 0.25, exclude_axis_lines: int = 1) -> float:
    """Ratio of axis-aligned to diagonal spectral power at high wavenumbers.

    The radially averaged spectrum averages over direction and therefore cannot detect
    grid-aligned striping. This diagnostic compares mean power in wedges around the kx
    and ky axes to wedges around the diagonals, restricted to |k| above k_min_frac of the
    Nyquist wavenumber. Isotropic fields give ~1; grid-aligned filamentary artefacts give
    values well above 1.

    Two corrections are essential and were both necessary in testing. A Hann window is
    applied before the transform, because a non-periodic patch leaks energy along the kx
    and ky axes and that leakage alone drives the raw ratio to ~1e6 on a perfectly smooth
    blob. The exact kx=0 and ky=0 lines are additionally excluded, since residual leakage
    concentrates there. Without both, the diagnostic fires on benign fields.

    field: [B, H, W] or [H, W].
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim == 2:
        f = f[None]
    B, H, W = f.shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(f * win[None], axes=(-2, -1)), axes=(-2, -1))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H))[:, None] * np.ones((1, W))
    kx = np.fft.fftshift(np.fft.fftfreq(W))[None, :] * np.ones((H, 1))
    k = np.sqrt(kx**2 + ky**2)
    ang = np.degrees(np.arctan2(ky, kx)) % 180.0
    kmax = k.max()
    band = k >= k_min_frac * kmax
    if exclude_axis_lines > 0:
        iy, ix = np.arange(H)[:, None] * np.ones((1, W)), np.ones((H, 1)) * np.arange(W)[None, :]
        cy, cx = H // 2, W // 2
        on_axis = (np.abs(iy - cy) < exclude_axis_lines) | (np.abs(ix - cx) < exclude_axis_lines)
        band = band & ~on_axis
    axis = band & ((ang < wedge_deg) | (ang > 180.0 - wedge_deg) | (np.abs(ang - 90.0) < wedge_deg))
    diag = band & ((np.abs(ang - 45.0) < wedge_deg) | (np.abs(ang - 135.0) < wedge_deg))
    if axis.sum() == 0 or diag.sum() == 0:
        return float("nan")
    # A band-limited (perfectly smooth) field has no high-wavenumber content, so the ratio
    # there is numerical noise over numerical noise. Report NaN rather than a garbage value.
    if float(P[:, band].sum()) < 1e-8 * float(P.sum()):
        return float("nan")
    pa = float(P[:, axis].mean())
    pd = float(P[:, diag].mean())
    floor = 1e-12 * float(P[:, band].mean())
    return pa / max(pd, floor)


def peak_ratio(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-patch ratio of predicted to observed maximum (SAL A compares means, not peaks)."""
    p = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1).max(axis=1)
    o = np.asarray(obs, dtype=np.float64).reshape(obs.shape[0], -1).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(o > 1e-6, p / o, np.nan)
    return r