"""Global empirical quantile mapping of predicted precipitation (Baselines v2, G1).

The rival explanation it tests: "a post-hoc correction of the marginal gives the tail gain".
The map sends each predicted value to the observed value at the same non-exceedance level,
both marginals pooled over every pixel of a training sample. Above the top fitted quantile
it extrapolates multiplicatively, x * q_obs_top / q_pred_top. Fitted on the cut list, the
observed marginal stops below the cut, so a cut model's QM can reach the tail only through
that extrapolation.

Marginals are accumulated as histograms in log1p(mm), with the exact zeros counted apart, so
a fit over ~1e8 pixels needs no pixel storage; the bin width (2e-4 in log1p) is a 0.02%
relative error in the quantiles.
"""
from __future__ import annotations

import json

import numpy as np

LOG_MAX, NBINS = float(np.log1p(2000.0)), 40000


def default_levels():
    """Non-exceedance levels: uniform to 0.9, then log-spaced exceedances down to 1e-6."""
    return np.unique(np.concatenate([np.linspace(0.0, 0.9, 901), 1.0 - np.logspace(-1, -6, 101)]))


class LogHistogram:
    """Pooled pixel marginal of one field type (prediction or target)."""

    def __init__(self):
        self.counts = np.zeros(NBINS, dtype=np.int64)
        self.n_zero = 0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        z = x <= 0
        self.n_zero += int(z.sum())
        self.counts += np.histogram(np.log1p(x[~z]), bins=NBINS, range=(0.0, LOG_MAX))[0]
        self.counts[-1] += int((np.log1p(x[~z]) > LOG_MAX).sum())     # clamp into the top bin

    def quantiles(self, levels):
        """Quantile (mm) at each level; linear in log1p within a bin, 0 inside the zero mass."""
        n = self.n_zero + self.counts.sum()
        edges = np.linspace(0.0, LOG_MAX, NBINS + 1)
        cdf = (self.n_zero + np.concatenate([[0], np.cumsum(self.counts)])) / n    # at each edge
        out = np.zeros(len(levels))
        for i, p in enumerate(levels):
            if p * n <= self.n_zero:
                continue
            j = min(int(np.searchsorted(cdf, p, side="left")), NBINS)              # cdf[j-1] < p <= cdf[j]
            j = max(j, 1)
            lo, hi = cdf[j - 1], cdf[j]
            f = 0.0 if hi <= lo else (p - lo) / (hi - lo)
            out[i] = np.expm1(edges[j - 1] + f * (edges[j] - edges[j - 1]))
        return out


def fit_table(h_pred: LogHistogram, h_obs: LogHistogram, levels=None, min_above: int = 100) -> dict:
    """Quantile table on the levels with at least min_above pixels above them: higher levels
    sit on a handful of pixels, and the top level sets the extrapolation ratio."""
    levels = default_levels() if levels is None else np.asarray(levels)
    n = int(h_obs.n_zero + h_obs.counts.sum())
    levels = levels[(1.0 - levels) * n >= min_above]
    return {"levels": levels.tolist(), "q_pred": h_pred.quantiles(levels).tolist(),
            "q_obs": h_obs.quantiles(levels).tolist(),
            "n_pixels": n, "min_above": min_above}


class QuantileMap:
    """x -> q_obs(F_pred(x)), interpolated in log1p between the fitted quantiles."""

    def __init__(self, q_pred, q_obs):
        qp, qo = np.asarray(q_pred, float), np.asarray(q_obs, float)
        # A tied prediction quantile (e.g. a zero mass) maps to the lowest observed quantile of
        # the tie, so a dry prediction stays at the observed value of its lowest level.
        qp, first = np.unique(qp, return_index=True)
        self.kp, self.ko = np.log1p(qp), np.log1p(np.maximum.accumulate(qo)[first])
        self.ratio = float(np.expm1(self.ko[-1]) / np.expm1(self.kp[-1])) if self.kp[-1] > 0 else 1.0

    @classmethod
    def load(cls, path):
        t = json.load(open(path))
        return cls(t["q_pred"], t["q_obs"])

    def __call__(self, x):
        """numpy array or torch tensor in mm, any shape."""
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return torch.as_tensor(self(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device)
        except ImportError:
            pass
        x = np.asarray(x, dtype=np.float64)
        top = np.expm1(self.kp[-1])
        y = np.expm1(np.interp(np.log1p(np.clip(x, 0.0, None)), self.kp, self.ko))
        return np.where(x > top, x * self.ratio, y)
