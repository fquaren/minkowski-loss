"""Day-block uncertainty for the extreme-precipitation tables (DECISIONS §32 reporting rule).

Tail statistics on the test split are driven by a few outbreak days (v4: 20 of 486 days hold 27%
of the pixel-hours >= 20.2 mm), so every table column is reported with a 95% day-block bootstrap
interval and with leave-period-out variants. The bootstrap resamples processing days and applies
the same day weights to prediction and observation, so model-vs-observation comparisons stay
paired.

`DayRecord` keeps, per sample: its day, the FSS numerator/denominator per (threshold, window), the
squared error, the patch maximum, and the POT exceedances of prediction and observation. `table`
recomputes the columns from those under any day weights.

GPD fits here use a weighted maximum likelihood (`gpd_fit_weighted`), so the interval is taken
for that estimator; its full-sample value (`*_dayblock_point`) is reported beside the
interval, and it differs from `genpareto.fit` only by the optimiser.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.tail_metrics import gpd_comparison


def gpd_fit_weighted(x: np.ndarray, w: np.ndarray):
    """Weighted MLE of GPD(xi, sigma), location 0, for excesses x > 0 with weights w."""
    from scipy.optimize import minimize
    W = w.sum()
    if W <= 0 or x.size < 3:
        return np.nan, np.nan
    m = (w * x).sum() / W
    s2 = (w * (x - m) ** 2).sum() / W
    xi0 = float(np.clip(0.5 * (1 - m * m / max(s2, 1e-12)), -0.4, 0.8))
    sg0 = max(0.5 * m * (m * m / max(s2, 1e-12) + 1), 1e-6)
    xmax = x.max()

    def nll(p):
        ls, xi = p
        sg = np.exp(ls)
        if xi < 0 and xmax >= -sg / xi:
            return 1e300
        if abs(xi) < 1e-7:
            return W * ls + (w * x).sum() / sg
        return W * ls + (1 + 1 / xi) * (w * np.log1p(xi * x / sg)).sum()

    r = minimize(nll, [np.log(sg0), xi0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10 * W, "maxiter": 4000})
    return float(r.x[1]), float(np.exp(r.x[0]))


def processing_days(meta_path: str, n: int) -> np.ndarray:
    """YYYYMMDD processing day of the first n rows of a metadata file (hour end at 00:00
    belongs to the day before, as in the hourly chain)."""
    import pandas as pd
    ts = pd.read_csv(meta_path, header=None, usecols=[0], dtype=str, nrows=n)[0]
    t = pd.to_datetime(ts.str[:14], format="%Y%m%d%H%M%S")
    return (t - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d").values


class DayRecord:
    def __init__(self, fss_keys, pot_u: float):
        self.fss_keys = list(fss_keys)
        self.u = float(pot_u)
        self.num = {k: [] for k in self.fss_keys}
        self.den = {k: [] for k in self.fss_keys}
        self.mse, self.tmax, self.npix = [], [], []
        self.exc = {"pred": [], "obs": []}            # (sample index, excess)
        self.n = 0

    def update(self, fss_per_sample: dict, mse, tmax, pred, obs):
        B = len(mse)
        for k in self.fss_keys:
            self.num[k].append(fss_per_sample[k][0]); self.den[k].append(fss_per_sample[k][1])
        self.mse.append(np.asarray(mse)); self.tmax.append(np.asarray(tmax))
        self.npix.append(np.full(B, pred[0].size))
        for tag, f in (("pred", pred), ("obs", obs)):
            f = np.asarray(f, dtype=np.float64).reshape(B, -1)
            s, j = np.nonzero(f > self.u)
            self.exc[tag].append((s + self.n, f[s, j] - self.u))
        self.n += B

    def finalize(self, days: np.ndarray, ext_thr: float, lam_targets):
        cat = np.concatenate
        self.days, self.day_codes = np.unique(days[: self.n], return_inverse=True)
        self.D = len(self.days)
        self.num = {k: cat(v) for k, v in self.num.items()}
        self.den = {k: cat(v) for k, v in self.den.items()}
        self.mse, self.tmax, self.npix = cat(self.mse), cat(self.tmax), cat(self.npix)
        self.ext = self.tmax >= ext_thr
        for tag in self.exc:
            s = cat([a for a, _ in self.exc[tag]]) if self.exc[tag] else np.zeros(0, int)
            e = cat([b for _, b in self.exc[tag]]) if self.exc[tag] else np.zeros(0)
            self.exc[tag] = (self.day_codes[s], e)
        self.lams = list(lam_targets)

    def table(self, day_w: np.ndarray) -> dict:
        """Table columns under day weights day_w [D] (1 = full sample)."""
        w = day_w[self.day_codes]
        out = {}
        sw = w.sum()
        out["rmse_overall"] = float(np.sqrt((w * self.mse).sum() / sw)) if sw > 0 else np.nan
        we = w * self.ext
        out["rmse_extreme"] = float(np.sqrt((we * self.mse).sum() / we.sum())) if we.sum() > 0 else np.nan
        for (t, win) in self.fss_keys:
            den = (w * self.den[(t, win)]).sum()
            out[f"fss_thr{t:g}_win{win}"] = float(1 - (w * self.num[(t, win)]).sum() / den) if den > 0 else np.nan
        ntot = (w * self.npix).sum()
        fits = {}
        for tag in ("pred", "obs"):
            d, e = self.exc[tag]
            ew = day_w[d]
            xi, sg = gpd_fit_weighted(e, ew)
            fits[tag] = {"xi": xi, "sigma": sg, "rate": float(ew.sum() / ntot) if ntot > 0 else np.nan,
                         "n_exceedances": float(ew.sum()), "threshold": self.u}
        g = gpd_comparison(fits["pred"], fits["obs"], self.lams)
        for k in ("gpd_xi_pred", "gpd_xi_obs", "gpd_xi_abs_err", "gpd_exc_count_ratio", "rl_bias_mean_abs"):
            out[k] = g[k]
        out.update({k: v for k, v in g.items() if k.startswith("rl_relbias")})
        return out

    def intervals(self, B: int, seed: int = 0, level: float = 95.0) -> dict:
        rng = np.random.default_rng(seed)
        rows = [self.table(rng.multinomial(self.D, np.full(self.D, 1 / self.D)).astype(float)) for _ in range(B)]
        lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
        out = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows], float)
            out[k] = [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] if np.isfinite(v).any() else [np.nan, np.nan]
        return out

    def leave_out(self, drop_days) -> dict:
        drop = set(drop_days)
        w = np.array([0.0 if d in drop else 1.0 for d in self.days])
        return {"n_days_dropped": int((w == 0).sum()), **self.table(w)}


def parse_leave_out(specs):
    """Specs: 'YYYYMMDD-YYYYMMDD' ranges, or '@file' with one YYYYMMDD per line. Returns
    {name: set of days}."""
    import pandas as pd
    out = {}
    for s in specs or []:
        if s.startswith("@"):
            out[s[1:]] = {l.strip() for l in open(s[1:]) if l.strip()}
        else:
            a, b = s.split("-")
            out[s] = {d.strftime("%Y%m%d") for d in pd.date_range(a, b)}
    return out
