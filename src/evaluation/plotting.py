"""Comparison plots for super-resolution quality: accuracy, perception, and the
Minkowski functionals themselves.

Every function consumes the artefacts the evaluation scripts already write
(``*_summary.yaml`` and ``*_arrays.npz``) and takes a list of ``Model`` records, so adding a
row to a figure means pointing at another eval directory rather than re-running anything.

Three groups:

  accuracy    perception_distortion, tail_summary, survival_and_return_levels
  perception  rapsd, sal_plane
  Minkowski   gamma_curves, gamma_residual_by_threshold, isoperimetric

The Minkowski group is the one that is hard to get elsewhere. ``gamma_curves`` shows what the
loss actually compares -- the area, perimeter and topology curves against threshold -- and
``gamma_residual_by_threshold`` shows *where* along the threshold axis the discrepancy sits.
That second plot is the one to read alongside any claim about extremes: the loss integrates
with equal weight per decade, but its gradient is concentrated wherever pixels happen to lie
near a threshold, which for precipitation is the drizzle boundary.

Conventions follow the earlier Mink-DDPM / ExtremePrecipSR figures: masked ``Blues`` for
precipitation fields, log-log spectra with a wavelength axis on top, physical units
(mm/h) on every axis that has them.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# A colour-blind-safe qualitative cycle; the target is always black.
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494", "#ECE133"]
TARGET_STYLE = dict(color="black", linestyle="--", linewidth=2.2, zorder=10)
CHANNELS = ["area $A(u)$", "perimeter $P(u)$", "topology $T(u)$"]


@dataclass
class Model:
    """One evaluated model: a label plus whatever artefacts exist for it."""
    label: str
    summary: dict = field(default_factory=dict)
    arrays: Optional[dict] = None
    color: Optional[str] = None

    def get(self, key, default=None):
        return self.summary.get(key, default)

    def arr(self, key, default=None):
        if self.arrays is None or key not in self.arrays:
            return default
        return self.arrays[key]


def load_model(label: str, eval_dir: str, color: Optional[str] = None) -> Optional[Model]:
    """Load one eval directory. Returns None if no summary is present."""
    import yaml
    summary, arrays = None, None
    for name in ("extremes_summary.yaml", "backbone_summary.yaml", "fm_baseline_summary.yaml"):
        p = os.path.join(eval_dir, name)
        if os.path.exists(p):
            summary = yaml.safe_load(open(p))
            break
    if summary is None:
        return None
    for name in ("extremes_arrays.npz", "backbone_arrays.npz", "fm_baseline_arrays.npz"):
        p = os.path.join(eval_dir, name)
        if os.path.exists(p):
            arrays = dict(np.load(p, allow_pickle=False))
            break
    return Model(label=label, summary=summary, arrays=arrays, color=color)


def assign_colors(models: Sequence[Model]) -> None:
    for i, m in enumerate(models):
        if m.color is None:
            m.color = PALETTE[i % len(PALETTE)]


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _wavelength_axis(ax, pixel_km: float = 2.0):
    """Secondary top axis in km, given a frequency axis in cycles per pixel."""
    def f2w(x):
        return pixel_km / np.maximum(x, 1e-10)

    def w2f(x):
        return pixel_km / np.maximum(x, 1e-10)

    sec = ax.secondary_xaxis("top", functions=(f2w, w2f))
    sec.set_xlabel("wavelength [km]")
    return sec


# ----------------------------------------------------------------------------------
# accuracy and perception
# ----------------------------------------------------------------------------------
def perception_distortion(models: Sequence[Model], out_dir: str,
                          distortion_key: str = "rmse_extreme",
                          perception_key: str = "rapsd_log_distance",
                          name: str = "perception_distortion.png"):
    """The perception--distortion plane, one point per model, with the Pareto front.

    Distortion on x, perceptual error on y, both lower-better, so the lower-left corner is
    best and the front traces the achievable compromise. A structural loss that only trades
    accuracy for realism moves along the front; one that dominates moves inside it.
    """
    pts = [(m, m.get(distortion_key), m.get(perception_key)) for m in models]
    pts = [(m, x, y) for m, x, y in pts if x is not None and y is not None]
    if len(pts) < 2:
        print("  [skip] perception_distortion: need >= 2 models with both metrics")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m, x, y in pts:
        ax.scatter(x, y, s=140, color=m.color, edgecolor="k", linewidth=0.8, zorder=5)
        ax.annotate(m.label, (x, y), textcoords="offset points", xytext=(9, 5), fontsize=9)

    # Pareto front: points not dominated on both axes simultaneously.
    front = [(x, y) for _, x, y in pts
             if not any((xx <= x and yy <= y and (xx < x or yy < y)) for _, xx, yy in pts)]
    if len(front) > 1:
        front.sort()
        ax.plot([p[0] for p in front], [p[1] for p in front],
                color="grey", linestyle=":", linewidth=1.5, zorder=1,
                label="Pareto front")
        ax.legend(frameon=False, fontsize=9)

    ax.set_xlabel(f"distortion — {distortion_key}  (lower better)")
    ax.set_ylabel(f"perception — {perception_key}  (lower better)")
    ax.set_title("Perception–distortion plane")
    ax.grid(alpha=0.3, linestyle="--")
    return _save(fig, out_dir, name)


def rapsd(models: Sequence[Model], out_dir: str, pixel_km: float = 2.0,
          name: str = "rapsd.png"):
    """Radially averaged power spectra, and the ratio to the target.

    The lower panel is the informative one: a flat line at 1 means the model puts the right
    amount of variance at every scale, and the wavenumber at which it falls away marks the
    effective resolution. Note this is a radial average, so it is blind to grid-aligned
    structure by construction -- read it alongside the anisotropy diagnostic.
    """
    have = [m for m in models if m.arr("rapsd_pred") is not None
            or m.arr("rapsd_member") is not None]
    if not have:
        print("  [skip] rapsd: no spectra in the arrays")
        return None
    tgt = None
    for m in have:
        t = m.arr("rapsd_target")
        if t is not None:
            tgt = np.asarray(t)
            break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    for m in have:
        p = m.arr("rapsd_pred")
        if p is None:
            p = m.arr("rapsd_member")
        p = np.asarray(p)
        k = np.arange(1, len(p) + 1) / (2.0 * len(p))       # cycles per pixel
        ax1.loglog(k, p, color=m.color, linewidth=1.8, label=m.label)
        if tgt is not None and len(tgt) == len(p):
            ax2.semilogx(k, p / np.maximum(tgt, 1e-30), color=m.color, linewidth=1.6)
    if tgt is not None:
        k = np.arange(1, len(tgt) + 1) / (2.0 * len(tgt))
        ax1.loglog(k, tgt, label="target", **TARGET_STYLE)
        ax2.axhline(1.0, **TARGET_STYLE)

    ax1.set_ylabel("power")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3, which="both", linestyle="--")
    ax1.set_title("Radially averaged power spectrum")
    _wavelength_axis(ax1, pixel_km)

    ax2.set_xlabel("wavenumber [cycles / pixel]")
    ax2.set_ylabel("ratio to target")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both", linestyle="--")
    return _save(fig, out_dir, name)


def survival_and_return_levels(models: Sequence[Model], out_dir: str,
                               name: str = "tail_survival.png"):
    """Left: exceedance counts relative to observed. Right: GPD return-level curves.

    The return-level panel is the one a practitioner reads: it converts the fitted tail into
    the intensity expected once in every m observations. A curve below the target that stays
    parallel is a scale error; one that crosses is a shape error, and the two call for
    different fixes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels, ratios, colors = [], [], []
    for m in models:
        r = m.get("gpd_exc_count_ratio")
        if r is not None:
            labels.append(m.label); ratios.append(r); colors.append(m.color)
    if ratios:
        y = np.arange(len(ratios))
        ax1.barh(y, ratios, color=colors, edgecolor="k", linewidth=0.6)
        ax1.axvline(1.0, **TARGET_STYLE)
        ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel("predicted / observed exceedance count")
        ax1.set_title("Tail population (1.0 = correct)")
        ax1.grid(alpha=0.3, axis="x", linestyle="--")

    lam = np.logspace(0.3, 3.2, 40)
    drew_target = False
    for m in models:
        gp, go = m.arr("gpd_pred"), m.arr("gpd_obs")
        if gp is None or go is None:
            continue
        u = m.get("pot_threshold_mmph", m.arr("pot_threshold"))
        if u is None:
            print(f"  [warn] {m.label}: no stored POT threshold; "
                  f"re-run eval_extremes to record it. Skipping its return levels.")
            continue
        u = float(u)
        if not drew_target:
            ax2.semilogx(lam, _return_level_curve(np.asarray(go), u, lam),
                         label="target", **TARGET_STYLE)
            drew_target = True
        ax2.semilogx(lam, _return_level_curve(np.asarray(gp), u, lam),
                     color=m.color, linewidth=1.8, label=m.label)

    ax2.set_xlabel(r"expected exceedance count $\lambda$")
    ax2.set_ylabel("return level [mm h$^{-1}$]")
    ax2.set_title("Generalised-Pareto return levels")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3, which="both", linestyle="--")
    return _save(fig, out_dir, name)


def _return_level_curve(gpd, u, lam):
    """gpd = [xi, sigma, rate]; return levels at expected exceedance counts lam."""
    xi, sg, rate = float(gpd[0]), float(gpd[1]), float(gpd[2])
    out = np.full_like(lam, np.nan, dtype=float)
    ok = lam > 1.0
    if abs(xi) < 1e-8:
        out[ok] = u + sg * np.log(lam[ok])
    else:
        out[ok] = u + (sg / xi) * (lam[ok] ** xi - 1.0)
    return out


def tail_summary(models: Sequence[Model], out_dir: str, name: str = "tail_summary.png"):
    """Dot chart of the tail and safety diagnostics, each on its own scale.

    Anisotropy and peak ratio are included deliberately: they are not optimised by any of the
    losses, so they are the columns that catch a model satisfying a structural objective with
    grid-aligned filaments or suppressed peaks.
    """
    metrics = [
        ("fss_thr89_win5", "FSS @ 89 mm/h", "higher", None),
        ("gpd_exc_count_ratio", "exceedance ratio", "target", 1.0),
        ("rl_bias_mean_abs", "return-level bias", "lower", 0.0),
        ("peak_ratio_median", "peak ratio", "target", 1.0),
        ("anisotropy_axis_over_diag", "anisotropy", "target", 1.0),
        ("rapsd_log_distance", "RAPSD distance", "lower", 0.0),
    ]
    present = [(k, t, d, r) for k, t, d, r in metrics
               if any(m.get(k) is not None for m in models)]
    if not present:
        print("  [skip] tail_summary: no metrics found")
        return None

    fig, axes = plt.subplots(1, len(present), figsize=(2.5 * len(present), 5), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(models))
    for ax, (key, title, direction, ref) in zip(axes, present):
        vals = [m.get(key, np.nan) for m in models]
        ax.scatter(vals, y, s=90, color=[m.color for m in models],
                   edgecolor="k", linewidth=0.7, zorder=5)
        for yy, v in zip(y, vals):
            if v is not None and np.isfinite(v):
                ax.plot([ref if ref is not None else min(v, 0), v], [yy, yy],
                        color="lightgrey", zorder=1)
        if ref is not None:
            ax.axvline(ref, **TARGET_STYLE)
        ax.set_title(f"{title}\n({direction})", fontsize=9)
        ax.grid(alpha=0.3, axis="x", linestyle="--")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([m.label for m in models], fontsize=9)
    fig.suptitle("Tail and safety diagnostics", y=1.02)
    return _save(fig, out_dir, name)


def sal_plane(models: Sequence[Model], out_dir: str, name: str = "sal_plane.png"):
    """SAL structure vs amplitude, medians with interquartile whiskers.

    The origin is perfect. Positive S means objects too large and flat (over-smoothing);
    negative S means too peaked. Amplitude is a domain-mean comparison and is insensitive to
    peak height, which is why peak ratio is reported separately.
    """
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for m in models:
        S, A = m.arr("S"), m.arr("A")
        if S is None or A is None:
            s, a = m.get("SAL_median", [None, None, None])[:2]
            if s is None:
                continue
            ax.scatter(a, s, s=140, color=m.color, edgecolor="k", zorder=5)
            ax.annotate(m.label, (a, s), textcoords="offset points", xytext=(8, 5), fontsize=9)
            continue
        S, A = np.asarray(S), np.asarray(A)
        ms, ma = np.nanmedian(S), np.nanmedian(A)
        ax.errorbar(ma, ms,
                    xerr=[[ma - np.nanpercentile(A, 25)], [np.nanpercentile(A, 75) - ma]],
                    yerr=[[ms - np.nanpercentile(S, 25)], [np.nanpercentile(S, 75) - ms]],
                    fmt="o", markersize=10, color=m.color, ecolor=m.color,
                    elinewidth=1.2, capsize=3, markeredgecolor="k", zorder=5)
        ax.annotate(m.label, (ma, ms), textcoords="offset points", xytext=(9, 6), fontsize=9)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("SAL amplitude $A$")
    ax.set_ylabel("SAL structure $S$   (positive = over-smoothed)")
    ax.set_title("SAL plane (median, IQR whiskers)")
    ax.grid(alpha=0.3, linestyle="--")
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------
# the Minkowski functionals themselves
# ----------------------------------------------------------------------------------
def _gamma_pred_log(m: Model):
    """Predicted gamma in the same signed-log space as the stored target."""
    g = m.arr("gamma_hat")
    if g is None:
        return None
    g = np.asarray(g, dtype=np.float64)
    return np.sign(g) * np.log1p(np.abs(g))


def gamma_curves(models: Sequence[Model], out_dir: str, name: str = "gamma_curves.png"):
    """The three Minkowski curves against threshold: what the loss actually compares.

    Area is the marginal survival function scaled by domain area, so the right-hand end of
    the first panel is the extreme tail directly. A model whose area curve falls below the
    target at high u is under-producing extremes, by definition.
    """
    have = [m for m in models if _gamma_pred_log(m) is not None]
    if not have:
        print("  [skip] gamma_curves: no gamma_hat in the arrays "
              "(re-run eval_backbone after the gamma_target patch)")
        return None
    u = None
    tgt = None
    for m in have:
        if u is None and m.arr("thresholds") is not None:
            u = np.asarray(m.arr("thresholds"), dtype=float)
        if tgt is None and m.arr("gamma_target") is not None:
            tgt = np.asarray(m.arr("gamma_target"), dtype=np.float64).mean(axis=0)
    Q = _gamma_pred_log(have[0]).shape[-1]
    if u is None or len(u) != Q:
        u = np.arange(1, Q + 1, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for c, ax in enumerate(axes):
        for m in have:
            g = _gamma_pred_log(m).mean(axis=0)
            ax.plot(u, g[c], color=m.color, linewidth=1.8, marker="o", markersize=3,
                    label=m.label)
        if tgt is not None and tgt.shape[0] > c:
            ax.plot(u, tgt[c], label="target", marker="s", markersize=3, **TARGET_STYLE)
        ax.set_xscale("log")
        ax.set_xlabel("threshold $u$ [mm h$^{-1}$]")
        ax.set_ylabel(r"signed $\log(1+\cdot)$")
        ax.set_title(CHANNELS[c] if c < len(CHANNELS) else f"channel {c}")
        ax.grid(alpha=0.3, which="both", linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Minkowski functional curves (dataset mean)", y=1.03)
    return _save(fig, out_dir, name)


def gamma_residual_by_threshold(models: Sequence[Model], out_dir: str,
                                name: str = "gamma_residual.png"):
    """Where along the threshold axis the Minkowski discrepancy actually sits.

    The loss integrates over log-intensity with equal weight per decade, so a flat residual
    would mean every decade contributes equally. In practice the residual grows toward the
    tail while the gradient available there collapses, because almost no pixels lie near the
    high thresholds -- this figure is the evidence for that claim, and the reason threshold
    reweighting matters more than the smoothing parameter.
    """
    have = [m for m in models
            if _gamma_pred_log(m) is not None and m.arr("gamma_target") is not None]
    if not have:
        print("  [skip] gamma_residual: needs both gamma_hat and gamma_target")
        return None
    u = np.asarray(have[0].arr("thresholds"), dtype=float) \
        if have[0].arr("thresholds") is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in have:
        p = _gamma_pred_log(m)
        t = np.asarray(m.arr("gamma_target"), dtype=np.float64)
        n = min(len(p), len(t))
        resid = np.abs(p[:n] - t[:n])                     # [N,3,Q]
        per_thr = resid.mean(axis=(0, 1))                 # [Q]
        x = u if (u is not None and len(u) == len(per_thr)) else np.arange(len(per_thr))
        axes[0].plot(x, per_thr, color=m.color, linewidth=1.8, marker="o", markersize=3,
                     label=m.label)
        per_ch = resid.mean(axis=(0, 2))                  # [3]
        axes[1].bar(np.arange(len(per_ch)) + 0.8 * (have.index(m) / max(len(have), 1)) - 0.4,
                    per_ch, width=0.8 / max(len(have), 1), color=m.color,
                    edgecolor="k", linewidth=0.5, label=m.label)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("threshold $u$ [mm h$^{-1}$]")
    axes[0].set_ylabel(r"mean $|\log\hat\gamma - \log\gamma|$")
    axes[0].set_title("Minkowski residual by threshold")
    axes[0].grid(alpha=0.3, which="both", linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xticks(np.arange(len(CHANNELS)))
    axes[1].set_xticklabels(["area", "perimeter", "topology"])
    axes[1].set_ylabel(r"mean $|\Delta\log\gamma|$")
    axes[1].set_title("Residual by channel")
    axes[1].grid(alpha=0.3, axis="y", linestyle="--")
    return _save(fig, out_dir, name)


def isoperimetric(models: Sequence[Model], out_dir: str, name: str = "isoperimetric.png"):
    """Perimeter against the isoperimetric bound $P \\ge \\sqrt{4\\pi A}$.

    Points below the line are geometrically impossible and indicate the functionals have been
    driven off the manifold of realisable shapes. Points far *above* it are thin and
    filamentary -- not impossible, but the signature of a model manufacturing perimeter, so
    read the vertical spread as well as the violation count.
    """
    have = [m for m in models if m.arr("gamma_hat") is not None]
    if not have:
        print("  [skip] isoperimetric: no gamma_hat in the arrays")
        return None
    n = len(have)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    for ax, m in zip(axes[0], have):
        g = np.asarray(m.arr("gamma_hat"), dtype=np.float64)
        A, P = g[:, 0, :].ravel(), g[:, 1, :].ravel()
        keep = A > 1e-2
        A, P = A[keep], P[keep]
        if A.size == 0:
            ax.set_title(f"{m.label}\n(no valid excursion sets)")
            continue
        Pmin = np.sqrt(4.0 * np.pi * A)
        viol = float(np.mean(P < Pmin - 1e-4) * 100.0)
        ax.scatter(np.sqrt(A), P, s=4, alpha=0.25, color=m.color, edgecolor="none")
        xs = np.linspace(0, np.sqrt(A).max(), 50)
        ax.plot(xs, np.sqrt(4.0 * np.pi) * xs, color="k", linewidth=1.6,
                label=r"$P=\sqrt{4\pi A}$")
        ax.set_xlabel(r"$\sqrt{A}$")
        ax.set_ylabel("$P$")
        ax.set_title(f"{m.label}\nviolations {viol:.2f}%")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
    fig.suptitle("Isoperimetric consistency", y=1.03)
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------
# qualitative fields
# ----------------------------------------------------------------------------------
def precip_cmap(bad="lightgrey"):
    cm = copy.copy(plt.get_cmap("Blues"))
    cm.set_bad(color=bad, alpha=1.0)
    return cm


def field_panel(fields: dict, out_dir: str, drizzle: float = 0.1,
                name: str = "fields.png", suptitle: Optional[str] = None):
    """Side-by-side precipitation fields on one shared colour scale.

    ``fields`` maps a panel title to a 2-D array in mm/h; sub-drizzle pixels are masked so the
    dry background reads as grey rather than as the bottom of the colour scale. A shared
    ``vmax`` is essential -- per-panel normalisation hides exactly the amplitude differences
    these comparisons are about.
    """
    if not fields:
        return None
    vmax = max(float(np.nanmax(v)) for v in fields.values())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cm = precip_cmap()
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0), squeeze=False)
    for ax, (title, arr) in zip(axes[0], fields.items()):
        a = np.ma.masked_less(np.asarray(arr, dtype=float), drizzle)
        im = ax.imshow(a, cmap=cm, norm=norm, origin="lower")
        ax.set_title(f"{title}\nmax {np.nanmax(arr):.1f} mm h$^{{-1}}$", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label="precipitation [mm h$^{-1}$]")
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    return _save(fig, out_dir, name)


def make_all(models: Sequence[Model], out_dir: str, pixel_km: float = 2.0):
    """Every comparison figure the loaded artefacts support."""
    assign_colors(models)
    print(f"plotting {len(models)} models -> {out_dir}")
    perception_distortion(models, out_dir)
    rapsd(models, out_dir, pixel_km=pixel_km)
    survival_and_return_levels(models, out_dir)
    tail_summary(models, out_dir)
    sal_plane(models, out_dir)
    gamma_curves(models, out_dir)
    gamma_residual_by_threshold(models, out_dir)
    isoperimetric(models, out_dir)
