#!/usr/bin/env python
"""Figures for notes/minkowski_reward_coupling.tex (DECISIONS §39).

1. survival.pdf: vertical (threshold) vs horizontal (quantile) comparison of a real
   prediction/target pair (patch 979 of the v4adv heavy validation subset, clean FM member 0).
2. share.pdf: share of the reward gradient landing on pixels of each intensity bin, original
   form vs quantile_mass (alpha = 1), from logs/reward_threshold_share_v4adv.txt.
3. fmloss.pdf: share of pixels, of the FM loss and of |dL/dv| per target-intensity bin
   (DECISIONS §40a), from logs/fm_loss_intensity_share_v4adv.txt.
4. tailsnr.pdf: bias/spread of the four reward features and the share of members out of
   reach, per threshold (DECISIONS §40b), from logs/tail_reading_snr_v4adv.txt.

Run from the repository root.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "notes/figures/reward_coupling"
C_TGT, C_PRED = "#2a78d6", "#eb6834"         # categorical slots 1 and 2
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- 1. survival curves ------------------------------------------------------------
d = np.load("eval_results/fields/v4adv_valext_pct.npz", allow_pickle=True)
p = int(np.where(d["indices"] == 979)[0][0])
y = np.sort(d["target"][p].ravel())[::-1]
f = np.sort(d["preds"][list(d["labels"]).index("FM"), p].ravel())[::-1]
u = d["thresholds"]
rank = np.arange(1, y.size + 1)

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.step(y, rank, where="post", color=C_TGT, lw=2, label="target")
ax.step(f, rank, where="post", color=C_PRED, lw=2, label="prediction (clean FM, one member)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.08, 120); ax.set_ylim(0.8, 2e4)
ax.set_xlabel("intensity v [mm]"); ax.set_ylabel("pixels with value ≥ v")
for uj in u:
    ax.axvline(uj, color=MUTED, lw=0.5, alpha=0.4, zorder=0)
# vertical comparison at u = 50.1: prediction count vs target count
uj = 50.1
ny, nf = int((y >= uj).sum()), int((f >= uj).sum())
ax.annotate("", xy=(uj, max(nf, 1)), xytext=(uj, ny),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(110, 2500, f"vertical arrow, at u = {uj:g} mm:\n{ny} target px vs {nf} predicted px",
        fontsize=8, color=INK, ha="right", va="center")
# horizontal comparison at the same target count n
n = ny
ax.annotate("", xy=(f[n - 1], n), xytext=(y[n - 1], n),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(110, 400, f"horizontal arrow, at n = {n} px:\n{y[n-1]:.1f} mm (target) vs {f[n-1]:.1f} mm",
        fontsize=8, color=INK, ha="right", va="center")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT}/survival.pdf")
print(f"survival: patch 979, u={uj}: target n={ny}, pred n={nf}; at rank n "
      f"target {y[n-1]:.2f} mm, pred {f[n-1]:.2f} mm; maxima {y[0]:.1f} / {f[0]:.1f}")

# ---- 2. gradient share by intensity bin -------------------------------------------
txt = open("logs/reward_threshold_share_v4adv.txt").read()
blocks = re.split(r"####### ", txt)[1:]
bins = ["<1", "1–10", "10–20", "20–50", "≥50"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
for ax, blk in zip(axes, blocks):
    name = blk.split()[0]
    def share(tag):
        m = re.search(re.escape(tag) + r".*?share by pixel intensity:(.*)", blk, re.S)
        return [float(v) for v in re.findall(r"\) ([0-9.]+)%", m.group(1).splitlines()[0])]
    old, new = share("=== threshold:"), share("=== quantile_mass (alpha=1):")
    x = np.arange(len(bins)); w = 0.38
    ax.bar(x - w / 2 - 0.01, old, w, color=C_TGT, label="original (threshold form)")
    ax.bar(x + w / 2 + 0.01, new, w, color=C_PRED, label="quantile area + mass weights")
    ax.set_xticks(x, bins); ax.set_xlabel("pixel intensity [mm]")
    ax.set_title("heavy subset" if name == "evalext" else "random subset", fontsize=9, color=INK)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
axes[0].set_ylabel("share of |∂R/∂z| [%]")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/share.pdf")
print("share: written")


# ---- 3. FM loss by target intensity --------------------------------------------------
C3, C4 = "#1baf7a", "#eda100"                # categorical slots 3 and 4
txt = open("logs/fm_loss_intensity_share_v4adv.txt").read()
blocks = re.split(r"####### ", txt)[1:]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, blk in zip(axes, blocks):
    name = blk.split()[0]
    rows = [l.split() for l in blk.splitlines() if re.match(r"\s*\[\d", l)]
    px, loss, dv = (np.array([float(r[k]) for r in rows]) for k in (1, 2, 3))
    x = np.arange(len(rows)); w = 0.27
    ax.bar(x - w - 0.02, px, w, color=MUTED, label="pixels")
    ax.bar(x, loss, w, color=C_TGT, label="loss value")
    ax.bar(x + w + 0.02, dv, w, color=C_PRED, label="|∂L/∂v| (gradient)")
    ax.set_yscale("log"); ax.set_ylim(3e-4, 150)
    ax.set_xticks(x, bins); ax.set_xlabel("target intensity [mm]")
    ax.set_title("heavy subset" if name == "evalext" else "random subset", fontsize=9, color=INK)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
axes[0].set_ylabel("share [%] (log scale)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
           loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(f"{OUT}/fmloss.pdf")

# ---- 4. tail signal / noise of the reward features ----------------------------------
txt = open("logs/tail_reading_snr_v4adv.txt").read()
blocks = re.split(r"####### ", txt)[1:]
CHS = [("count (threshold area)", C_TGT, "o", "-"), ("rank (quantile area)", C_PRED, "s", "--"),
       ("perimeter", C3, "^", "-."), ("Euler", C4, "D", ":")]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
for col, blk in enumerate(blocks):
    name = blk.split()[0]
    rows = [l for l in blk.splitlines() if re.match(r"\s*[0-9.]+\s+\d+\s+[0-9.]+\s+\|", l)]
    uu = np.array([float(l.split()[0]) for l in rows])
    reach = np.array([float(l.split()[2]) for l in rows])
    cells = [l.split("|")[1:] for l in rows]
    ax = axes[0, col]
    for k, (lab, c, mk, ls) in enumerate(CHS):
        bs = np.array([float(cc[k].split()[0]) for cc in cells])
        ax.plot(uu, bs, color=c, marker=mk, ms=4, lw=1.5, ls=ls, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(-4.2, 2.0)
    ax.set_title("heavy subset" if name == "evalext" else "random subset", fontsize=9, color=INK)
    ax.grid(color=MUTED, alpha=0.25, lw=0.5)
    axb = axes[1, col]
    axb.bar(uu, reach, width=uu * 0.25, color=MUTED)
    axb.set_ylim(0, 100); axb.set_xlabel("threshold u [mm]")
    axb.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
axes[0, 0].set_ylabel("bias / member spread")
axes[1, 0].set_ylabel("members with\nmax < u [%]")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT}/tailsnr.pdf")
print("fmloss.pdf, tailsnr.pdf written")
