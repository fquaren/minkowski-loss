"""Figure for the isolated-high-value count (scripts/dataset_v2/isolated_v4.py report):
share of tiles whose max is isolated, by tile max and set; and wet fraction around the max."""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = pd.read_csv("/home/fquareng/work/data/extremes/OPERA/quality_v4/calib/isolated/all.csv.gz",
                low_memory=False)
out = sys.argv[1]
bins = [31, 53, 89, 150, 500.01]
labels = ["31-53", "53-89", "89-150", "150-500"]
k = x[~x.rej_v4 & (x["max"] <= 500)].copy()
k["b"] = pd.cut(k["max"], bins, right=False, labels=labels)
sets = [("random", "random days", "tab:gray"), ("event", "event days", "tab:blue"),
        ("ceiling_4862", "48.62 days (Nov 2019)", "tab:red"),
        ("valjevo_ceiling", "Valjevo days (2023)", "tab:orange"), ("madrid", "Madrid days", "tab:purple")]

fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
w = 0.16
for i, (s, lab, c) in enumerate(sets):
    g = k[k.set == s].groupby("b", observed=False)
    p, n = g.iso_at_max.mean(), g.size()
    xs = np.arange(len(labels)) + (i - 2) * w
    ax[0].bar(xs, p.values, w, color=c, label=lab)
    for xx, pp, nn in zip(xs, p.values, n.values):
        if nn:
            ax[0].text(xx, (pp if np.isfinite(pp) else 0) + 0.01, f"{nn}", ha="center", fontsize=6, rotation=90)
ax[0].set_xticks(range(len(labels)), labels)
ax[0].set_xlabel("tile max (mm/h)")
ax[0].set_ylabel("share of kept tiles whose max is isolated")
ax[0].set_title("(a) max pixel with 5x5 median < 0.1 mm/h (n above bars)", fontsize=9)
ax[0].legend(fontsize=7)

for s, lab, c in sets[:3]:
    v = k[(k.set == s) & (k["max"] >= 89)].wet11_at_max
    if len(v):
        ax[1].hist(v, bins=np.linspace(0, 1, 26), density=True, histtype="step", color=c,
                   lw=1.5, label=f"{lab} (n={len(v)})")
ax[1].set_xlabel("wet fraction (>= 0.1 mm/h) in the 11x11 window (22 km) around the max")
ax[1].set_ylabel("density")
ax[1].set_title("(b) kept tiles with max >= 89 mm/h", fontsize=9)
ax[1].legend(fontsize=7)
fig.savefig(out, dpi=130)
print("wrote", out)
