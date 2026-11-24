"""Why is the Minkowski U-Net's anisotropy 0.5 (MSE 2.0)? Target vs MSE s42 vs Minkowski s42, full arm, valext.
Fig 1: sample patches, a 32x32 zoom at the target maximum, the 2-D power spectrum ratio pred/target.
Fig 2: over all 1024 patches, the mean log power ratio pred/target and the angular power profile in the high-k
band that directional_anisotropy uses (|k| >= 0.25 k_max). Prints per-patch anisotropy of target and both models.
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from torch.utils.data import DataLoader

ROOT = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4"
sys.path.insert(0, ROOT)
from src.utils import load_config, load_scaler_val
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.evaluation.tail_metrics import directional_anisotropy

W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
R = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c"
CKS = {"MSE": f"{R}/runs/sr_analytical/UNet_Ana_20261105_150530/unet_latest.pth",
       "Minkowski": f"{R}/runs/sr_analytical/UNet_Ana_20261105_150601/unet_latest.pth"}
OUT = f"{ROOT}/logs/tab/aniso_panels"
os.makedirs(OUT, exist_ok=True)

cfg = load_config(f"{W}/configs/config_v4_adv_fm_evalext.yaml")
dev = torch.device("cuda")
sv = load_scaler_val(cfg)
mx = torch.tensor(sv, device=dev, dtype=torch.float32)
d = json.load(open(cfg["DEM_STATS"]))
ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["VAL_METADATA_FILE"], (d["dem_mean"], d["dem_std"]), sv,
                            split="val", topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
models = {}
for k, p in CKS.items():
    m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); m.eval(); models[k] = m
decode = lambda y: torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * mx))

# ---- full pass: fields kept on CPU (1024 x 128 x 128 x 3 floats = 200 MB)
F = {"target": [], "MSE": [], "Minkowski": []}
with torch.no_grad():
    for X, Y, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
        X = X.to(dev)
        F["target"].append(decode(Y[:, 0:1].to(dev))[:, 0].cpu().numpy())
        for k, m in models.items():
            F[k].append(decode(m(X)[:, 0:1])[:, 0].cpu().numpy())
F = {k: np.concatenate(v) for k, v in F.items()}
N, H, Wd = F["target"].shape

an = {k: np.array([directional_anisotropy(f) for f in v]) for k, v in F.items()}
for k, a in an.items():
    print(f"per-patch anisotropy {k:10s}: nanmean {np.nanmean(a):.2f}  median {np.nanmedian(a):.2f}  "
          f"nan {np.isnan(a).mean():.0%}")
wet = F["target"].max((1, 2)) >= 1.0
for k, a in an.items():
    print(f"  wet patches (target max >= 1 mm/h, n={wet.sum()}) {k:10s}: median {np.nanmedian(a[wet]):.2f}")

# ---- spectra, same window and band as directional_anisotropy
win = np.hanning(H)[:, None] * np.hanning(Wd)[None, :]
def power(f):
    return np.abs(np.fft.fftshift(np.fft.fft2(f * win, axes=(-2, -1)), axes=(-2, -1))) ** 2
ky = np.fft.fftshift(np.fft.fftfreq(H))[:, None] * np.ones((1, Wd))
kx = np.fft.fftshift(np.fft.fftfreq(Wd))[None, :] * np.ones((H, 1))
kk, ang = np.hypot(kx, ky), np.degrees(np.arctan2(ky, kx)) % 180
band = (kk >= 0.25 * kk.max()) & (np.abs(kx) > 0) & (np.abs(ky) > 0)
# per-patch normalised power (each patch's spectrum / its total), averaged: big storms do not dominate
P = {}
for k, v in F.items():
    acc = np.zeros((H, Wd))
    for i in range(0, N, 64):
        p = power(v[i:i + 64]); acc += (p / np.maximum(p.sum((1, 2), keepdims=True), 1e-30)).sum(0)
    P[k] = acc / N
bins = np.arange(0, 181, 7.5)
def angular(Pk):
    return np.array([Pk[band & (ang >= a) & (ang < b)].mean() for a, b in zip(bins[:-1], bins[1:])])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for j, k in enumerate(("MSE", "Minkowski")):
    r = np.log10(P[k] / P["target"])
    im = ax[j].imshow(r, extent=[kx.min(), kx.max(), ky.max(), ky.min()], cmap="RdBu_r",
                      norm=TwoSlopeNorm(0, vmin=min(r.min(), -0.1), vmax=max(r.max(), 0.1)))
    ax[j].set_title(f"log10 power {k} / target (mean of normalised spectra)"); ax[j].set_xlabel("kx"); ax[j].set_ylabel("ky")
    plt.colorbar(im, ax=ax[j])
c = 0.5 * (bins[1:] + bins[:-1])
for k in P:
    ax[2].semilogy(c, angular(P[k]), "o-", label=k)
for a in (0, 90, 180):
    ax[2].axvline(a, c="k", lw=0.5)
for a in (45, 135):
    ax[2].axvline(a, c="grey", lw=0.5, ls="--")
ax[2].set_xlabel("wavevector angle (deg); 0/90 = axes, 45/135 = diagonals"); ax[2].set_ylabel("mean power, |k| >= 0.25 kmax")
ax[2].legend(); ax[2].set_title("angular profile, high-k band")
fig.tight_layout(); fig.savefig(f"{OUT}/spectra.png", dpi=110)

# ---- sample panels: four event patches across intensity, two weak wet ones
tmax = F["target"].max((1, 2))
order = np.argsort(tmax)
pick = [order[np.searchsorted(tmax[order], q)] for q in (55, 80, 120)] + [order[-1]]
weak = np.flatnonzero((tmax > 5) & (tmax < 15)); pick += list(weak[:: max(len(weak) // 2, 1)][:2])
fig, ax = plt.subplots(len(pick), 6, figsize=(22, 3.6 * len(pick)))
for r, i in enumerate(pick):
    t = F["target"][i]
    vmax = max(t.max(), 1.0)
    norm = LogNorm(vmin=0.1, vmax=vmax)
    y0, x0 = np.unravel_index(t.argmax(), t.shape)
    y0, x0 = np.clip(y0 - 16, 0, H - 32), np.clip(x0 - 16, 0, Wd - 32)
    for j, k in enumerate(("target", "MSE", "Minkowski")):
        f = F[k][i]
        ax[r, j].imshow(np.maximum(f, 0.1), norm=norm, cmap="viridis")
        ax[r, j].set_title(f"{k}  max {f.max():.1f}  aniso {an[k][i]:.2f}", fontsize=9)
        ax[r, j].add_patch(plt.Rectangle((x0, y0), 32, 32, fill=False, ec="r", lw=0.8))
    for j, k in enumerate(("target", "Minkowski"), start=3):
        ax[r, j].imshow(np.maximum(F[k][i][y0:y0 + 32, x0:x0 + 32], 0.1), norm=norm, cmap="viridis",
                        interpolation="nearest")
        ax[r, j].set_title(f"{k} zoom 32x32 (red box)", fontsize=9)
    pr = np.log10(power(F["Minkowski"][i]) / np.maximum(power(t), 1e-30))
    ax[r, 5].imshow(np.clip(pr, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3)
    ax[r, 5].set_title("log10 power Mink / target (clipped ±3)", fontsize=9)
    ax[r, 0].set_ylabel(f"valext #{i}  target max {t.max():.0f}")
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/samples.png", dpi=90)
np.savez(f"{OUT}/aniso_per_patch.npz", **{k: v for k, v in an.items()}, tmax=tmax, pick=np.array(pick))
print("picked", pick, "->", OUT)
