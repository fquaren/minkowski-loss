"""FM + Minkowski (energy_qm) against vanilla FM, full arm s42, valext: same figures as aniso_panels.py.
Rows: target, backbone mean mu, FM clean (no fine-tune), FM control (fine-tune, reward weight 0), FM energy_qm.
All FM models share the backbone and the noise (same seed per batch), so the images are paired draws.
Spectra: mean of per-patch normalised spectra over 1024 patches x M members.
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
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.evaluation.tail_metrics import directional_anisotropy

W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
BB = f"{W}/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth"
FMS = {"FM clean": f"{W}/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth",
       "FM control": f"{W}/runs/sr_fm_minkowski/FMenergyctl_v4adv_20261030_141741/fm_mink_latest.pth",
       "FM energy": f"{W}/runs/sr_fm_minkowski/FMenergyqm_v4adv_20261031_022957/fm_mink_latest.pth"}
M = 2
OUT = f"{ROOT}/logs/tab/aniso_panels_fm"
os.makedirs(OUT, exist_ok=True)

cfg = load_config(f"{W}/configs/config_v4_adv_fm_evalext.yaml")
assert not cfg.get("FM_USE_RESIDUAL_MAP", False)
sigma = float(cfg["FM_RESIDUAL_SCALE"])
print(f"config FM_RESIDUAL_SCALE={sigma}")
dev = torch.device("cuda")
sv = load_scaler_val(cfg)
mx = torch.tensor(sv, device=dev, dtype=torch.float32)
d = json.load(open(cfg["DEM_STATS"]))
ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["VAL_METADATA_FILE"], (d["dem_mean"], d["dem_std"]), sv,
                            split="val", topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
bb = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
bb.load_state_dict(torch.load(BB, map_location=dev)["model_state_dict"]); bb.eval()
cond_on_mean = cfg.get("FM_CONDITION_ON_MEAN", True)
fms = {}
for k, p in FMS.items():
    f = FlowMatching(in_channels=1, c_in_condition=3 if cond_on_mean else 2, time_scale=cfg.get("FM_TIME_SCALE", 1000.0),
                     device=str(dev)).to(dev)   # this worktree's FlowMatching has no prior argument
    f.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); f.eval(); fms[k] = f
steps, sampler = cfg.get("FM_SAMPLE_STEPS", 16), cfg.get("FM_SAMPLER", "heun")
decode = lambda y: torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * mx))

F = {"target": [], "backbone mu": [], **{k: [] for k in fms}}      # FM entries: [B, M, H, W]
with torch.no_grad():
    for bi, (X, Y, _) in enumerate(DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)):
        X = X.to(dev)
        mu = bb(X)
        cond = torch.cat([X, mu[:, 0:1]], 1) if cond_on_mean else X
        F["target"].append(decode(Y[:, 0:1].to(dev))[:, 0].cpu().numpy())
        F["backbone mu"].append(decode(mu[:, 0:1])[:, 0].cpu().numpy())
        for k, f in fms.items():
            mem = []
            for m in range(M):
                torch.manual_seed(1000 * bi + m)          # same noise for every model
                r = f.sample(cond, n_steps=steps, method=sampler, in_channels=1)
                mem.append(decode(mu[:, 0:1] + sigma * r)[:, 0].cpu().numpy())
            F[k].append(np.stack(mem, 1))
        print("batch", bi, flush=True)
F = {k: np.concatenate(v) for k, v in F.items()}
N, H, Wd = F["target"].shape
flat = lambda k: F[k] if F[k].ndim == 3 else F[k].reshape(-1, H, Wd)     # members as extra patches
first = lambda k: F[k] if F[k].ndim == 3 else F[k][:, 0]

an = {k: np.array([directional_anisotropy(x) for x in flat(k)]) for k in F}
for k, a in an.items():
    print(f"per-patch anisotropy {k:12s}: mean {np.nanmean(a):.2f}  median {np.nanmedian(a):.2f}  nan {np.isnan(a).mean():.0%}")

win = np.hanning(H)[:, None] * np.hanning(Wd)[None, :]
power = lambda f: np.abs(np.fft.fftshift(np.fft.fft2(f * win, axes=(-2, -1)), axes=(-2, -1))) ** 2
ky = np.fft.fftshift(np.fft.fftfreq(H))[:, None] * np.ones((1, Wd))
kx = np.fft.fftshift(np.fft.fftfreq(Wd))[None, :] * np.ones((H, 1))
kk, ang = np.hypot(kx, ky), np.degrees(np.arctan2(ky, kx)) % 180
band = (kk >= 0.25 * kk.max()) & (np.abs(kx) > 0) & (np.abs(ky) > 0)
P = {}
for k in F:
    v = flat(k); acc = np.zeros((H, Wd))
    for i in range(0, len(v), 64):
        p = power(v[i:i + 64]); acc += (p / np.maximum(p.sum((1, 2), keepdims=True), 1e-30)).sum(0)
    P[k] = acc / len(v)
# radial spectrum (normalised, mean over patches) for the amplitude picture
kr = np.round(kk * 128).astype(int)
radial = lambda Pk: np.bincount(kr.ravel(), Pk.ravel()) / np.bincount(kr.ravel())
bins = np.arange(0, 181, 7.5); c = 0.5 * (bins[1:] + bins[:-1])
angular = lambda Pk: np.array([Pk[band & (ang >= a) & (ang < b)].mean() for a, b in zip(bins[:-1], bins[1:])])

ks = list(fms)
fig, ax = plt.subplots(1, 5, figsize=(26, 4.8))
for j, k in enumerate(ks):
    r = np.log10(P[k] / P["target"])
    im = ax[j].imshow(r, extent=[kx.min(), kx.max(), ky.max(), ky.min()], cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
    ax[j].set_title(f"log10 power {k} / target"); ax[j].set_xlabel("kx"); plt.colorbar(im, ax=ax[j])
for k in F:
    ax[3].semilogy(c, angular(P[k]), "o-", ms=3, label=k)
    rr = radial(P[k]); ax[4].loglog(np.arange(1, 65) / 128, rr[1:65], label=k)
for a in (45, 135):
    ax[3].axvline(a, c="grey", lw=0.5, ls="--")
ax[3].set_xlabel("wavevector angle (deg); 0/90 axes, 45/135 diagonals"); ax[3].set_title("angular profile, |k| >= 0.25 kmax")
ax[4].set_xlabel("|k| (cycles/px)"); ax[4].set_title("radial spectrum (normalised)"); ax[3].legend(fontsize=8); ax[4].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/spectra.png", dpi=105)

tmax = F["target"].max((1, 2)); order = np.argsort(tmax)
pick = [order[np.searchsorted(tmax[order], q)] for q in (55, 80, 120)] + [order[-1]]
cols = ["target", "backbone mu"] + ks
fig, ax = plt.subplots(len(pick), len(cols) + 3, figsize=(4 * (len(cols) + 3), 3.8 * len(pick)))
for r, i in enumerate(pick):
    t = F["target"][i]; norm = LogNorm(vmin=0.1, vmax=max(t.max(), 1.0))
    y0, x0 = np.unravel_index(t.argmax(), t.shape)
    y0, x0 = np.clip(y0 - 16, 0, H - 32), np.clip(x0 - 16, 0, Wd - 32)
    for j, k in enumerate(cols):
        f = first(k)[i]
        ax[r, j].imshow(np.maximum(f, 0.1), norm=norm, cmap="viridis")
        ax[r, j].add_patch(plt.Rectangle((x0, y0), 32, 32, fill=False, ec="r", lw=0.8))
        ax[r, j].set_title(f"{k}  max {f.max():.1f}  aniso {directional_anisotropy(f):.2f}", fontsize=9)
    for j, k in enumerate(("target", "FM control", "FM energy"), start=len(cols)):
        ax[r, j].imshow(np.maximum(first(k)[i][y0:y0 + 32, x0:x0 + 32], 0.1), norm=norm, cmap="viridis",
                        interpolation="nearest")
        ax[r, j].set_title(f"{k} zoom (red box)", fontsize=9)
    ax[r, 0].set_ylabel(f"valext #{i}  target max {t.max():.0f}")
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/samples.png", dpi=85)
np.savez(f"{OUT}/aniso_per_patch.npz", **{k.replace(" ", "_"): v for k, v in an.items()}, tmax=tmax, pick=np.array(pick))
print("picked", pick, "->", OUT)
