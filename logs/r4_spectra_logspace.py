"""RAPSD of R4 backbones in the log-normalised space the spectral loss acts on, and in mm/h.

Two patch sets from val: a random 16,384 (seed 0) and every patch with target max >= 50.1 mm/h.
Indices come from the saved eval arrays (shuffle=False, so array row = dataset index).
"""
import json, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from src.utils import load_config, load_scaler_val
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.evaluation.training_metrics import compute_radial_power_spectrum

W = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion"
R = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c/runs/sr_analytical"
RUNS = {"mse_full": "UNet_Ana_20261105_150530", "mink_full": "UNet_Ana_20261105_150601",
        "mink_cut": "UNet_Ana_20261105_150429", "spectral_full_0.15": "UNet_Ana_20261108_080705",
        "spectral_cut_0.15": "UNet_Ana_20261107_122302", "spectral_full_0.05": "UNet_Ana_20261110_013608"}
cfg = load_config("/work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_eval.yaml")
dev = torch.device("cuda")
sv = load_scaler_val(cfg); mv = torch.tensor(sv, device=dev, dtype=torch.float32)
dem = json.load(open(cfg["DEM_STATS"]))
ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["VAL_METADATA_FILE"],
                            (float(dem["dem_mean"]), float(dem["dem_std"])), sv, split="val",
                            topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
tmax = np.load(f"{W}/eval_results/backbone/v4adv_val/r4_mse_full_s42_v100/backbone_arrays.npz")["tmax"]
assert len(tmax) == len(ds), (len(tmax), len(ds))
sets = {"random": np.sort(np.random.default_rng(0).choice(len(ds), 16384, replace=False)),
        "extreme": np.flatnonzero(tmax >= 50.1)}
print({k: len(v) for k, v in sets.items()}, flush=True)
out = {}
for name, run in [("target", None)] + list(RUNS.items()):
    m = None
    if run:
        m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
        m.load_state_dict(torch.load(f"{R}/{run}/unet_latest.pth", map_location=dev, weights_only=False)["model_state_dict"])
        m.eval()
    for s, idx in sets.items():
        acc = {"norm": 0.0, "phys": 0.0}; n = 0
        dl = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=2)
        with torch.no_grad():
            for X, Y, _ in dl:
                X, Y = X.to(dev), Y.to(dev)
                y = Y[:, 0:1] if m is None else m(X)[:, 0:1]
                phys = torch.relu(torch.expm1(torch.clamp(y * mv, max=7.0)))   # trainer's conversion
                acc["norm"] = acc["norm"] + compute_radial_power_spectrum(y).sum(0)
                acc["phys"] = acc["phys"] + compute_radial_power_spectrum(phys).sum(0)
                n += len(X)
        for sp in acc:
            out[f"{name}|{s}|{sp}"] = (acc[sp] / n).cpu().numpy()
    print("done", name, flush=True)
np.savez(f"{W}/eval_results/spectra/r4_rapsd_norm_vs_phys.npz", **out)
print("=== SPECTRA_DONE", flush=True)
