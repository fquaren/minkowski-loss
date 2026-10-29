#!/usr/bin/env python
"""Where does the orientation of the Minkowski texture come from? (DECISIONS §38, researcher's question 2026-10-26.)

On N heavy validation patches, for each model:
  - the orientation histogram of the small-scale texture inside rain (high-pass of log(1+f), as
    `striping.highpass_orientation_shares`), pooled over patches, against the targets';
  - a rotation test: predict on the input rotated by 90 degrees (precipitation and DEM channels together), rotate
    the prediction back, and compare its texture with the prediction on the original input. A rotation-
    equivariant model gives the same texture (correlation ~1); a texture fixed in grid coordinates does not,
    and its dominant angle stays put instead of turning with the input.

    python scripts/evaluate/texture_orientation.py configs/config_v4_adv.yaml --models a=<ckpt> ... --n 64
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_config, load_scaler_val  # noqa: E402
from src.data.datasets import DeterministicSRDataset  # noqa: E402
from src.models.unet import LogSpaceResidualUNet  # noqa: E402


def texture(f, sigma=2.0):
    z = np.log1p(np.clip(f, 0, None).astype(np.float64))
    return z - ndimage.gaussian_filter(z, sigma)


def angles(f, wet=0.1, erode=3, q=0.75):
    """Gradient angles (deg, [0, 180)) of the strongest texture edges inside rain, magnitude-weighted."""
    r = texture(f)
    gx, gy = ndimage.sobel(r, axis=1), ndimage.sobel(r, axis=0)
    mag = np.hypot(gx, gy)
    m = ndimage.binary_erosion(f >= wet, iterations=erode) & (mag > 0)
    if m.sum() < 50:
        return np.empty(0), np.empty(0)
    e = m & (mag >= np.quantile(mag[m], q))
    return np.degrees(np.arctan2(gy[e], gx[e])) % 180.0, mag[e]


def hist(fields, bins=36):
    h = np.zeros(bins)
    for f in fields:
        a, w = angles(f)
        if a.size:
            h += np.histogram(a, bins=bins, range=(0, 180), weights=w)[0]
    return h / max(h.sum(), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--min_max", type=float, default=20.2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device("cuda")
    sv = load_scaler_val(cfg)
    mv = torch.tensor(sv, device=dev, dtype=torch.float32)
    d = json.load(open(cfg["DEM_STATS"]))
    ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["VAL_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), sv, split="val")
    ds.is_train = False
    mx = np.array([m[3] for m in ds.metadata])
    idx = np.sort(np.random.default_rng(1).choice(np.nonzero(mx >= a.min_max)[0], a.n, replace=False))
    X = torch.stack([ds[int(i)][0] for i in idx]).to(dev)
    T = torch.stack([ds[int(i)][1] for i in idx]).to(dev)
    dec = lambda t: torch.relu(torch.expm1(torch.clamp(t * mv, max=7.0)))  # noqa: E731
    tgt = dec(T[:, 0:1])[:, 0].cpu().numpy()
    centres = np.arange(2.5, 180, 5)
    ht = hist(tgt)
    print(f"targets: texture-angle histogram peak at {centres[ht.argmax()]:.1f} deg "
          f"(share in the peak 10 deg: {ht[ht.argmax()] + ht[(ht.argmax() + 1) % 36]:.3f}; uniform 0.056)")
    for spec in a.models:
        lab, ck = spec.split("=", 1)
        m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
        m.eval()
        with torch.no_grad():
            p0 = dec(m(X)[:, 0:1])[:, 0].cpu().numpy()
            pr = dec(m(torch.rot90(X, 1, dims=(2, 3)))[:, 0:1])[:, 0]
            p1 = torch.rot90(pr, -1, dims=(1, 2)).cpu().numpy()       # rotated back to the original frame
        h0, h1 = hist(p0), hist(p1)
        cors = []
        for f0, f1 in zip(p0, p1):
            r0, r1 = texture(f0), texture(f1)
            w = ndimage.binary_erosion(f0 >= 0.1, iterations=3)
            if w.sum() > 50:
                cors.append(np.corrcoef(r0[w], r1[w])[0, 1])
        # in the rotated frame the model's own preferred angle would appear shifted by 90 deg after rotating back
        print(f"{lab:9s} peak {centres[h0.argmax()]:5.1f} deg (peak share {h0[h0.argmax()] + h0[(h0.argmax() + 1) % 36]:.3f}) | "
              f"rotated input, rotated back: peak {centres[h1.argmax()]:5.1f} deg | texture corr. original vs "
              f"rotated-back: median {np.median(cors):.2f} | hist. L1 distance to targets {np.abs(h0 - ht).sum():.3f}")


if __name__ == "__main__":
    main()
