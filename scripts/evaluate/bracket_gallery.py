#!/usr/bin/env python
"""Side-by-side predictions of several backbones on the same validation patches (bracket review).

Draws N patches at random (seed 0) from the validation list among those with a corrected tile max
>= --min_max mm, runs every checkpoint on them, and writes one row per patch: input (bicubic-
interpolated coarse field), each model, target; plus the whole-patch maxima and the per-patch
directional anisotropy (axis/diagonal spectral power) so the images and the metric can be read together.

    python scripts/evaluate/bracket_gallery.py configs/config_v4_adv.yaml \
        --models vanilla=<ckpt> mink_6e-05=<ckpt> ... --n 12 --min_max 20.2 \
        --out notes/data_analysis_v4/bracket_gallery.pdf
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_config, load_scaler_val  # noqa: E402
from src.data.datasets import DeterministicSRDataset  # noqa: E402
from src.models.unet import LogSpaceResidualUNet  # noqa: E402
from src.evaluation.tail_metrics import directional_anisotropy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+", required=True, help="label=checkpoint")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min_max", type=float, default=20.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device("cuda")
    sv = load_scaler_val(cfg)
    mv = torch.tensor(sv, device=dev, dtype=torch.float32)
    d = json.load(open(cfg["DEM_STATS"]))
    ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["VAL_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), sv, split="val",
                                topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
    ds.is_train = False
    mx = np.array([m[3] for m in ds.metadata])
    pool = np.nonzero(mx >= a.min_max)[0]
    idx = np.sort(np.random.default_rng(a.seed).choice(pool, size=min(a.n, len(pool)), replace=False))
    X, Y, _ = zip(*[ds[int(i)] for i in idx])
    X, Y = torch.stack(X).to(dev), torch.stack(Y).to(dev)
    dec = lambda t: torch.relu(torch.expm1(torch.clamp(t * mv, max=7.0)))  # noqa: E731
    tgt = dec(Y[:, 0:1])[:, 0].cpu().numpy()
    inp = dec(X[:, 0:1])[:, 0].cpu().numpy()
    preds = {}
    for spec in a.models:
        lab, ck = spec.split("=", 1)
        m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
        m.eval()
        with torch.no_grad():
            preds[lab] = dec(m(X)[:, 0:1])[:, 0].cpu().numpy()
    cols = [("input (interpolated)", inp)] + list(preds.items()) + [("target", tgt)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    per_page = 3
    with PdfPages(a.out) as pdf:
        for p0 in range(0, len(idx), per_page):
            rows = range(p0, min(p0 + per_page, len(idx)))
            fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.9 * len(rows)), squeeze=False)
            for ri, k in enumerate(rows):
                vmax = max(float(tgt[k].max()), 1.0)
                norm = LogNorm(0.1, vmax)
                ts, r, c = ds.metadata[int(idx[k])][:3]
                for ci, (lab, arr) in enumerate(cols):
                    ax = axs[ri, ci]
                    f = arr[k]
                    ax.imshow(np.where(f >= 0.1, f, np.nan), cmap="Blues", norm=norm, origin="lower",
                              interpolation="nearest")
                    ax.set_facecolor("#f4f4f4")
                    an = directional_anisotropy(f) if lab != "input (interpolated)" else float("nan")
                    ax.set_title(f"{lab}\nmax {f.max():.0f} mm, aniso {an:.2f}", fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
                axs[ri, 0].set_ylabel(f"{ts[:10]} ({r},{c})", fontsize=7)
            fig.suptitle(f"Validation patches with corrected max >= {a.min_max} mm (random, seed {a.seed}); "
                         "colour scale per row up to the target max (mm in 1 h)", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110); plt.close(fig)
    print("->", a.out, "| patches", [int(i) for i in idx])


if __name__ == "__main__":
    main()
