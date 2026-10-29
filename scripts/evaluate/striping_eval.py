#!/usr/bin/env python
"""Striping measures for several backbones on the same validation patches (DECISIONS §37, §38).

Per model and patch: axis and diagonal edge-orientation excess against the target (`src/evaluation/striping.py`)
and the paired anisotropy log(aniso_pred / aniso_target). Summaries for all patches and for the heavy subset.

    python scripts/evaluate/striping_eval.py configs/config_v4_adv.yaml --models vanilla=<ckpt> ... \
        --n 1024 --min_max 1.0 --heavy 20.2 --out notes/data_analysis_v4/striping_bracket.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_config, load_scaler_val  # noqa: E402
from src.data.datasets import DeterministicSRDataset  # noqa: E402
from src.models.unet import LogSpaceResidualUNet  # noqa: E402
from src.evaluation.striping import striping_excess, highpass_orientation_shares, spectral_line_excess  # noqa: E402
from src.evaluation.tail_metrics import directional_anisotropy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+", required=True, help="label=checkpoint")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--min_max", type=float, default=1.0)
    ap.add_argument("--heavy", type=float, default=20.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
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
    idx = np.sort(np.random.default_rng(a.seed).choice(np.nonzero(mx >= a.min_max)[0], a.n, replace=False))
    dec = lambda t: torch.relu(torch.expm1(torch.clamp(t * mv, max=7.0)))  # noqa: E731
    X = torch.stack([ds[int(i)][0] for i in idx])
    T = torch.stack([ds[int(i)][1] for i in idx])
    tgt = np.concatenate([dec(T[s:s + a.batch, 0:1].to(dev))[:, 0].cpu().numpy() for s in range(0, len(idx), a.batch)])
    an_t = np.array([directional_anisotropy(f) for f in tgt])
    hp_t = np.array([highpass_orientation_shares(f)[:2] for f in tgt])
    sp_t = np.array([spectral_line_excess(f) for f in tgt])
    rows = []
    for spec in a.models:
        lab, ck = spec.split("=", 1)
        m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
        m.eval()
        with torch.no_grad():
            pr = np.concatenate([dec(m(X[s:s + a.batch].to(dev))[:, 0:1])[:, 0].cpu().numpy()
                                 for s in range(0, len(idx), a.batch)])
        ex = striping_excess(pr, tgt)
        an_p = np.array([directional_anisotropy(f) for f in pr])
        hp_p = np.array([highpass_orientation_shares(f)[:2] for f in pr])
        sp_p = np.array([spectral_line_excess(f) for f in pr])
        for k, i in enumerate(idx):
            rows.append({"model": lab, "row": int(i), "tile_max": mx[i], "axis_excess": ex[k, 0],
                         "diag_excess": ex[k, 1], "log_aniso_ratio": np.log(an_p[k] / an_t[k]),
                         "hp_axis_excess": hp_p[k, 0] - hp_t[k, 0], "hp_diag_excess": hp_p[k, 1] - hp_t[k, 1],
                         "spec_axis_excess": sp_p[k, 0] - sp_t[k, 0], "spec_diag_excess": sp_p[k, 1] - sp_t[k, 1]})
        print(lab, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    def summ(x):
        return pd.Series({"n": x.axis_excess.notna().sum(),
                          "axis_excess_median": x.axis_excess.median(), "axis_excess_mean": x.axis_excess.mean(),
                          "share_axis_gt_0.05": (x.axis_excess > 0.05).mean(),
                          "diag_excess_median": x.diag_excess.median(), "diag_excess_mean": x.diag_excess.mean(),
                          "share_diag_gt_0.05": (x.diag_excess > 0.05).mean(),
                          "log_aniso_ratio_median": x.log_aniso_ratio.median(),
                          "hp_axis_median": x.hp_axis_excess.median(), "hp_axis_mean": x.hp_axis_excess.mean(),
                          "hp_diag_median": x.hp_diag_excess.median(), "hp_diag_mean": x.hp_diag_excess.mean(),
                          "spec_axis_median": x.spec_axis_excess.median(), "spec_diag_median": x.spec_diag_excess.median()})
    order = [s.split("=")[0] for s in a.models]
    for name, sub in (("all patches", t), (f"tile max >= {a.heavy} mm", t[t.tile_max >= a.heavy])):
        s = sub.groupby("model").apply(summ).loc[order]
        print(f"\n== {name}\n" + s.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
