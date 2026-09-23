#!/usr/bin/env python
"""Dump DEM / input / target / per-model prediction fields for a handful of patches.

The quantitative tables say how much a structural loss changes the tail; they do not show
what the change looks like. Two pathologies in particular are invisible to every headline
metric and obvious in a field panel: the grid-aligned filaments a Minkowski-trained model
produces at too high a weight, and the peak overshoot that accompanies them.

This script evaluates several models on the *same* small set of patches and writes one npz
that ``scripts/evaluate/make_plots.py --field_bundle`` renders. Only the selected patches are
run, so it costs seconds rather than the hours a full evaluation pass takes.

Patch selection, in order of precedence:

  --indices 12 3400 ...   explicit dataset indices
  --pct 99.9 99 95 ...    one patch per percentile of the target-maximum distribution
  --n_extreme / --n_mid   the N largest target maxima, plus N from the middle

The ranking statistic is read from a previous run's ``tmax`` array via ``--rank_from``, which
both eval scripts write in dataset order; without it the split is scanned, which is slower and
gives the same answer.

Prefer ``--pct`` over ``--n_extreme``. The patch pool contains bad radar observations as well
as precipitation, and artefacts concentrate at the very top of the intensity distribution
because a corrupted return is usually more extreme than real rain -- the first selection made
here took the three largest maxima in the test split and all three were artefacts. Spreading
the selection over percentiles keeps the heaviest patches in view while also showing ordinary
ones, so an artefact reads as the outlier it is instead of standing in for the whole tail.
Look at the images before drawing conclusions from the extreme end either way.

Model specs reuse the pipe-separated shape of ``scripts/hpc/run_full_eval.sh``:

    --model "label|bicubic||"
    --model "label|backbone|<unet.pth>|"
    --model "label|fm|<fm.pth>|<backbone.pth>"
    --model "label|fm|<fm.pth>|<backbone.pth>|<config.yaml>"

The optional fifth field is a per-model config, and it exists because the flow-matching
models pin different ``FM_RESIDUAL_SCALE`` values: ``residual_stats.json`` holds whichever
backbone ``compute_residual_stats.py`` last ran against, so a model that falls back to it is
silently mis-scaled. Only the flow-matching settings are taken from that config -- the
dataset, the split and the thresholds always come from the positional config, so every panel
in a figure shows the same patches, which is the entire point of the comparison.

Usage:
  python scripts/evaluate/dump_fields.py config.yaml \
      --model "MSE|backbone|runs/sr_analytical/<run>/unet_best.pth|" \
      --model "+ Minkowski|backbone|runs/sr_analytical/<run>/unet_best.pth|" \
      --n_extreme 4 --n_mid 2 --output eval_results/fields/study1.npz
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.utils import load_config, load_scaler_val, load_physical_thresholds, \
    load_persistence_thresholds
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss

_META = {"test": "TEST_METADATA_FILE", "validation": "VAL_METADATA_FILE",
         "val": "VAL_METADATA_FILE", "train": "TRAIN_METADATA_FILE"}


def _load_sigma(config, device):
    """Resolve sigma_r exactly as eval_extremes.py does (config -> map -> json).

    Kept identical on purpose: a silent mismatch between the residual scale used here and the
    one the flow-matching model was trained with produces fields that look like a
    catastrophic model failure and are in fact a scaling bug.
    """
    if config.get("FM_USE_RESIDUAL_MAP", False):
        path = config.get("FM_RESIDUAL_STD_PATH",
                          os.path.join(config["PREPROCESSED_DATA_DIR"], "residual_std.npy"))
        t = torch.from_numpy(np.load(path).astype(np.float32)).to(device).clamp_min(1e-3)
        return t.unsqueeze(0), f"per-pixel map {path}"
    if "FM_RESIDUAL_SCALE" in config and config["FM_RESIDUAL_SCALE"] is not None:
        v = float(config["FM_RESIDUAL_SCALE"])
        return v, f"config FM_RESIDUAL_SCALE={v:.4g}"
    sp = os.path.join(config["PREPROCESSED_DATA_DIR"], "residual_stats.json")
    if os.path.exists(sp):
        with open(sp) as f:
            v = float(json.load(f)["sigma_r_scalar"])
        return v, f"residual_stats.json sigma_r_scalar={v:.4g}"
    return 1.0, "DEFAULT 1.0"


def _parse_spec(entry):
    parts = [p.strip() for p in entry.split("|")]
    if len(parts) == 4:
        parts.append("")
    if len(parts) != 5:
        raise SystemExit(
            f"--model expects 'label|kind|ckpt|backbone[|config]', got {entry!r}")
    label, kind, ckpt, bb, cfg = parts
    if kind not in ("bicubic", "backbone", "fm"):
        raise SystemExit(f"unknown model kind {kind!r} in {entry!r}")
    return label, kind, ckpt or None, bb or None, cfg or None


def _select_indices(args, ds, decode_max):
    """Dataset indices to render, by explicit list, by percentile spread, or by rank."""
    if args.indices:
        return np.asarray(args.indices, dtype=int)

    if args.rank_from:
        tmax = np.load(args.rank_from, allow_pickle=False)["tmax"]
        if tmax.shape[0] != len(ds):
            raise SystemExit(
                f"{args.rank_from} holds {tmax.shape[0]} samples but the {args.split} split "
                f"has {len(ds)}; the arrays are in dataset order only for the split they "
                f"were produced from")
    else:
        print(f"[dump] scanning {len(ds)} patches for target maxima "
              f"(pass --rank_from to reuse a previous run's tmax)")
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4,
                            pin_memory=False, multiprocessing_context="spawn")
        chunks = []
        with torch.no_grad():
            for _, Y, _ in tqdm(loader, desc="scan"):
                chunks.append(decode_max(Y[:, 0:1]).numpy())
        tmax = np.concatenate(chunks)

    order = np.argsort(tmax)[::-1]

    if args.pct:
        # One patch per requested percentile of the target maximum. Percentiles are taken
        # descending so the heaviest patch is rendered first, and duplicates are dropped
        # (close percentiles can land on the same patch in a small split).
        picked, seen = [], set()
        for q in sorted(args.pct, reverse=True):
            rank = int(round((1.0 - q / 100.0) * (len(order) - 1)))
            rank = min(max(rank, 0), len(order) - 1)
            idx = int(order[rank])
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            print(f"[dump]   p{q:<6g} -> patch {idx:>7d}  target max {tmax[idx]:7.1f} mm/h")
        return np.asarray(picked, dtype=int)

    picked = list(order[:args.n_extreme])
    if args.n_mid > 0:
        mid = order[len(order) // 2: len(order) // 2 + args.n_mid]
        picked += list(mid)
    return np.asarray(picked, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model", action="append", default=[],
                    help="'label|kind|ckpt|backbone[|config]', repeatable; order sets "
                         "panel order. The optional config supplies the flow-matching "
                         "settings (notably FM_RESIDUAL_SCALE) for that model only")
    ap.add_argument("--split", default="test", choices=list(_META))
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit dataset indices; overrides the selection heuristics")
    ap.add_argument("--rank_from", default=None,
                    help="an *_arrays.npz from a previous run on this split; its tmax array "
                         "is reused to rank patches instead of rescanning the data")
    ap.add_argument("--pct", type=float, nargs="+", default=None,
                    help="percentiles of the target-maximum distribution, one patch each "
                         "(e.g. 99.9 99 95 90 75 50). Preferred over --n_extreme: the very "
                         "top of the distribution is artefact-dominated")
    ap.add_argument("--n_extreme", type=int, default=4)
    ap.add_argument("--n_mid", type=int, default=2)
    ap.add_argument("--ensemble", type=int, default=1,
                    help="flow-matching members; the panel shows member 0, which is the "
                         "right perceptual reference (the ensemble mean is blurry)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="eval_results/fields/fields.npz")
    args = ap.parse_args()

    if not args.model:
        raise SystemExit("at least one --model is required")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    max_val_cpu = torch.tensor(scaler_val, dtype=torch.float32)
    pixel_km = float(config.get("PIXEL_SIZE_KM", 2.0))
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    u_phys = load_physical_thresholds(config)

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    dem_mean, dem_std = float(d["dem_mean"]), float(d["dem_std"])

    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config[_META[args.split]],
                                (dem_mean, dem_std), scaler_val, split=args.split,
                                topology_mode=topo_mode)

    def decode(y, mv):
        return torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * mv))

    idx = _select_indices(args, ds,
                          lambda y: decode(y, max_val_cpu).amax(dim=(1, 2, 3)))
    print(f"[dump] patches: {list(map(int, idx))}")

    sub = Subset(ds, [int(i) for i in idx])
    loader = DataLoader(sub, batch_size=min(len(sub), 16), shuffle=False, num_workers=0)

    thresh_b0 = load_persistence_thresholds(config)[0] if topo_mode == "b0" else 0.0
    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys, pixel_size_km=pixel_km, topology_mode=topo_mode,
        area_mode="ste", persistence_thresh_b0=thresh_b0,
    ).to(device)

    # ---- the shared fields, read once ------------------------------------------------
    Xs, Ys, Gs = [], [], []
    for X, Y, Ygamma in loader:
        Xs.append(X); Ys.append(Y); Gs.append(Ygamma)
    X = torch.cat(Xs).to(device); Y = torch.cat(Ys).to(device)
    Ygamma = torch.cat(Gs).to(device)

    target = decode(Y[:, 0:1], max_val)
    lr_field = decode(X[:, 0:1], max_val)                      # channel 0: interpolated input
    dem = X[:, 1].cpu().numpy() * dem_std + dem_mean           # channel 1: z-scored DEM

    labels, preds, gammas = [], [], []
    with torch.no_grad():
        for entry in args.model:
            label, kind, ckpt, bb_path, cfg_path = _parse_spec(entry)
            if kind == "bicubic":
                pred = lr_field
                print(f"[dump] {label}: bicubic (the interpolated input)")
            else:
                bpath = ckpt if kind == "backbone" else (bb_path or config.get("BACKBONE_CHECKPOINT"))
                if bpath is None:
                    raise SystemExit(f"{label}: a backbone checkpoint is required")
                backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
                backbone.load_state_dict(
                    torch.load(bpath, map_location=device)["model_state_dict"])
                backbone.eval()
                if kind == "backbone":
                    pred = decode(backbone(X)[:, 0:1], max_val)
                    print(f"[dump] {label}: backbone {bpath}")
                else:
                    # Flow-matching settings come from the per-model config when one is
                    # given; the data already read above stays on the positional config.
                    mcfg = load_config(cfg_path) if cfg_path else config
                    if cfg_path:
                        print(f"[dump] {label}: flow-matching config {cfg_path}")
                    sigma, sigma_src = _load_sigma(mcfg, device)
                    print(f"[dump] {label}: residual scale sigma_r: {sigma_src}")
                    if "residual_stats.json" in sigma_src or "DEFAULT" in sigma_src:
                        print(f"    !! WARNING: {label} did not get sigma_r from a config. "
                              f"residual_stats.json holds the Minkowski-backbone value, so a "
                              f"vanilla-backbone run is mis-scaled; DEFAULT 1.0 is worse "
                              f"still. Pin FM_RESIDUAL_SCALE and re-run.")
                    cond_on_mean = mcfg.get("FM_CONDITION_ON_MEAN", True)
                    _kw = dict(in_channels=1, c_in_condition=3 if cond_on_mean else 2,
                               time_scale=mcfg.get("FM_TIME_SCALE", 1000.0),
                               device=str(device))
                    import inspect as _inspect
                    if "prior" in _inspect.signature(FlowMatching.__init__).parameters:
                        _kw["prior"] = mcfg.get("FM_PRIOR", "gaussian")
                        _kw["prior_nu"] = mcfg.get("FM_PRIOR_NU", 5.0)
                    fm = FlowMatching(**_kw).to(device)
                    fm.load_state_dict(
                        torch.load(ckpt, map_location=device)["model_state_dict"])
                    fm.eval()
                    steps = args.steps or mcfg.get("FM_SAMPLE_STEPS", 16)
                    sampler = args.sampler or mcfg.get("FM_SAMPLER", "heun")
                    mu = backbone(X)
                    cond = torch.cat([X, mu[:, 0:1]], dim=1) if cond_on_mean else X
                    r = fm.sample(cond, n_steps=steps, method=sampler, in_channels=1)
                    pred = decode(mu[:, 0:1] + sigma * r, max_val)
                    print(f"[dump] {label}: fm {ckpt} ({sampler}, {steps} steps, member 0)")

            a, p, t = geom_fn._functionals(pred, anneal_factor=0.05)
            gammas.append(torch.stack([a, p, t], dim=1).cpu().numpy())   # [N,3,Q] physical
            preds.append(pred[:, 0].cpu().numpy())
            labels.append(label)

    np.savez_compressed(
        args.output,
        dem=dem.astype(np.float32),
        input=lr_field[:, 0].cpu().numpy().astype(np.float32),
        target=target[:, 0].cpu().numpy().astype(np.float32),
        preds=np.stack(preds).astype(np.float32),
        labels=np.array(labels),
        indices=idx.astype(np.int64),
        gamma_pred=np.stack(gammas).astype(np.float32),
        gamma_target=Ygamma.cpu().numpy().astype(np.float32),
        thresholds=np.asarray(u_phys, dtype=np.float32),
        split=np.array(args.split),
    )
    print(f"[dump] wrote {args.output}: {len(labels)} models x {len(idx)} patches")


if __name__ == "__main__":
    main()
