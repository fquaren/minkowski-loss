#!/usr/bin/env python
"""Tail-aware extreme-event evaluation: one row of the in-distribution / o.o.d. table.

Runs the same protocol for every model class so the rows are comparable:

    bicubic   -- the interpolated low-resolution input (no learning)
    backbone  -- deterministic LogSpaceResidualUNet (vanilla or Minkowski-trained)
    fm        -- flow-matching residual model on top of a frozen backbone

and reports RMSE_ext, FSS at high thresholds, RAPSD log-distance, Minkowski distance,
GPD shape-parameter error and return-level bias, plus two diagnostics that the headline
metrics are blind to: directional anisotropy (the radially averaged spectrum integrates
over direction) and the patch-peak ratio (SAL A compares domain means, not maxima).

Both diagnostics exist because a structural loss can be satisfied by grid-aligned
filamentary textures that improve SAL S, isoperimetric violation and RAPSD while badly
over-shooting peak intensity. Report them alongside any structural claim.

Usage:
  python scripts/evaluate/eval_extremes.py config.yaml --model bicubic
  python scripts/evaluate/eval_extremes.py config.yaml --model backbone \
      --checkpoint runs/sr_analytical/<run>/unet_best.pth --variant minkowski
  python scripts/evaluate/eval_extremes.py config.yaml --model fm \
      --fm_checkpoint runs/sr_flow_matching/<run>/fm_best.pth \
      --backbone runs/sr_analytical/<run>/unet_best.pth --ensemble 16
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import (
    load_config, load_scaler_val, load_physical_thresholds, load_persistence_thresholds,
)
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.evaluation.training_metrics import compute_radial_power_spectrum
from src.evaluation.extreme_metrics import sal, crps_ensemble
from src.evaluation.tail_metrics import (
    FSSAccumulator, POTAccumulator, gpd_comparison, directional_anisotropy, peak_ratio,
)

_META = {"test": "TEST_METADATA_FILE", "validation": "VAL_METADATA_FILE",
         "val": "VAL_METADATA_FILE", "train": "TRAIN_METADATA_FILE"}


def _load_sigma(config, device):
    """Resolve sigma_r exactly as the flow-matching trainer does (config -> map -> json)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model", required=True, choices=["bicubic", "backbone", "fm"])
    ap.add_argument("--checkpoint", default=None, help="backbone checkpoint (--model backbone)")
    ap.add_argument("--fm_checkpoint", default=None, help="flow-matching checkpoint (--model fm)")
    ap.add_argument("--backbone", default=None, help="frozen backbone for --model fm")
    ap.add_argument("--split", default="test", choices=list(_META))
    ap.add_argument("--ensemble", type=int, default=16, help="ensemble size (--model fm)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--extreme_pct", type=float, default=99.0)
    ap.add_argument("--pot_threshold", type=float, default=None,
                    help="POT threshold in mm/h; default = the second-highest PHYSICAL_THRESHOLD")
    ap.add_argument("--fss_thresholds", type=float, nargs="+", default=None,
                    help="mm/h; default = the top three PHYSICAL_THRESHOLDS")
    ap.add_argument("--fss_windows", type=int, nargs="+", default=[1, 5, 11, 21])
    ap.add_argument("--rl_lambdas", type=float, nargs="+", default=[10.0, 100.0, 1000.0],
                    help="return levels at these expected exceedance counts")
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--tag", default=None, help="row label; defaults to --model")
    ap.add_argument("--output_dir", default="eval_results/extremes")
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.model
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] model={args.model} tag={tag} -> {out_dir}")

    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    pixel_km = float(config.get("PIXEL_SIZE_KM", 2.0))
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    u_phys = load_physical_thresholds(config)
    fss_thr = args.fss_thresholds or [float(u) for u in u_phys[-4:-1]]
    pot_u = args.pot_threshold if args.pot_threshold is not None else float(u_phys[-3])
    print(f"[eval] FSS thresholds {fss_thr} mm/h, windows {args.fss_windows}; POT u={pot_u} mm/h")

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    dem_stats = (float(d["dem_mean"]), float(d["dem_std"]))

    # ---- models -----------------------------------------------------------------
    backbone = fm = None
    sigma, sigma_src = 1.0, "n/a"
    if args.model in ("backbone", "fm"):
        bpath = args.checkpoint if args.model == "backbone" else (
            args.backbone or config.get("BACKBONE_CHECKPOINT"))
        if bpath is None:
            raise SystemExit("a backbone checkpoint is required for this model")
        backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
        backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
        backbone.eval()
        print(f"[eval] backbone: {bpath}")
    if args.model == "fm":
        if args.fm_checkpoint is None:
            raise SystemExit("--fm_checkpoint is required for --model fm")
        sigma, sigma_src = _load_sigma(config, device)
        print(f"[eval] residual scale sigma_r: {sigma_src}")
        cond_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
        _kw = dict(in_channels=1, c_in_condition=3 if cond_on_mean else 2,
                   time_scale=config.get("FM_TIME_SCALE", 1000.0), device=str(device))
        import inspect as _inspect
        if "prior" in _inspect.signature(FlowMatching.__init__).parameters:
            _kw["prior"] = config.get("FM_PRIOR", "gaussian")
            _kw["prior_nu"] = config.get("FM_PRIOR_NU", 5.0)
        fm = FlowMatching(**_kw).to(device)
        fm.load_state_dict(torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"])
        fm.eval()
        steps = args.steps or config.get("FM_SAMPLE_STEPS", 16)
        sampler = args.sampler or config.get("FM_SAMPLER", "heun")
        print(f"[eval] fm: {args.fm_checkpoint} ({sampler}, {steps} steps, M={args.ensemble})")

    thresh_b0 = load_persistence_thresholds(config)[0] if topo_mode == "b0" else 0.0
    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys, pixel_size_km=pixel_km, topology_mode=topo_mode,
        area_mode="ste", persistence_thresh_b0=thresh_b0,
    ).to(device)

    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config[_META[args.split]],
                                dem_stats, scaler_val, split=args.split, topology_mode=topo_mode)
    loader = DataLoader(ds, batch_size=config.get("BATCH_SIZE", 128), shuffle=False,
                        num_workers=config.get("NUM_WORKERS", 4), pin_memory=False,
                        multiprocessing_context="spawn")

    def decode(y):
        return torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * max_val))

    fss = FSSAccumulator(fss_thr, args.fss_windows)
    pot_p, pot_o = POTAccumulator(pot_u), POTAccumulator(pot_u)
    mse_i, tmax_i, S_i, A_i, L_i, aniso_i, pk_i, crps_i = [], [], [], [], [], [], [], []
    mink_sum, nb = 0.0, 0
    spec_i = []
    rapsd_p = rapsd_o = None

    with torch.no_grad():
        for bi, (X, Y, Ygamma) in enumerate(tqdm(loader, desc=f"extremes[{tag}]")):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
            target = decode(Y[:, 0:1])

            if args.model == "bicubic":
                pred = decode(X[:, 0:1])              # channel 0 is the interpolated input
            elif args.model == "backbone":
                pred = decode(backbone(X)[:, 0:1])
            else:
                mu = backbone(X)
                cond = torch.cat([X, mu[:, 0:1]], dim=1) if cond_on_mean else X
                members = []
                for _ in range(args.ensemble):
                    r = fm.sample(cond, n_steps=steps, method=sampler, in_channels=1)
                    members.append(decode(mu[:, 0:1] + sigma * r))
                ens = torch.stack(members, 0)          # [M,B,1,H,W]
                crps_i.append(crps_ensemble(ens[:, :, 0].cpu().numpy(),
                                            target[:, 0].cpu().numpy()).mean(axis=(1, 2)))
                pred = ens[0]                          # arbitrary member for structure
                pred_mean = ens.mean(0)
                mse_i.append(((pred_mean - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())

            if args.model != "fm":
                mse_i.append(((pred - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())

            tmax_i.append(target.amax(dim=(1, 2, 3)).cpu().numpy())
            mink_sum += geom_fn(pred, Ygamma, anneal_factor=0.05).item(); nb += 1
            rp_b = compute_radial_power_spectrum(pred)          # [B,K]
            ro_b = compute_radial_power_spectrum(target)
            # Per-sample log-spectral distance, kept for the perception-distortion cloud.
            # For --model fm this is a single ensemble member (as `pred` is), not the
            # ensemble mean, which is the right perception reference; the distortion axis
            # `mse` is the ensemble mean. The batch means below are unchanged, so
            # rapsd_log_distance is untouched.
            spec_i.append((torch.log(rp_b + 1e-12) - torch.log(ro_b + 1e-12))
                          .abs().mean(dim=-1).cpu().numpy())
            rp, ro = rp_b.mean(0), ro_b.mean(0)
            rapsd_p = rp if rapsd_p is None else rapsd_p + rp
            rapsd_o = ro if rapsd_o is None else rapsd_o + ro

            pp = pred.squeeze(1).cpu().numpy()
            tt = target.squeeze(1).cpu().numpy()
            fss.update(pp, tt)
            pot_p.update(pp); pot_o.update(tt)
            aniso_i.append(directional_anisotropy(pp))
            pk_i.append(peak_ratio(pp, tt))
            for i in range(pp.shape[0]):
                s, a, l = sal(pp[i], tt[i]); S_i.append(s); A_i.append(a); L_i.append(l)

    mse = np.concatenate(mse_i); tmax = np.concatenate(tmax_i)
    thr = float(np.percentile(tmax, args.extreme_pct)); ext = tmax >= thr
    rapsd_p = (rapsd_p / nb).cpu().numpy(); rapsd_o = (rapsd_o / nb).cpu().numpy()
    pk = np.concatenate(pk_i)
    spectral_dist = np.concatenate(spec_i)

    fit_p, fit_o = pot_p.fit(), pot_o.fit()
    summary = {
        "model": args.model, "tag": tag, "split": args.split,
        "n_samples": int(mse.size), "topology_mode": topo_mode,
        # --- table columns ---
        "rmse_extreme": float(np.sqrt(mse[ext].mean())),
        "rmse_overall": float(np.sqrt(mse.mean())),
        "rapsd_log_distance": float(np.mean(np.abs(np.log(rapsd_p + 1e-12)
                                                   - np.log(rapsd_o + 1e-12)))),
        "minkowski_distance": float(mink_sum / max(nb, 1)),
        **fss.result(),
        **gpd_comparison(fit_p, fit_o, args.rl_lambdas),
        # --- diagnostics the headline metrics cannot see ---
        "anisotropy_axis_over_diag": float(np.nanmean(aniso_i)),
        "peak_ratio_median": float(np.nanmedian(pk)),
        "peak_ratio_extreme_median": float(np.nanmedian(pk[ext])) if ext.any() else float("nan"),
        "SAL_median": [float(np.median(S_i)), float(np.median(A_i)), float(np.median(L_i))],
        "extreme_threshold_mmph": thr,
        "pot_threshold_mmph": pot_u,
        "note": ("minkowski_distance is the training objective for Minkowski-trained models "
                 "and is not independent evidence there; anisotropy ~1 is isotropic, >1 means "
                 "grid-aligned structure; peak_ratio is predicted/observed patch maximum."),
    }
    if crps_i:
        crps = np.concatenate(crps_i)
        summary["crps_mean"] = float(crps.mean())
        summary["crps_extreme"] = float(crps[ext].mean())
        summary["ensemble_size"] = args.ensemble

    with open(os.path.join(out_dir, "extremes_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    np.savez_compressed(os.path.join(out_dir, "extremes_arrays.npz"),
                        mse=mse, tmax=tmax, peak_ratio=pk,
                        S=np.array(S_i), A=np.array(A_i), L=np.array(L_i),
                        spectral_dist=spectral_dist,
                        rapsd_pred=rapsd_p, rapsd_target=rapsd_o,
                        gpd_pred=np.array([fit_p["xi"], fit_p["sigma"], fit_p["rate"]]),
                        gpd_obs=np.array([fit_o["xi"], fit_o["sigma"], fit_o["rate"]]),
                        pot_threshold=np.float32(pot_u))
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
