"""Phase 3: couple the Minkowski image loss to the flow-matching model (DRaFT-K).

The structural loss cannot be added to the flow-matching objective directly: the velocity
target lives on noised intermediate states, whereas the Minkowski functionals are defined
on a clean precipitation field. Following DRaFT-K (Clark et al., 2024, arXiv:2309.17400)
the reward is instead evaluated on the *generated sample* and back-propagated through only
the last K steps of the probability-flow ODE, which bounds memory and avoids the
vanishing-credit problem of full-trajectory differentiation.

Reward form. Extreme-value quantities are ensemble expectations -- E[A(u)] is the marginal
survival function and E[beta_0(A_u)] the field-maximum exceedance probability -- so the
theoretically aligned target is the *distribution* of functionals, not a per-sample match.
Two modes are provided so the choice can be tested rather than assumed:

    per_sample     mean over members of the Minkowski distance to that condition's target
    distributional conditional-mean matching E_m[log gamma_hat] vs log gamma, plus a
                   spread term matching the pooled standard deviation of log gamma

Guardrails. A per-sample structural reward invites degenerate textures: grid-aligned
filaments raise perimeter and component count at fixed area, improving SAL S, the
isoperimetric violation rate and even the radially averaged spectrum, while overshooting
peak intensity. Training therefore (i) anchors to the frozen clean model with a proximal
penalty, (ii) monitors directional anisotropy and the patch-peak ratio on held-out data,
and (iii) aborts if either crosses a configured bound. These two diagnostics are not part
of the reward, so they cannot be optimised away.
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.utils import (
    load_config, load_scaler_val, load_physical_thresholds, load_persistence_thresholds,
    managed_logger, set_seed,
)
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.evaluation.tail_metrics import directional_anisotropy, peak_ratio


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _load_sigma(config, device):
    """Identical precedence to the flow-matching trainer: map -> config -> json -> 1.0."""
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


def _base_sample(fm, shape, device, dtype):
    """Base-noise draw compatible with both the Gaussian-only model at HEAD and the
    optional Student-t prior (`_sample_base`) if that edit has been applied."""
    fn = getattr(fm, "_sample_base", None)
    if callable(fn):
        return fn(shape, device, dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def _make_fm(config, c_in, device):
    """Construct FlowMatching, passing prior kwargs only if this build supports them."""
    kw = dict(in_channels=1, c_in_condition=c_in,
              time_scale=config.get("FM_TIME_SCALE", 1000.0), device=str(device))
    import inspect as _inspect
    if "prior" in _inspect.signature(FlowMatching.__init__).parameters:
        kw["prior"] = config.get("FM_PRIOR", "gaussian")
        kw["prior_nu"] = config.get("FM_PRIOR_NU", 5.0)
    return FlowMatching(**kw).to(device)


def draft_k_sample(fm, cond, n_steps, method, K, in_channels=1, use_checkpoint=False):
    """Integrate the probability-flow ODE, keeping gradients only on the last K steps.

    The first n_steps-K steps run under no_grad; the remainder build a graph, so memory
    scales with K rather than with n_steps.
    """
    shape = (cond.shape[0], in_channels, cond.shape[2], cond.shape[3])
    x = _base_sample(fm, shape, cond.device, cond.dtype)
    dt = 1.0 / n_steps
    n_free = max(n_steps - K, 0)

    def _v(xx, tt):
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(lambda a, b: fm.velocity(a, b, cond), xx, tt, use_reentrant=False)
        return fm.velocity(xx, tt, cond)

    def _step(xx, i):
        t0 = i * dt
        t = torch.full((xx.shape[0],), t0, device=xx.device, dtype=xx.dtype)
        v = _v(xx, t)
        if method == "euler":
            return xx + dt * v
        xe = xx + dt * v
        t1 = torch.full_like(t, t0 + dt)
        return xx + 0.5 * dt * (v + _v(xe, t1))

    with torch.no_grad():
        for i in range(n_free):
            x = _step(x, i)
    for i in range(n_free, n_steps):
        x = _step(x, i)
    return x


def _log_gamma(geom_fn, field, anneal):
    """Differentiable log-space functional triple [B,3,Q] for a physical field."""
    a, p, t = geom_fn._functionals(field, anneal_factor=anneal)
    return torch.stack(
        [torch.log1p(a), torch.log1p(p), torch.sign(t) * torch.log1p(t.abs())], dim=1
    )


def _integrate_xi(geom_fn, resid):
    """Integrate |resid| [B,3,Q] over log-intensity with the loss's own measure -> scalar."""
    d = torch.trapezoid(resid.abs(), geom_fn.xi, dim=2) / geom_fn.xi_range   # [B,3]
    return d.mean()


def minkowski_reward(geom_fn, fields, target_log_gamma, mode, anneal, w_spread):
    """Structural reward. fields: [M,B,1,H,W]; target_log_gamma: [B,3,Q]."""
    M, B = fields.shape[0], fields.shape[1]
    lg = torch.stack([_log_gamma(geom_fn, fields[m], anneal) for m in range(M)], 0)  # [M,B,3,Q]

    if mode == "per_sample":
        return _integrate_xi(geom_fn, (lg - target_log_gamma.unsqueeze(0)).reshape(M * B, *lg.shape[2:]))

    if mode == "energy":
        # Energy score (kernel CRPS) on the functional curves, with the Minkowski distance
        # as the metric. Proper for the distribution of gamma: its minimiser is the true
        # conditional law, and unlike conditional-mean matching it needs only the single
        # observed gamma per condition -- which is all the data provides.
        #
        #   S = (1/M) sum_m d(g_m, g)  -  1/(2 M (M-1)) sum_{m != m'} d(g_m, g_m')
        #
        # The M(M-1) normalisation is the fair estimator; the biased M^2 form rewards
        # under-dispersion, which is precisely the failure mode being corrected. The
        # negative sign on the second term rewards spread between members, so no separate
        # spread penalty is needed and REWARD_SPREAD_WEIGHT is ignored in this mode.
        term1 = _integrate_xi(
            geom_fn, (lg - target_log_gamma.unsqueeze(0)).reshape(M * B, *lg.shape[2:])
        )
        if M < 2:
            return term1
        # pairwise distances between members, per condition
        a = lg.unsqueeze(1)                      # [M,1,B,3,Q]
        b_ = lg.unsqueeze(0)                     # [1,M,B,3,Q]
        pair = (a - b_).reshape(M * M * B, *lg.shape[2:])
        term2_all = _integrate_xi(geom_fn, pair) * (M * M)   # undo the mean over M*M*B
        # remove the M zero self-distances, then apply the fair 1/(M(M-1)) normalisation
        term2 = 0.5 * term2_all / (M * (M - 1))
        return term1 - term2

    if mode == "distributional":
        # conditional mean: E_m[log gamma_hat | c] against this condition's target
        cond_mean = lg.mean(0)                                        # [B,3,Q]
        loss_mean = _integrate_xi(geom_fn, cond_mean - target_log_gamma)
        # pooled spread: std over (members, conditions) against the target spread
        gen_std = lg.reshape(M * B, *lg.shape[2:]).std(dim=0, unbiased=False)   # [3,Q]
        tgt_std = target_log_gamma.std(dim=0, unbiased=False)                   # [3,Q]
        loss_std = _integrate_xi(
            geom_fn, (torch.log1p(gen_std) - torch.log1p(tgt_std)).unsqueeze(0)
        )
        return loss_mean + w_spread * loss_std

    raise ValueError(f"unknown reward mode {mode!r}")


# ----------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------
def run_reward_finetune(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed(getattr(args, "seed", 42))

    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    u_phys = load_physical_thresholds(config)
    pixel_km = float(config.get("PIXEL_SIZE_KM", 2.0))

    run_name = f"{config.get('EXPERIMENT_NAME', 'FM_MINK')}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(config.get("RUNS_DIR", "runs"), "sr_fm_minkowski", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger("reward_finetune", config.get("LOG_DIR", "logs")) as logger:
        # ---- frozen backbone -------------------------------------------------------
        bpath = getattr(args, "backbone", None) or config["BACKBONE_CHECKPOINT"]
        backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
        backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        sigma, sigma_src = _load_sigma(config, device)
        logger.info(f"residual scale sigma_r: {sigma_src}")

        cond_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
        c_in = 3 if cond_on_mean else 2

        def _mk_fm():
            return _make_fm(config, c_in, device)

        ckpt = torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"]
        fm = _mk_fm(); fm.load_state_dict(ckpt); fm.train()
        fm_ref = _mk_fm(); fm_ref.load_state_dict(copy.deepcopy(ckpt)); fm_ref.eval()
        for p in fm_ref.parameters():
            p.requires_grad_(False)
        logger.info(f"initialised from clean checkpoint {args.fm_checkpoint}")

        # ---- reward and guardrail settings ----------------------------------------
        mode = config.get("REWARD_MODE", "distributional")
        K = int(config.get("REWARD_K", 1))
        M = int(config.get("REWARD_ENSEMBLE", 4))
        w_reward = float(getattr(args, "w_reward", None) or config.get("REWARD_WEIGHT", 1e-3))
        w_prox = float(config.get("REWARD_PROXIMAL_WEIGHT", 1.0))
        w_retain = float(config.get("REWARD_FM_RETENTION_WEIGHT", 1.0))
        w_spread = float(config.get("REWARD_SPREAD_WEIGHT", 0.5))
        anneal = float(config.get("REWARD_ANNEAL", 0.05))
        n_steps = int(config.get("FM_SAMPLE_STEPS", 16))
        sampler = config.get("FM_SAMPLER", "heun")
        # Guards must be calibrated against this model family, not an absolute constant.
        # On OPERA the deterministic backbone already sits at anisotropy ~1.87 and bicubic
        # at ~31, because the coarse conditioning input carries grid-aligned interpolation
        # structure that any model inherits. A fixed bound of 1.35 therefore rejects
        # perfectly clean models. With AUTO, the frozen reference model's own held-out
        # values are measured before training and the bounds are multiples of those.
        guard_auto = bool(config.get("REWARD_GUARD_AUTO", True))
        guard_factor = float(config.get("REWARD_GUARD_FACTOR", 1.3))
        _am = config.get("REWARD_ANISO_MAX", None)
        _pm = config.get("REWARD_PEAK_RATIO_MAX", None)
        aniso_max = float(_am) if _am is not None else None
        peak_max = float(_pm) if _pm is not None else None
        use_ckpt = bool(config.get("REWARD_GRAD_CHECKPOINT", True))
        logger.info(f"reward mode={mode} K={K} M={M} w_reward={w_reward:g} w_prox={w_prox:g} "
                    f"w_retain={w_retain:g} "
                    f"w_spread={w_spread:g} guards: aniso<{aniso_max} peak<{peak_max}")

        thresh_b0 = load_persistence_thresholds(config)[0] if topo_mode == "b0" else 0.0
        geom_fn = AnalyticalMinkowskiLoss(
            physical_thresholds=u_phys, pixel_size_km=pixel_km, topology_mode=topo_mode,
            area_mode="ste", persistence_thresh_b0=thresh_b0,
        ).to(device)

        # ---- data ------------------------------------------------------------------
        with open(config["DEM_STATS"]) as f:
            d = json.load(f)
        dem_stats = (float(d["dem_mean"]), float(d["dem_std"]))
        mk_ds = lambda split, key: DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"], config[key], dem_stats, scaler_val,
            split=split, topology_mode=topo_mode)
        bs = int(config.get("REWARD_BATCH_SIZE", 8))
        train_loader = DataLoader(mk_ds("train", "TRAIN_METADATA_FILE"), batch_size=bs,
                                  shuffle=True, num_workers=config.get("NUM_WORKERS", 4),
                                  pin_memory=False, multiprocessing_context="spawn",
                                  persistent_workers=True, drop_last=True)
        val_loader = DataLoader(mk_ds("validation", "VAL_METADATA_FILE"), batch_size=bs,
                                shuffle=False, num_workers=2, pin_memory=False,
                                multiprocessing_context="spawn", persistent_workers=True)

        # ---- calibrate the guards on the frozen reference before any update ----------

        def _decode(y):
                    return torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * max_val))
        
        def _cond(X, mu):
            return torch.cat([X, mu[:, 0:1]], dim=1) if cond_on_mean else X
        
        if guard_auto and (aniso_max is None or peak_max is None):
            _a, _p, _n = [], [], 0
            with torch.no_grad():
                for vi, (Xg, Yg, _gg) in enumerate(val_loader):
                    if vi >= int(config.get("REWARD_GUARD_BATCHES", 8)):
                        break
                    Xg, Yg = Xg.to(device), Yg.to(device)
                    mug = backbone(Xg)
                    f0 = _decode(mug[:, 0:1] + sigma * fm_ref.sample(
                        _cond(Xg, mug), n_steps=n_steps, method=sampler, in_channels=1))
                    a0 = f0.squeeze(1).cpu().numpy()
                    _a.append(directional_anisotropy(a0))
                    _p.append(np.nanmedian(peak_ratio(a0, _decode(Yg[:, 0:1]).squeeze(1).cpu().numpy())))
                    _n += 1
            base_a, base_p = float(np.nanmean(_a)), float(np.nanmedian(_p))
            if aniso_max is None:
                aniso_max = guard_factor * base_a
            if peak_max is None:
                peak_max = max(guard_factor * base_p, 1.15)
            logger.info(f"guard baseline (clean reference): anisotropy={base_a:.3f} "
                        f"peak_ratio={base_p:.3f} -> bounds anisotropy<{aniso_max:.3f} "
                        f"peak_ratio<{peak_max:.3f}")
        aniso_max = float("inf") if aniso_max is None else aniso_max
        peak_max = float("inf") if peak_max is None else peak_max

        opt = torch.optim.AdamW(fm.parameters(), lr=float(config.get("REWARD_LR", 1e-5)),
                                weight_decay=0.0)
        epochs = int(config.get("REWARD_EPOCHS", 5))
        # One optimiser step costs M members x n_steps ODE evaluations; measured at ~7.7 s
        # for M=4, n_steps=16, heun. A full pass over the training split is ~600 h, so reward
        # fine-tuning is capped: it is a short correction to a trained model, not a new run.
        _cap = config.get("REWARD_STEPS_PER_EPOCH", 2000)
        max_steps = getattr(args, "max_steps", None) or (int(_cap) if _cap else None)
        best = float("inf")

        for epoch in range(epochs):
            fm.train()
            agg = {"reward": 0.0, "prox": 0.0, "fm": 0.0, "n": 0}
            pbar = tqdm(train_loader, desc=f"reward {epoch+1}/{epochs}")
            for si, (X, Y, Ygamma) in enumerate(pbar):
                if max_steps is not None and si >= max_steps:
                    break
                X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
                with torch.no_grad():
                    mu = backbone(X)
                cond = _cond(X, mu)

                fields = []
                for _ in range(M):
                    r = draft_k_sample(fm, cond, n_steps, sampler, K, 1, use_ckpt)
                    fields.append(_decode(mu[:, 0:1] + sigma * r))
                fields = torch.stack(fields, 0)                       # [M,B,1,H,W]

                reward = minkowski_reward(geom_fn, fields, Ygamma, mode, anneal, w_spread)

                # proximal anchor: keep the velocity field near the frozen reference
                with torch.no_grad():
                    x0 = _base_sample(fm, fields.shape[1:], device, X.dtype)
                    x1 = ((fields[0] / max_val)).clamp(0, 1)          # cheap in-range surrogate
                    tt = torch.rand(x0.shape[0], device=device)
                    xt = (1 - tt.view(-1, 1, 1, 1)) * x0 + tt.view(-1, 1, 1, 1) * x1
                    v_ref = fm_ref.velocity(xt, tt, cond)
                v_now = fm.velocity(xt, tt, cond)
                prox = ((v_now - v_ref) ** 2).mean()

                # Retention term. The reward and the proximal anchor between them contain no
                # pixel-aligned data-fidelity signal: the anchor constrains the velocity
                # field toward the reference, not where rain falls, and the Minkowski
                # functionals are rigid-motion invariant by construction. Without this term
                # nothing in the objective supplies placement, which is what the fractions
                # skill score measures. Re-adding the flow-matching velocity loss on the
                # real residual restores that anchor.
                loss_fm = torch.tensor(0.0, device=device)
                if w_retain > 0:
                    r_true = (Y[:, 0:1] - mu[:, 0:1]) / sigma
                    loss_fm = fm.training_loss(r_true, cond)

                loss = w_reward * reward + w_prox * prox + w_retain * loss_fm
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt.step()

                agg["reward"] += float(reward.detach()); agg["prox"] += float(prox.detach())
                agg["fm"] += float(loss_fm.detach()); agg["n"] += 1
                pbar.set_postfix(reward=f"{float(reward):.4f}", prox=f"{float(prox):.4f}")

            # ---- validation with reward-hacking guards ------------------------------
            fm.eval()
            v_reward, v_aniso, v_peak, nv = 0.0, [], [], 0
            with torch.no_grad():
                for vi, (X, Y, Ygamma) in enumerate(val_loader):
                    if vi >= int(config.get("REWARD_VAL_BATCHES", 20)):
                        break
                    X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
                    mu = backbone(X); cond = _cond(X, mu)
                    fields = torch.stack([
                        _decode(mu[:, 0:1] + sigma * fm.sample(cond, n_steps=n_steps,
                                                               method=sampler, in_channels=1))
                        for _ in range(M)], 0)
                    v_reward += float(minkowski_reward(geom_fn, fields, Ygamma, mode,
                                                       anneal, w_spread))
                    tgt = _decode(Y[:, 0:1])
                    p0 = fields[0].squeeze(1).cpu().numpy()
                    v_aniso.append(directional_anisotropy(p0))
                    v_peak.append(np.nanmedian(peak_ratio(p0, tgt.squeeze(1).cpu().numpy())))
                    nv += 1
            v_reward /= max(nv, 1)
            aniso = float(np.nanmean(v_aniso)); peak = float(np.nanmedian(v_peak))
            logger.info(f"Epoch {epoch+1} | train_reward={agg['reward']/max(agg['n'],1):.4f} "
                        f"prox={agg['prox']/max(agg['n'],1):.4f} fm={agg['fm']/max(agg['n'],1):.4f} "
                        f"val_reward={v_reward:.4f} "
                        f"| guards: anisotropy={aniso:.3f} peak_ratio={peak:.3f}")

            torch.save({"model_state_dict": fm.state_dict(), "epoch": epoch + 1},
                       os.path.join(out_dir, "fm_mink_latest.pth"))
            if v_reward < best:
                best = v_reward
                torch.save({"model_state_dict": fm.state_dict(), "epoch": epoch + 1,
                            "val_reward": v_reward, "anisotropy": aniso, "peak_ratio": peak},
                           os.path.join(out_dir, "fm_mink_best.pth"))

            tripped = []
            if aniso > aniso_max:
                tripped.append(f"anisotropy {aniso:.3f} > {aniso_max:.3f} (grid-aligned structure)")
            if peak > peak_max:
                tripped.append(f"peak ratio {peak:.3f} > {peak_max:.3f} (amplitude overshoot)")
            if tripped:
                logger.info("ABORT: reward hacking detected -- " + "; ".join(tripped) +
                            ". Lower REWARD_WEIGHT, raise REWARD_PROXIMAL_WEIGHT, or reduce "
                            "REWARD_K.")
                break

        logger.info(f"done. checkpoints in {out_dir}")
    return out_dir
