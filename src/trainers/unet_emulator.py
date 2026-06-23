"""
Deterministic UNet super-resolution with frozen emulator loss.

Loss = MSE + w_geom(epoch) * trust * MinkowskiLoss(emulator(pred), gamma_true)

The geometric weight follows a cosine warmup schedule (0 → w_max),
allowing MSE to stabilise before topological constraints are applied.

Early stopping, best-checkpoint selection, and LR scheduling all use the
combined validation objective  val_mse + w_max * val_geom  (fixed weight),
so training continues while either pixel accuracy OR geometric fidelity
is still improving.

Validation also reports structural metrics: per-feature gamma R²,
RAPS log-spectrum MSE, and isoperimetric violation rate on the
predicted physical field.
"""

import os
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import optuna

from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import MinkowskiLoss
from src.evaluation.training_metrics import (
    PerFeatureR2Accumulator,
    compute_isoperimetric_violation_rate,
    compute_raps_error,
)
from src.utils import (
    load_config,
    load_scaler_val,
    load_emulator,
    DataDenormalizer,
    managed_logger,
)
from src.trainers.base import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cosine_warmup_weight,
    save_sample_images,
)


# Physical thresholds for isoperimetric violation check (mm/h)
ISOPERIMETRIC_THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Soft-mask temperature for the drizzle threshold.
_DRIZZLE_THRESHOLD = 0.1
_DRIZZLE_TEMPERATURE = 0.05


def _soft_drizzle_mask(pred_phys: torch.Tensor) -> torch.Tensor:
    """Differentiable replacement for ``(pred_phys > 0.1).float()``."""
    return torch.sigmoid((pred_phys - _DRIZZLE_THRESHOLD) / _DRIZZLE_TEMPERATURE)


def _emulate(emulator, denormalizer, Y_norm):
    """Forward through the emulator. Returns (phys_field, log_gamma)."""
    phys = denormalizer.to_physical_torch(Y_norm.float())
    phys_masked = phys * _soft_drizzle_mask(phys)
    gamma_phys = emulator(phys_masked)
    gamma_log = torch.log1p(gamma_phys)
    return phys, gamma_log


def _compute_geometric_loss(
    emulator,
    criterion,
    denormalizer,
    Y_pred_norm,
    Y_clean_norm,
    Y_gamma_log,
    trust_tau,
    compute_trust=True,
    return_pred_gamma=False,
):
    """Compute emulator-based geometric loss with optional trust weighting.

    Parameters
    ----------
    return_pred_gamma : bool
        If True, also returns ``(pred_phys, pred_gamma_log)`` to feed
        validation-time structural metrics without a second forward pass.
    """
    device = Y_pred_norm.device

    with torch.autocast(device_type=device.type, enabled=False):
        gamma_f32 = Y_gamma_log.float()

        trust_w = torch.ones(Y_pred_norm.shape[0], device=device)
        avg_trust = 1.0

        if compute_trust:
            with torch.no_grad():
                _, gt_gamma_log = _emulate(emulator, denormalizer, Y_clean_norm)
                err_sq = (gt_gamma_log - gamma_f32).pow(2).mean(dim=(1, 2))
                trust_w = torch.exp(-trust_tau * err_sq)
                avg_trust = trust_w.mean().item()

        pred_phys, pred_gamma_log = _emulate(emulator, denormalizer, Y_pred_norm)

        batch_dist, _, _, _ = criterion(pred_gamma_log, gamma_f32)
        loss = (batch_dist * trust_w).mean()

    if return_pred_gamma:
        return loss, avg_trust, pred_phys, pred_gamma_log
    return loss, avg_trust


def run_training(config, args, trial=None):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    scaler_val = load_scaler_val(config)
    denormalizer = DataDenormalizer(scaler_val)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")
    pixel_km = config.get("PIXEL_SIZE_KM", 2.0)

    with open(config["DEM_STATS"]) as f:
        ds = json.load(f)
    dem_stats = (float(ds["dem_mean"]), float(ds["dem_std"]))

    with open(args.params_path) as f:
        unet_params = yaml.safe_load(f)

    if trial:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        lr = unet_params["lr"]
        wd = unet_params["weight_decay"]

    epochs = config.get("NUM_EPOCHS", 25)
    patience = config.get("PATIENCE", 5)
    w_max = (
        args.weight_geom
        if args.weight_geom is not None
        else config.get("MINKOWSKI_TARGET_WEIGHT", 0.0)
    )
    warmup = config.get("MINKOWSKI_WARMUP_EPOCHS", 5)
    trust_tau = config.get("TRUST_TAU", 0.1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    trial_s = f"_T{trial.number}" if trial else ""
    run_name = f"UNet_Emu{trial_s}_{ts}"
    out_dir = os.path.join("runs", "sr_emulator", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger(run_name, out_dir) as logger:
        # --- Data ---
        train_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"],
            config["TRAIN_METADATA_FILE"],
            dem_stats, scaler_val, split="train",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )
        val_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"],
            config["VAL_METADATA_FILE"],
            dem_stats, scaler_val, split="validation",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )

        nw = config.get("NUM_WORKERS", 4)
        bs = config.get("BATCH_SIZE", 128)
        sampler = WeightedRandomSampler(
            train_ds.sample_weights, len(train_ds), replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            num_workers=nw,
            pin_memory=False,
            persistent_workers=False, #nw > 0,
            multiprocessing_context="spawn",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=False,
            multiprocessing_context="spawn",
        )

        # --- Models ---
        model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)

        # Always load emulator if available — needed for the combined
        # early-stopping metric and structural metrics, even when w_max=0.
        emulator = None
        geom_criterion = None
        emu_ckpt = config.get("EMULATOR_CHECKPOINT_PATH", "")
        if emu_ckpt and os.path.exists(emu_ckpt):
            emulator = load_emulator(emu_ckpt, config, device)
            geom_criterion = MinkowskiLoss(config["QUANTILE_LEVELS"]).to(device)
            logger.info(f"Emulator loaded from {emu_ckpt}")
        else:
            logger.warning(
                "No emulator available — geometric term disabled; "
                "early stopping falls back to val_mse only."
            )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        mse_fn = nn.MSELoss()
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(1, patience // 2),
        )
        amp_enabled = device.type == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))

        best_val_combined = float("inf")

        for epoch in range(epochs):
            w_geom = cosine_warmup_weight(epoch, epochs, w_max, warmup)

            # ---------------- Train ----------------
            model.train()
            r_total = r_mse = r_geom = r_trust = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [w={w_geom:.4f}]",
                disable=(trial is not None),
                mininterval=30.0,
            )
            for X, Y, Y_gamma in pbar:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                Y_gamma = Y_gamma.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    Y_pred = model(X)
                    loss_mse = mse_fn(Y_pred, Y)
                    loss_geom = torch.tensor(0.0, device=device)
                    trust = 1.0

                    if w_geom > 0 and emulator is not None:
                        loss_geom, trust = _compute_geometric_loss(
                            emulator, geom_criterion, denormalizer,
                            Y_pred, Y, Y_gamma, trust_tau,
                        )

                    total = loss_mse + w_geom * loss_geom

                grad_scaler.scale(total).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                r_total += total.item()
                r_mse += loss_mse.item()
                r_geom += loss_geom.item()
                r_trust += trust

            n_train = len(train_loader)

            # ---------------- Validate ----------------
            model.eval()
            v_mse = v_geom = 0.0
            v_raps = 0.0
            v_iso_mean = v_iso_max = 0.0
            r2_acc = PerFeatureR2Accumulator()
            n_val_batches = 0

            with torch.no_grad():
                for X, Y, Y_gamma in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    Y_gamma = Y_gamma.to(device)

                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        Y_pred = model(X)
                        lm = mse_fn(Y_pred, Y)

                    # Geometric loss + intermediates (single emulator pass)
                    lg = torch.tensor(0.0, device=device)
                    if emulator is not None:
                        with torch.amp.autocast(device.type, enabled=amp_enabled):
                            lg, _, pred_phys, pred_gamma_log = (
                                _compute_geometric_loss(
                                    emulator, geom_criterion, denormalizer,
                                    Y_pred, Y, Y_gamma, trust_tau,
                                    compute_trust=False,
                                    return_pred_gamma=True,
                                )
                            )
                        r2_acc.update(pred_gamma_log, Y_gamma)
                    else:
                        pred_phys = denormalizer.to_physical_torch(Y_pred.float())

                    target_phys = denormalizer.to_physical_torch(Y.float())

                    # Structural metrics (always computed)
                    v_raps += compute_raps_error(pred_phys, target_phys).item()
                    iso_mean, iso_max = compute_isoperimetric_violation_rate(
                        pred_phys, ISOPERIMETRIC_THRESHOLDS, pixel_km,
                    )
                    v_iso_mean += iso_mean
                    v_iso_max += iso_max

                    v_mse += lm.item()
                    v_geom += lg.item()
                    n_val_batches += 1

            avg_val_mse = v_mse / n_val_batches
            avg_val_geom = v_geom / n_val_batches
            avg_val_raps = v_raps / n_val_batches
            avg_iso_mean = v_iso_mean / n_val_batches
            avg_iso_max = v_iso_max / n_val_batches
            r2_results = r2_acc.compute()

            # Combined early-stopping objective with FIXED target weight w_max.
            # Using w_max (not the ramping w_geom) keeps the metric on a
            # consistent scale across epochs, so improvements in geometry
            # are always credited at full strength.
            val_combined = avg_val_mse + w_max * avg_val_geom

            sched.step(val_combined)

            logger.info(
                f"Epoch {epoch+1} | "
                f"train_total={r_total/n_train:.4f} "
                f"train_mse={r_mse/n_train:.4f} val_mse={avg_val_mse:.4f} "
                f"geom={avg_val_geom:.4f} combined={val_combined:.4f} "
                f"w={w_geom:.4f} trust={r_trust/n_train:.3f}"
            )
            logger.info(
                f"  Structural | "
                f"R2_A={r2_results['R2_A']:.4f} "
                f"R2_P={r2_results['R2_P']:.4f} "
                f"R2_T={r2_results['R2_T']:.4f} | "
                f"RAPS_logMSE={avg_val_raps:.4f} | "
                f"iso_mean={avg_iso_mean*100:.2f}% "
                f"iso_max={avg_iso_max*100:.2f}%"
            )

            # --- Checkpoint on the combined metric ---
            is_best = val_combined < best_val_combined
            best_val_combined = min(best_val_combined, val_combined)

            save_checkpoint(
                os.path.join(out_dir, "unet_latest.pth"),
                epoch + 1, model, optimizer,
                scaler=grad_scaler if amp_enabled else None,
                scheduler=sched, early_stopper=early_stopper,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(out_dir, "unet_best.pth"),
                    epoch + 1, model, optimizer,
                    extra={
                        "best_val_combined": best_val_combined,
                        "val_mse": avg_val_mse,
                        "val_geom": avg_val_geom,
                        **r2_results,
                        "raps_logmse": avg_val_raps,
                        "iso_max_pct": avg_iso_max,
                    },
                )

            if not trial and ((epoch + 1) % 5 == 0 or epoch == 0):
                save_sample_images(
                    model, val_loader, device, out_dir, epoch + 1, denormalizer,
                )

            if trial:
                trial.report(val_combined, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # --- Early stopping on the combined metric ---
            if early_stopper(val_combined):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    return best_val_combined