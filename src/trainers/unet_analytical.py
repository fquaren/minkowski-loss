"""
Deterministic UNet super-resolution with analytical Minkowski loss.

Loss = MSE + w_geom(epoch) * AnalyticalMinkowskiLoss(pred_phys, gamma_true)

The sigmoid temperature anneals (sharpens) as training progresses,
while the geometric weight warms up from 0 → w_max. This ensures
the analytical approximation is most precise when its weight is highest.
"""

import json
import os
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss  # noqa: F401 (kept for compatibility)
from src.losses.competing import build_structural_loss
from src.models.unet import LogSpaceResidualUNet
from src.trainers.base import (
    EarlyStopping,
    cosine_warmup_weight,
    save_checkpoint,
    save_sample_images,
)
from src.utils import (
    DataDenormalizer,
    load_persistence_thresholds,
    load_physical_thresholds,
    load_scaler_val,
    managed_logger,
)


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

    with open(config["DEM_STATS"]) as f:
        ds = json.load(f)
    dem_stats = (float(ds["dem_mean"]), float(ds["dem_std"]))

    # Hyperparams
    if trial:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        with open(args.params_path) as f:
            p = yaml.safe_load(f)
        lr = p.get("lr", 1e-3)
        wd = p.get("weight_decay", 1e-4)

    epochs = (
        args.optuna_epochs
        if (trial and hasattr(args, "optuna_epochs"))
        else config.get("NUM_EPOCHS", 25)
    )
    patience = config.get("PATIENCE", 7)
    w_max = (
        args.weight_geom
        if args.weight_geom is not None
        else config.get("MINKOWSKI_TARGET_WEIGHT", 1.0)
    )
    warmup = config.get("MINKOWSKI_WARMUP_EPOCHS", 5)

    ts = time.strftime("%Y%m%d_%H%M%S")
    trial_s = f"_T{trial.number}" if trial else ""
    run_name = f"UNet_Ana{trial_s}_{ts}"
    out_dir = os.path.join("runs", "sr_analytical", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger(run_name, out_dir) as logger:
        train_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"],
            config["TRAIN_METADATA_FILE"],
            dem_stats,
            scaler_val,
            split="train",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )
        val_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"],
            config["VAL_METADATA_FILE"],
            dem_stats,
            scaler_val,
            split="validation",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )

        nw = config.get("NUM_WORKERS", 4)
        bs = config.get("BATCH_SIZE", 128)
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=WeightedRandomSampler(
                train_ds.sample_weights,
                len(train_ds),
                replacement=True,
            ),
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

        model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
        mse_fn = nn.MSELoss()
        # Study 1 selects the auxiliary structural loss by name; "minkowski" (the default)
        # reproduces the previous behaviour exactly, so an unset key changes nothing.
        structural_name = config.get("STRUCTURAL_LOSS", "minkowski")
        geom_fn = build_structural_loss(structural_name, config, device)
        logger.info(f"auxiliary structural loss: {geom_fn.name} (space={geom_fn.space})")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, patience // 2),
        )
        amp_enabled = device.type == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))
        max_val_t = torch.tensor(scaler_val, device=device, dtype=torch.float32)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            w_geom = cosine_warmup_weight(epoch, epochs, w_max, warmup)
            # Anneal sigmoid temperature: sharp when weight is high
            anneal = max(0.05, np.exp(-3.0 * epoch / max(epochs, 1)))

            model.train()
            r_total, r_mse, r_geom = 0.0, 0.0, 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [w={w_geom:.4f}]",
                disable=(trial is not None),
                mininterval=15.0,
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

                    if w_geom > 0:
                        pred_phys = F.relu(torch.expm1(
                            torch.clamp(Y_pred[:, 0:1] * max_val_t, max=7.0)))
                        target_phys = F.relu(torch.expm1(
                            torch.clamp(Y[:, 0:1] * max_val_t, max=7.0)))
                        ref_phys = F.relu(torch.expm1(
                            torch.clamp(X[:, 0:1] * max_val_t, max=7.0))) \
                            if geom_fn.needs_reference else None
                        loss_geom = geom_fn(
                            Y_pred[:, 0:1], Y[:, 0:1], pred_phys, target_phys,
                            gamma_target=Y_gamma, reference_phys=ref_phys,
                            anneal_factor=anneal,
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

            n_train = len(train_loader)

            # --- Validate ---
            model.eval()
            v_mse, v_geom = 0.0, 0.0

            with torch.no_grad():
                for X, Y, Y_gamma in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    Y_gamma = Y_gamma.to(device)

                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        Y_pred = model(X)
                        lm = mse_fn(Y_pred, Y)
                        lg = torch.tensor(0.0, device=device)
                        if w_max > 0:
                            pp = F.relu(torch.expm1(
                                torch.clamp(Y_pred[:, 0:1] * max_val_t, max=7.0)))
                            tp = F.relu(torch.expm1(
                                torch.clamp(Y[:, 0:1] * max_val_t, max=7.0)))
                            rp = F.relu(torch.expm1(
                                torch.clamp(X[:, 0:1] * max_val_t, max=7.0))) \
                                if geom_fn.needs_reference else None
                            lg = geom_fn(Y_pred[:, 0:1], Y[:, 0:1], pp, tp,
                                         gamma_target=Y_gamma, reference_phys=rp,
                                         anneal_factor=anneal)

                    v_mse += lm.item()
                    v_geom += lg.item()

            n_val = len(val_loader)
            avg_val_mse = v_mse / n_val
            avg_val_geom = v_geom / n_val
            # Selection / early-stopping monitor: full-target-weight total loss.
            # w_max (not the warmup w_geom) keeps the criterion stable across the
            # warmup ramp; reduces to val_mse when w_max == 0 (MSE-only control).
            val_monitor = avg_val_mse + w_max * avg_val_geom
            sched.step(val_monitor)

            logger.info(
                f"Epoch {epoch+1} | "
                f"train_mse={r_mse/n_train:.4f} val_mse={avg_val_mse:.4f} "
                f"geom={avg_val_geom:.4f} val_monitor={val_monitor:.4f} "
                f"w={w_geom:.4f} anneal={anneal:.3f}"
            )

            is_best = val_monitor < best_val_loss
            best_val_loss = min(best_val_loss, val_monitor)

            if trial is None:
                save_checkpoint(
                    os.path.join(out_dir, "unet_latest.pth"),
                    epoch + 1,
                    model,
                    optimizer,
                    scaler=grad_scaler if amp_enabled else None,
                    scheduler=sched,
                    early_stopper=early_stopper,
                )
                if is_best:
                    save_checkpoint(
                        os.path.join(out_dir, "unet_best.pth"),
                        epoch + 1,
                        model,
                        optimizer,
                        extra={"best_val_monitor": best_val_loss},
                    )
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    save_sample_images(
                        model,
                        val_loader,
                        device,
                        out_dir,
                        epoch + 1,
                        denormalizer,
                    )

            if trial:
                trial.report(val_monitor, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if early_stopper(val_monitor):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    return best_val_loss
