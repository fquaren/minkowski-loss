"""
Deterministic UNet super-resolution with frozen emulator loss.

Loss = MSE + w_geom(epoch) * trust * MinkowskiLoss(emulator(pred), gamma_true)

The geometric weight follows a cosine warmup schedule (0 → w_max),
allowing MSE to stabilise before topological constraints are applied.
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


def _compute_geometric_loss(
    emulator,
    criterion,
    denormalizer,
    Y_pred_norm,
    Y_clean_norm,
    Y_gamma_log,
    trust_tau,
    compute_trust=True,
):
    """Compute emulator-based geometric loss with optional trust weighting."""
    device = Y_pred_norm.device

    with torch.autocast(device_type=device.type, enabled=False):
        pred_f32 = Y_pred_norm.float()
        clean_f32 = Y_clean_norm.float()
        gamma_f32 = Y_gamma_log.float()

        trust_w = torch.ones(pred_f32.shape[0], device=device)
        avg_trust = 1.0

        if compute_trust:
            with torch.no_grad():
                gt_phys = denormalizer.to_physical_torch(clean_f32)
                gt_phys = gt_phys * (gt_phys > 0.1).float()
                gt_gamma_phys = emulator(gt_phys)
                gt_gamma_log = torch.log1p(gt_gamma_phys)
                err_sq = (gt_gamma_log - gamma_f32).pow(2).mean(dim=(1, 2))
                trust_w = torch.exp(-trust_tau * err_sq)
                avg_trust = trust_w.mean().item()

        pred_phys = denormalizer.to_physical_torch(pred_f32)
        pred_phys = pred_phys * (pred_phys > 0.1).float()
        pred_gamma_phys = emulator(pred_phys)
        pred_gamma_log = torch.log1p(pred_gamma_phys)

        batch_dist, _, _, _ = criterion(pred_gamma_log, gamma_f32)
        loss = (batch_dist * trust_w).mean()

    return loss, avg_trust


def run_training(config, args, trial=None):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Load prerequisites
    scaler_val = load_scaler_val(config)
    denormalizer = DataDenormalizer(scaler_val)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")

    with open(config["DEM_STATS"]) as f:
        ds = json.load(f)
    dem_stats = (float(ds["dem_mean"]), float(ds["dem_std"]))

    # Hyperparameters
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

    # Run directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    trial_s = f"_T{trial.number}" if trial else ""
    run_name = f"UNet_Emu{trial_s}_{ts}"
    out_dir = os.path.join("runs", "sr_emulator", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger(run_name, out_dir) as logger:
        # Data
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
        sampler = WeightedRandomSampler(
            train_ds.sample_weights,
            len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )

        # Models
        model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
        emulator = None
        geom_criterion = None
        if w_max > 0:
            geom_criterion = MinkowskiLoss(config["QUANTILE_LEVELS"]).to(device)
            emulator = load_emulator(
                config["EMULATOR_CHECKPOINT_PATH"],
                config,
                device,
            )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        mse_fn = nn.MSELoss()
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, patience // 2),
        )
        amp_enabled = device.type == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))

        best_val_loss = float("inf")

        for epoch in range(epochs):
            w_geom = cosine_warmup_weight(epoch, epochs, w_max, warmup)

            # --- Train ---
            model.train()
            r_total, r_mse, r_geom, r_trust = 0.0, 0.0, 0.0, 0.0

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
                            emulator,
                            geom_criterion,
                            denormalizer,
                            Y_pred,
                            Y,
                            Y_gamma,
                            trust_tau,
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

            # --- Validate ---
            model.eval()
            v_total, v_mse, v_geom = 0.0, 0.0, 0.0

            with torch.no_grad():
                for X, Y, Y_gamma in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    Y_gamma = Y_gamma.to(device)

                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        Y_pred = model(X)
                        lm = mse_fn(Y_pred, Y)
                        lg = torch.tensor(0.0, device=device)
                        if w_geom > 0 and emulator is not None:
                            lg, _ = _compute_geometric_loss(
                                emulator,
                                geom_criterion,
                                denormalizer,
                                Y_pred,
                                Y,
                                Y_gamma,
                                trust_tau,
                                compute_trust=False,
                            )
                    v_total += (lm + w_geom * lg).item()
                    v_mse += lm.item()
                    v_geom += lg.item()

            n_val = len(val_loader)
            avg_val_mse = v_mse / n_val
            sched.step(avg_val_mse)

            logger.info(
                f"Epoch {epoch+1} | "
                f"train_mse={r_mse/n_train:.4f} val_mse={avg_val_mse:.4f} "
                f"geom={v_geom/n_val:.4f} w={w_geom:.4f} "
                f"trust={r_trust/n_train:.3f}"
            )

            # Checkpoint on val_mse (consistent with early stopping)
            is_best = avg_val_mse < best_val_loss
            best_val_loss = min(best_val_loss, avg_val_mse)

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
                    extra={"best_val_mse": best_val_loss},
                )

            if not trial and ((epoch + 1) % 5 == 0 or epoch == 0):
                save_sample_images(
                    model,
                    val_loader,
                    device,
                    out_dir,
                    epoch + 1,
                    denormalizer,
                )

            if trial:
                trial.report(avg_val_mse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if early_stopper(avg_val_mse):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    return best_val_loss
