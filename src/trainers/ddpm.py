"""
DDPM super-resolution training loop.

Standard denoising score matching: the model learns to predict noise
ε given a noisy version of the target at timestep t, conditioned on
the low-resolution input + DEM.

No geometric loss is applied (this is the stochastic baseline).
"""

import os
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna

from src.models.ddpm import ContextUnet
from src.models.diffusion import Diffusion
from src.data.datasets import DiffusionSRDataset
from src.utils import load_config, load_scaler_val, DataDenormalizer, managed_logger
from src.trainers.base import (
    EarlyStopping,
    save_checkpoint,
    save_sample_images,
)


def run_training(config, args, trial=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    scaler_val = load_scaler_val(config)
    denormalizer = DataDenormalizer(scaler_val)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")

    with open(config["DEM_STATS"]) as f:
        ds = json.load(f)
    dem_stats = (float(ds["dem_mean"]), float(ds["dem_std"]))

    # Hyperparams
    with open(args.params_path) as f:
        ddpm_params = yaml.safe_load(f)

    if trial:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        lr = ddpm_params["lr"]
        wd = ddpm_params["weight_decay"]

    epochs = config.get("NUM_EPOCHS", 25)
    patience = config.get("PATIENCE", 5)
    bs = config.get("BATCH_SIZE", 128)
    nw = config.get("NUM_WORKERS", 4)
    ps = config.get("PATCH_SIZE", 128)

    ts = time.strftime("%Y%m%d_%H%M%S")
    trial_s = f"_T{trial.number}" if trial else ""
    exp = config.get("EXPERIMENT_NAME", "DDPM_SR")
    run_name = f"{exp}{trial_s}_{ts}"
    out_dir = os.path.join("runs", "sr_ddpm", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger(run_name, out_dir) as logger:
        train_ds = DiffusionSRDataset(
            preprocessed_data_dir=config["PREPROCESSED_DATA_DIR"],
            metadata_file=config["TRAIN_METADATA_FILE"],
            dem_stats=dem_stats,
            scaler_max_val=scaler_val,
            split="train",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )

        val_ds = DiffusionSRDataset(
            preprocessed_data_dir=config["PREPROCESSED_DATA_DIR"],
            metadata_file=config["VAL_METADATA_FILE"],
            dem_stats=dem_stats,
            scaler_max_val=scaler_val,
            split="validation",
            data_percentage=args.data_percentage,
            topology_mode=topology_mode,
        )
        train_loader = DataLoader(
            train_ds,
            bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            val_ds,
            bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )

        model = ContextUnet(
            in_channels=1,
            c_in_condition=2,
            device=device,
        ).to(device)
        diffusion = Diffusion(img_size=ps, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        mse_fn = nn.MSELoss()
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, patience // 2),
        )
        amp_enabled = device == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))

        best_val_mse = float("inf")

        for epoch in range(epochs):
            model.train()
            r_loss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=(trial is not None),
                mininterval=30.0,
            )
            for X, Y, *_ in pbar:
                X = X.to(device)
                Y = Y.to(device)
                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred_noise = model(x_t, t, X)
                    loss = mse_fn(noise, pred_noise)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                r_loss += loss.item()

            # --- Validate (every epoch) ---
            model.eval()
            v_mse = 0.0
            with torch.no_grad():
                for X, Y, *_ in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    t = diffusion.sample_timesteps(X.shape[0])
                    x_t, noise = diffusion.noise_images(Y, t)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        pred_n = model(x_t, t, X)
                        v_mse += mse_fn(noise, pred_n).item()

            avg_val_mse = v_mse / len(val_loader)
            sched.step(avg_val_mse)

            logger.info(
                f"Epoch {epoch+1} | "
                f"train={r_loss/len(train_loader):.4f} val={avg_val_mse:.4f}"
            )

            # Full checkpoint (not just state_dict)
            is_best = avg_val_mse < best_val_mse
            best_val_mse = min(best_val_mse, avg_val_mse)

            save_checkpoint(
                os.path.join(out_dir, "ddpm_latest.pth"),
                epoch + 1,
                model,
                optimizer,
                scaler=grad_scaler if amp_enabled else None,
                scheduler=sched,
                early_stopper=early_stopper,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(out_dir, "ddpm_best.pth"),
                    epoch + 1,
                    model,
                    optimizer,
                    extra={"best_val_mse": best_val_mse},
                )

            if not trial and ((epoch + 1) % 5 == 0 or epoch == 0):
                save_sample_images(
                    model,
                    val_loader,
                    device,
                    out_dir,
                    epoch + 1,
                    denormalizer,
                    diffusion=diffusion,
                )

            if trial:
                trial.report(avg_val_mse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if early_stopper(avg_val_mse):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    return best_val_mse
