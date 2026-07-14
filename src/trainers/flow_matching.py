"""
Conditional flow-matching trainer for residual super-resolution (CorrDiff-style).

Stage 1 (frozen): the backbone LogSpaceResidualUNet predicts the conditional mean mu(c) in
log-normalised [0,1] field space. Stage 2 (trained here): a flow-matching model learns the
standardised residual r_tilde = (x1 - mu(c)) / sigma_r, conditioned on c = [interp, dem, mu].
Sampling reconstructs the full field as clamp(mu + sigma_r * r_hat, 0, 1) -> physical mm/h.

sigma_r is a scalar (config FM_RESIDUAL_SCALE) or a per-pixel map (residual_std.npy), per the
recommendation from scripts/preprocess/compute_residual_stats.py.
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import optuna
except Exception:  # optuna optional
    optuna = None

from src.utils import load_config, load_scaler_val, managed_logger  # noqa: F401 (load_config used by callers)
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.trainers.base import EarlyStopping, save_checkpoint

_META = {"train": "TRAIN_METADATA_FILE", "validation": "VAL_METADATA_FILE"}


def _load_sigma(config, device):
    """Return sigma_r as a float scalar or a broadcastable tensor [1,1,H,W] (floored)."""
    if config.get("FM_USE_RESIDUAL_MAP", False):
        path = config.get(
            "FM_RESIDUAL_STD_PATH",
            os.path.join(config["PREPROCESSED_DATA_DIR"], "residual_std.npy"),
        )
        arr = np.load(path).astype(np.float32)              # [1,H,W]
        t = torch.from_numpy(arr).to(device).clamp_min(1e-3)
        return t.unsqueeze(0)                               # [1,1,H,W]
    return float(config.get("FM_RESIDUAL_SCALE", 1.0))


def run_training(config, args, trial=None):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    import json
    with open(config["DEM_STATS"]) as f:
        dstats = json.load(f)
    dem_stats = (float(dstats["dem_mean"]), float(dstats["dem_std"]))

    # --- Stage 1: frozen backbone (conditional mean) ---
    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    bpath = getattr(args, "backbone", None) or config["BACKBONE_CHECKPOINT"]
    backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # --- residual standardisation + flow-matching model ---
    sigma = _load_sigma(config, device)
    condition_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
    c_cond = 3 if condition_on_mean else 2
    fm = FlowMatching(
        in_channels=1, c_in_condition=c_cond,
        time_scale=config.get("FM_TIME_SCALE", 1000.0), device=str(device),
    ).to(device)

    # --- hyperparameters ---
    if trial is not None:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    else:
        lr = config.get("LEARNING_RATE", 1e-4)
        wd = config.get("WEIGHT_DECAY", 1e-5)
    epochs = config.get("NUM_EPOCHS", 100)
    patience = config.get("PATIENCE", 10)
    bs = config.get("BATCH_SIZE", 128)
    nw = config.get("NUM_WORKERS", 4)
    n_steps = config.get("FM_SAMPLE_STEPS", 16)
    sampler = config.get("FM_SAMPLER", "heun")

    run_name = f"{config.get('EXPERIMENT_NAME', 'FM_SR')}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("runs", "sr_flow_matching", run_name)
    os.makedirs(out_dir, exist_ok=True)

    def _build_cond(X, mu):
        return torch.cat([X, mu[:, 0:1]], dim=1) if condition_on_mean else X

    with managed_logger(run_name, out_dir) as logger:
        train_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"], config[_META["train"]], dem_stats, scaler_val,
            split="train", data_percentage=getattr(args, "data_percentage", 100.0),
            topology_mode=topo_mode,
        )
        val_ds = DeterministicSRDataset(
            config["PREPROCESSED_DATA_DIR"], config[_META["validation"]], dem_stats, scaler_val,
            split="validation", topology_mode=topo_mode,
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=False,
            multiprocessing_context="spawn",
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False,
            multiprocessing_context="spawn",
        )

        optimizer = optim.AdamW(fm.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                     patience=max(1, patience // 2))
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))
        amp_enabled = device.type == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_val = float("inf")

        for epoch in range(epochs):
            fm.train()
            r_train, n_train = 0.0, 0
            for X, Y, _ in tqdm(train_loader, desc=f"FM {epoch+1}/{epochs}"):
                X, Y = X.to(device), Y.to(device)
                with torch.no_grad():
                    mu = backbone(X)                          # [B,1,H,W] in [0,1]
                r_tilde = (Y[:, 0:1] - mu[:, 0:1]) / sigma     # standardised residual
                cond = _build_cond(X, mu)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    loss = fm.training_loss(r_tilde, cond)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fm.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                r_train += loss.item()
                n_train += 1

            # --- validation: same velocity loss ---
            fm.eval()
            v_loss, n_val = 0.0, 0
            with torch.no_grad():
                for X, Y, _ in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    mu = backbone(X)
                    r_tilde = (Y[:, 0:1] - mu[:, 0:1]) / sigma
                    cond = _build_cond(X, mu)
                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        v_loss += fm.training_loss(r_tilde, cond).item()
                    n_val += 1
            avg_val = v_loss / max(n_val, 1)
            sched.step(avg_val)
            logger.info(
                f"Epoch {epoch+1} | train_fm={r_train/max(n_train,1):.4f} "
                f"val_fm={avg_val:.4f}"
            )

            is_best = avg_val < best_val
            best_val = min(best_val, avg_val)
            if trial is None:
                save_checkpoint(os.path.join(out_dir, "fm_latest.pth"), epoch + 1, fm,
                                optimizer, scaler=grad_scaler if amp_enabled else None,
                                scheduler=sched, early_stopper=early_stopper)
                if is_best:
                    save_checkpoint(os.path.join(out_dir, "fm_best.pth"), epoch + 1, fm,
                                    optimizer, extra={"best_val_fm": best_val})
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    _save_reconstructions(fm, backbone, val_loader, sigma, max_val,
                                          condition_on_mean, n_steps, sampler, device,
                                          os.path.join(out_dir, f"recon_epoch{epoch+1}.npz"))

            if trial is not None:
                trial.report(avg_val, epoch)
                if optuna is not None and trial.should_prune():
                    raise optuna.TrialPruned()
            if early_stopper(avg_val):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    return best_val


@torch.no_grad()
def _save_reconstructions(fm, backbone, loader, sigma, max_val, condition_on_mean,
                          n_steps, sampler, device, path, n=8):
    """Sample residuals, reconstruct full physical fields, and save a few for inspection."""
    fm.eval()
    X, Y, _ = next(iter(loader))
    X, Y = X[:n].to(device), Y[:n].to(device)
    mu = backbone(X)
    cond = torch.cat([X, mu[:, 0:1]], dim=1) if condition_on_mean else X
    r_hat = sigma * fm.sample(cond, n_steps=n_steps, method=sampler, in_channels=1)
    full = torch.clamp(mu[:, 0:1] + r_hat, 0.0, 1.0)
    pred_phys = torch.relu(torch.expm1(full * max_val))
    target_phys = torch.relu(torch.expm1(Y[:, 0:1] * max_val))
    mean_phys = torch.relu(torch.expm1(torch.clamp(mu[:, 0:1] * max_val, max=7.0)))
    np.savez_compressed(
        path,
        pred_phys=pred_phys.cpu().numpy(), target_phys=target_phys.cpu().numpy(),
        mean_phys=mean_phys.cpu().numpy(),
    )