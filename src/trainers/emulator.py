"""
Training loop for the Minkowski functional emulator.

Supports both single-run and Optuna hyperparameter search.
Uses HomoscedasticMinkowskiLoss for learned multi-task weighting.
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.losses.minkowski import HomoscedasticMinkowskiLoss
from src.data.datasets import ZarrMixupDataset
from src.models.emulators import BaselineCNN, LipschitzCNN, ConstrainedLipschitzCNN
from src.utils import (
    load_config, load_scaler_val, set_seed, managed_logger,
)
from src.trainers.base import (
    EarlyStopping, save_checkpoint, load_checkpoint,
)


def _build_model(arch, config, topology_mode, device):
    n_q = len(config["QUANTILE_LEVELS"])
    ps = config["PATCH_SIZE"]
    shape = (1, ps, ps)
    pixel_km = config.get("PIXEL_SIZE_KM", 2.0)

    if arch == "Baseline":
        m = BaselineCNN(n_q, shape, topology_mode=topology_mode)
    elif arch == "Lipschitz":
        m = LipschitzCNN(n_q, shape, topology_mode=topology_mode)
    elif arch == "Constrained":
        m = ConstrainedLipschitzCNN(
            n_q, shape, config["QUANTILE_LEVELS"],
            pixel_area_km2=pixel_km ** 2,
            topology_mode=topology_mode,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m.to(device)


def _get_loaders(config, scaler_val, data_fraction, topology_mode):
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    nw = config.get("NUM_WORKERS", 4)

    train_ds = ZarrMixupDataset(
        zarr_path, split="train", scaler_val=scaler_val,
        augment=True, include_original=True, include_mixup=True,
        subset_fraction=data_fraction, topology_mode=topology_mode,
    )
    val_ds = ZarrMixupDataset(
        zarr_path, split="validation", scaler_val=scaler_val,
        augment=False, include_original=True, include_mixup=False,
        subset_fraction=data_fraction, topology_mode=topology_mode,
    )

    bs = config.get("BATCH_SIZE", 128)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
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
    return train_loader, val_loader


def run_training(config, args, trial=None):
    """Single training session (called directly or from Optuna)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_val = load_scaler_val(config)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")
    num_epochs = config.get("NUM_EPOCHS", 25)
    patience = config.get("EARLY_STOPPING_PATIENCE", 15)

    # Hyperparameters
    if trial:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    elif args.load_params and os.path.exists(args.load_params):
        import yaml
        with open(args.load_params) as f:
            p = yaml.safe_load(f)
        lr = p.get("lr", args.lr)
        wd = p.get("weight_decay", args.wd)
    else:
        lr = args.lr
        wd = args.wd

    # Run directory
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"T{trial.number}" if trial else "single"
    run_name = f"Emulator_{args.arch}_{run_id}_{ts}"
    out_dir = os.path.join("runs", "emulator", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with managed_logger(run_name, out_dir) as logger:
        logger.info(f"Architecture: {args.arch} | topology: {topology_mode}")
        logger.info(f"LR: {lr:.2e} | WD: {wd:.2e} | Device: {device}")

        train_loader, val_loader = _get_loaders(
            config, scaler_val, args.data_fraction, topology_mode,
        )

        model = _build_model(args.arch, config, topology_mode, device)
        criterion = HomoscedasticMinkowskiLoss(
            config["QUANTILE_LEVELS"]
        ).to(device)

        # Include criterion's log_vars in the optimizer
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=lr, weight_decay=wd,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(1, patience // 3),
        )
        early_stopper = EarlyStopping(patience=patience, verbose=(trial is None))

        best_val_loss = float("inf")
        log_path = os.path.join(out_dir, "training_log.csv")
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_A,val_P,val_T,"
                    "w_A,w_P,w_T,lr\n")

        for epoch in range(num_epochs):
            # --- Train ---
            model.train()
            criterion.train()
            train_loss_acc = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=(trial is not None),
                mininterval=30.0,
            )
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                pred_phys = model(inputs)
                pred_log = torch.log1p(pred_phys) if topology_mode == "b0" else \
                    torch.sign(pred_phys) * torch.log1p(torch.abs(pred_phys))
                total, d_a, d_p, d_t, weights = criterion(pred_log, targets)
                total.backward()
                optimizer.step()

                train_loss_acc += total.item()

            avg_train = train_loss_acc / len(train_loader)

            # --- Validate ---
            model.eval()
            criterion.eval()
            val_acc = {"total": 0.0, "A": 0.0, "P": 0.0, "T": 0.0}

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pred_phys = model(inputs)
                    pred_log = torch.log1p(pred_phys) if topology_mode == "b0" else \
                        torch.sign(pred_phys) * torch.log1p(torch.abs(pred_phys))
                    total, d_a, d_p, d_t, _ = criterion(pred_log, targets)
                    val_acc["total"] += total.item()
                    val_acc["A"] += d_a.item()
                    val_acc["P"] += d_p.item()
                    val_acc["T"] += d_t.item()

            n = len(val_loader)
            avg_val = {k: v / n for k, v in val_acc.items()}
            scheduler.step(avg_val["total"])

            # Learned weights (precision = 1/sigma^2)
            w = torch.exp(-criterion.log_vars).detach().cpu().numpy()
            cur_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch+1} | train={avg_train:.4f} "
                f"val={avg_val['total']:.4f} "
                f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}] lr={cur_lr:.2e}"
            )

            with open(log_path, "a") as f:
                f.write(
                    f"{epoch+1},{avg_train:.6f},{avg_val['total']:.6f},"
                    f"{avg_val['A']:.6f},{avg_val['P']:.6f},{avg_val['T']:.6f},"
                    f"{w[0]:.4f},{w[1]:.4f},{w[2]:.4f},{cur_lr:.2e}\n"
                )

            # Checkpoint
            is_best = avg_val["total"] < best_val_loss
            best_val_loss = min(best_val_loss, avg_val["total"])

            save_checkpoint(
                os.path.join(out_dir, "latest_checkpoint.pth"),
                epoch + 1, model, optimizer, scheduler=scheduler,
                early_stopper=early_stopper,
                extra={"arch": args.arch, "best_val_loss": best_val_loss},
            )
            if is_best:
                save_checkpoint(
                    os.path.join(out_dir, "best_model_checkpoint.pth"),
                    epoch + 1, model, optimizer,
                    extra={"arch": args.arch, "best_val_loss": best_val_loss},
                )

            # Optuna
            if trial:
                trial.report(avg_val["total"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Early stopping
            if early_stopper(avg_val["total"]):
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

        logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return best_val_loss
