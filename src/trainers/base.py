"""
Shared training utilities.

Consolidates EarlyStopping, checkpoint management, sample visualisation,
physical metrics, and loss scheduling from the three training scripts
into a single source of truth.
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------
# 1. Early stopping
# ---------------------------------------------------------------

class EarlyStopping:
    """Monitors a metric and signals when training should stop.

    Parameters
    ----------
    patience : int
        Number of calls without improvement before stopping.
    delta : float
        Minimum improvement to qualify as "better".
    verbose : bool
        Print counter updates.
    """

    def __init__(self, patience=7, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        """Returns True if training should stop."""
        score = -metric
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
                return True
            return False

        self.best_score = score
        self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.triggered = False

    def state_dict(self):
        return {
            "patience": self.patience,
            "counter": self.counter,
            "best_score": self.best_score,
            "triggered": self.triggered,
            "delta": self.delta,
        }

    def load_state_dict(self, d):
        self.patience = d["patience"]
        self.counter = d["counter"]
        self.best_score = d["best_score"]
        self.triggered = d.get("triggered", False)
        self.delta = d["delta"]


# ---------------------------------------------------------------
# 2. Checkpointing
# ---------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, scaler=None,
                    scheduler=None, early_stopper=None, history=None,
                    extra=None):
    """Save a full training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if early_stopper is not None:
        state["early_stop_state"] = early_stopper.state_dict()
    if history is not None:
        state["history"] = history
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None,
                    scheduler=None, early_stopper=None, device="cpu"):
    """Load a training checkpoint. Returns (start_epoch, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if early_stopper and "early_stop_state" in ckpt:
        early_stopper.load_state_dict(ckpt["early_stop_state"])
    return ckpt.get("epoch", 0), ckpt.get("history", {})


# ---------------------------------------------------------------
# 3. Loss scheduling
# ---------------------------------------------------------------

def cosine_warmup_weight(epoch: int, total_epochs: int, w_max: float,
                         warmup_epochs: int = 0) -> float:
    if epoch < warmup_epochs:
        return 0.5 * w_max * (1 - np.cos(np.pi * epoch / warmup_epochs))
    return w_max


# ---------------------------------------------------------------
# 4. Physical metrics
# ---------------------------------------------------------------

def compute_physical_metrics(real: np.ndarray, gen: np.ndarray,
                             drizzle: float = 0.1) -> dict:
    """Compute Wasserstein distance and max intensity error.

    Both inputs are in physical space (mm/h).
    """
    real_f = real[real > drizzle].flatten()
    gen_f = gen[gen > drizzle].flatten()
    if len(real_f) == 0 or len(gen_f) == 0:
        return {"wasserstein_dist": 0.0, "max_intensity_err": 0.0}
    return {
        "wasserstein_dist": float(wasserstein_distance(real_f, gen_f)),
        "max_intensity_err": float(abs(np.max(real_f) - np.max(gen_f))),
    }


# ---------------------------------------------------------------
# 5. Sample visualisation
# ---------------------------------------------------------------

def save_sample_images(model, loader, device, out_dir, epoch, denormalizer,
                       diffusion=None):
    """Save a comparison plot of the wettest sample in the first batch.

    Handles both deterministic (UNet) and diffusion (DDPM) models.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : str
    out_dir : str
    epoch : int
    denormalizer : DataDenormalizer
    diffusion : Diffusion, optional
        If provided, uses DDIM sampling instead of direct forward pass.
    """
    model.eval()
    batch = next(iter(loader))
    X = batch[0].to(device)
    Y = batch[1].to(device)

    is_diffusion = diffusion is not None

    # Find wettest sample
    if is_diffusion:
        X_01 = (X[:, 0] + 1.0) / 2.0
    else:
        X_01 = X[:, 0]
    total_precip = denormalizer.to_physical_torch(X_01).sum(dim=(1, 2))
    max_idx = total_precip.argmax().item()
    X_s = X[max_idx:max_idx + 1]
    Y_s = Y[max_idx:max_idx + 1]

    with torch.no_grad():
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type):
            if is_diffusion:
                gen = diffusion.sample_ddim(model, 1, X_s, ddim_steps=50)
            else:
                gen = model(X_s)

    # Convert to physical space
    if is_diffusion:
        img_in = denormalizer.to_physical_np(
            (X_s[:, 0].cpu().numpy() + 1) / 2
        )[0]
        img_target = denormalizer.to_physical_np(
            (Y_s[:, 0].cpu().numpy() + 1) / 2
        )[0]
        img_gen = denormalizer.to_physical_np(
            (gen[:, 0].cpu().clamp(-1, 1).numpy() + 1) / 2
        )[0]
    else:
        img_in = denormalizer.to_physical_np(X_s[:, 0].cpu().numpy())[0]
        img_target = denormalizer.to_physical_np(Y_s[:, 0].cpu().numpy())[0]
        img_gen = denormalizer.to_physical_np(
            gen[:, 0].cpu().clamp(0, 1).numpy()
        )[0]

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"sample_data_epoch_{epoch:03d}.npz"),
        img_in=img_in, img_target=img_target, img_gen=img_gen,
    )

    cmap = copy.copy(plt.get_cmap("Blues"))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    def _mask(img, thresh=0.1):
        m = img.copy()
        m[m <= thresh] = np.nan
        return m

    vmax = max(np.nanmax(img_in), np.nanmax(img_target), np.nanmax(img_gen), 1.0)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, img, title in zip(
        axs,
        [img_in, img_gen, img_target],
        ["Input (LR)", "Generated (SR)", "Target (HR)"],
    ):
        im = ax.imshow(_mask(img), cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"{title} | Max: {np.nanmax(img):.2f}")
        ax.axis("off")

    plt.colorbar(im, ax=axs[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"),
        bbox_inches="tight", dpi=100,
    )
    plt.close()
