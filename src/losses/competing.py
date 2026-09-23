"""Competing structural losses for the Study-1 comparison (research question 2).

Each loss here is an alternative to the Minkowski image loss as the auxiliary structural
term added to the mean squared error. They are wrapped in a common interface so the
trainer selects one by name and nothing else changes.

The wrapper exists because these losses do not share the Minkowski loss's signature. The
Minkowski loss compares a predicted field against precomputed gamma-curve targets; the
others compare two fields directly, and the optical-flow loss additionally needs the
coarse conditioning field. The wrapper also chooses the space each loss operates in:
scale-sensitive losses (spectral, structural similarity) run on the log-normalised field,
where the dynamic range is bounded and a handful of convective peaks cannot dominate,
while threshold-based losses (wet area, optical flow) run on physical intensities where
the drizzle threshold has its meaning.

Note on the spectral loss: with a deterministic backbone there is one prediction per
condition, so the ensemble term of the kernel score vanishes and the loss reduces exactly
to an L1 distance between Fourier magnitude spectra. That is a reasonable spectral
structural loss and a fair competitor, but it is not a proper scoring rule in this
setting -- the kernel-CRPS interpretation only applies when several members are supplied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# individual losses
# ----------------------------------------------------------------------------------
class FFT2DKernelCRPSLoss(nn.Module):
    """Kernel-CRPS / energy-score style loss on Fourier magnitude spectra.

    With an ensemble dimension it is the energy score of the spectra; with a single
    prediction (deterministic backbone) the second term drops and it is a spectral L1.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def _to_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                       # [B,C,H,W] -> [B,1,C,H,W], ensemble of one
            x = x[:, None, ...]
        mag = torch.abs(torch.fft.rfft2(x, dim=(-2, -1), norm="ortho"))
        return mag.flatten(start_dim=-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self._to_features(pred)
        tgt_f = self._to_features(target)
        if pred_f.shape[0] != tgt_f.shape[0]:
            raise ValueError("batch size mismatch between pred and target")
        if pred_f.shape[-1] != tgt_f.shape[-1]:
            raise ValueError("feature size mismatch between pred and target")
        if tgt_f.shape[1] == 1 and pred_f.shape[1] > 1:
            tgt_f = tgt_f.expand(pred_f.shape[0], pred_f.shape[1], tgt_f.shape[2])

        diff = torch.abs(pred_f - tgt_f).clamp_min(self.eps)
        d1 = diff.pow(self.alpha).mean(dim=-1).pow(1.0 / self.alpha)
        term1 = d1.mean(dim=1)

        m = pred_f.shape[1]
        if m <= 1:
            return term1.mean()
        pdiff = torch.abs(pred_f[:, :, None, :] - pred_f[:, None, :, :]).clamp_min(self.eps)
        d2 = pdiff.pow(self.alpha).mean(dim=-1).pow(1.0 / self.alpha)
        return (term1 - 0.5 * d2.mean(dim=(1, 2))).mean()


class ImageSSIMLoss(nn.Module):
    """Structural similarity loss on [B, C, H, W] tensors, with a per-sample data range."""

    def __init__(self, window_size: int = 11, k1: float = 0.01, k2: float = 0.03,
                 eps: float = 1.0e-6) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.k1, self.k2, self.eps = float(k1), float(k2), float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shapes")
        c = pred.shape[1]
        kernel = pred.new_full((c, 1, self.window_size, self.window_size),
                               1.0 / (self.window_size**2))
        pad = self.window_size // 2
        mu_p = F.conv2d(pred, kernel, padding=pad, groups=c)
        mu_t = F.conv2d(target, kernel, padding=pad, groups=c)
        var_p = F.conv2d(pred * pred, kernel, padding=pad, groups=c) - mu_p.square()
        var_t = F.conv2d(target * target, kernel, padding=pad, groups=c) - mu_t.square()
        cov = F.conv2d(pred * target, kernel, padding=pad, groups=c) - mu_p * mu_t
        var_p = torch.clamp(var_p, min=0.0)
        var_t = torch.clamp(var_t, min=0.0)

        hi = torch.maximum(pred, target).amax(dim=(-2, -1), keepdim=True)
        lo = torch.minimum(pred, target).amin(dim=(-2, -1), keepdim=True)
        data_range = (hi - lo).clamp_min(self.eps)
        c1 = (self.k1 * data_range).square()
        c2 = (self.k2 * data_range).square()

        num = (2.0 * mu_p * mu_t + c1) * (2.0 * cov + c2)
        den = (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)
        ssim = num / (den + self.eps)
        return torch.clamp((1.0 - ssim) * 0.5, min=0.0).mean()


class WeightedSoftWetAreaLoss(nn.Module):
    """Asymmetric soft wet/dry area agreement.

    The threshold must sit at the drizzle level of whichever space the loss runs in, and
    the temperature must be small relative to it, or every dry pixel is scored as half
    wet: sigmoid((0 - 0)/tau) = 0.5.
    """

    def __init__(self, threshold: float = 0.1, temperature: float = 0.05,
                 false_positive_weight: float = 1.0,
                 false_negative_weight: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.fp_w = float(false_positive_weight)
        self.fn_w = float(false_negative_weight)

    def _soft_wet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.threshold) / self.temperature)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = self._soft_wet(pred) - self._soft_wet(target)
        fp, fn = torch.relu(diff), torch.relu(-diff)
        return (self.fp_w * fp.square() + self.fn_w * fn.square()).mean()


def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k1 = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    k1 = k1 / k1.sum()
    kernel = torch.outer(k1, k1).view(1, 1, 2 * radius + 1, -1).expand(x.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x, (radius,) * 4, mode="replicate"), kernel, groups=x.shape[1])


def _shift_with_border(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    _, _, h, w = x.shape
    padded = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)), mode="replicate")
    return padded[..., max(dy, 0): max(dy, 0) + h, max(dx, 0): max(dx, 0) + w]


class OpticalFlowConsistencyLoss(nn.Module):
    """Advection-consistency loss: warp the coarse conditioning field onto the target and
    use the result as a teacher for the prediction.

    A block-matching flow is estimated between the conditioning field and the target, so
    the teacher carries the target's displacement but the conditioning field's texture.
    The flow is obtained by an argmin over candidate shifts and is not differentiable;
    only the prediction-to-teacher term carries gradient, which is intended.

    Cost scales as (2 * search_radius + 1)^2 correlation volumes, so the defaults keep the
    search small and the matching is done on a downsampled field.
    """

    def __init__(self, x_dim: int, y_dim: int, patch_size: int = 21,
                 search_radius: int = 4, downsample: int = 4,
                 preprocess_sigma: float = 1.0, delta: float = 0.2,
                 mask_threshold: float | None = 0.05, loss_type: str = "huber",
                 rain_threshold: float = 0.1) -> None:
        super().__init__()
        self.x_dim, self.y_dim = int(x_dim), int(y_dim)
        self.patch_size = int(patch_size)
        self.search_radius = int(search_radius)
        self.downsample = int(downsample)
        self.preprocess_sigma = float(preprocess_sigma)
        self.delta = float(delta)
        self.mask_threshold = mask_threshold if mask_threshold is None else float(mask_threshold)
        self.loss_type = str(loss_type)
        self.rain_threshold = float(rain_threshold)
        yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, self.y_dim),
                                torch.linspace(-1.0, 1.0, self.x_dim), indexing="ij")
        self.register_buffer("_base_grid", torch.stack((xx, yy), -1).unsqueeze(0),
                             persistent=False)

    def _working_size(self):
        return max(1, self.y_dim // self.downsample), max(1, self.x_dim // self.downsample)

    def _prepare(self, frame: torch.Tensor) -> torch.Tensor:
        work = torch.log1p(frame.to(torch.float32).clamp_min(0.0))
        if self.downsample > 1:
            work = F.interpolate(work, size=self._working_size(), mode="bilinear",
                                 align_corners=False)
        return _gaussian_blur2d(work, self.preprocess_sigma)

    def _estimate_dense_flow(self, prev_frame, curr_frame):
        prev_w, curr_w = self._prepare(prev_frame), self._prepare(curr_frame)
        hh, ww = prev_w.shape[-2:]
        curr_rain = curr_frame.to(torch.float32)
        if curr_rain.shape[-2:] != (hh, ww):
            curr_rain = F.interpolate(curr_rain, size=(hh, ww), mode="bilinear",
                                      align_corners=False)
        rain_mask = curr_rain.ge(self.rain_threshold)

        costs, dxs, dys = [], [], []
        radius = self.patch_size // 2
        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                cost = (prev_w - _shift_with_border(curr_w, dx, dy)).square()
                if radius > 0:
                    cost = F.avg_pool2d(cost, self.patch_size, stride=1, padding=radius)
                costs.append(cost); dxs.append(dx); dys.append(dy)

        best = torch.cat(costs, dim=1).argmin(dim=1)
        dxl = torch.tensor(dxs, device=best.device, dtype=prev_w.dtype)
        dyl = torch.tensor(dys, device=best.device, dtype=prev_w.dtype)
        fx, fy = dxl[best].unsqueeze(1), dyl[best].unsqueeze(1)
        if rain_mask.shape != fx.shape:
            rain_mask = rain_mask.expand_as(fx)
        fx, fy = fx * rain_mask, fy * rain_mask
        if self.preprocess_sigma > 0:
            fx = _gaussian_blur2d(fx, self.preprocess_sigma) * rain_mask
            fy = _gaussian_blur2d(fy, self.preprocess_sigma) * rain_mask
        fx = F.interpolate(fx, size=(self.y_dim, self.x_dim), mode="bilinear",
                           align_corners=False) * (self.x_dim / ww)
        fy = F.interpolate(fy, size=(self.y_dim, self.x_dim), mode="bilinear",
                           align_corners=False) * (self.y_dim / hh)
        return fx, fy

    def _advect(self, field, fx, fy):
        base = self._base_grid.to(device=field.device, dtype=field.dtype)
        grid = torch.empty((field.shape[0], self.y_dim, self.x_dim, 2),
                           device=field.device, dtype=field.dtype)
        xs = 0.0 if self.x_dim <= 1 else 2.0 / (self.x_dim - 1)
        ys = 0.0 if self.y_dim <= 1 else 2.0 / (self.y_dim - 1)
        grid[..., 0] = base[..., 0] - fx[:, 0] * xs
        grid[..., 1] = base[..., 1] - fy[:, 0] * ys
        return F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                             align_corners=True).clamp_min(0.0)

    def forward(self, pred, target, reference=None):
        reference = target if reference is None else reference
        with torch.no_grad():
            try:
                fx, fy = self._estimate_dense_flow(reference.float(), target.float())
                teacher = self._advect(reference.float(), fx, fy).to(pred.dtype)
            except Exception:
                teacher = target.to(pred.dtype)
        if self.loss_type == "l1":
            out = torch.abs(pred - teacher)
        elif self.loss_type == "huber":
            out = F.huber_loss(pred, teacher, reduction="none", delta=self.delta)
        else:
            raise ValueError(f"unsupported optical-flow loss_type: {self.loss_type}")
        if self.mask_threshold is not None:
            mask = teacher.gt(self.mask_threshold).logical_or(target > self.mask_threshold)
            out = out * mask.to(out.dtype)
        return out.mean()


# ----------------------------------------------------------------------------------
# uniform interface for the trainer
# ----------------------------------------------------------------------------------
class StructuralLoss(nn.Module):
    """Adapter giving every structural loss one call signature.

    space: "norm" (log-normalised network space) or "phys" (mm/h).
    """

    def __init__(self, fn: nn.Module, name: str, space: str,
                 needs_gamma: bool = False, needs_reference: bool = False):
        super().__init__()
        self.fn = fn
        self.name = name
        self.space = space
        self.needs_gamma = needs_gamma
        self.needs_reference = needs_reference

    def forward(self, pred_norm, target_norm, pred_phys, target_phys,
                gamma_target=None, reference_phys=None, anneal_factor: float = 0.05):
        if self.needs_gamma:
            return self.fn(pred_phys, gamma_target, anneal_factor=anneal_factor)
        pred = pred_phys if self.space == "phys" else pred_norm
        target = target_phys if self.space == "phys" else target_norm
        if self.needs_reference:
            return self.fn(pred, target, reference=reference_phys)
        return self.fn(pred, target)


def build_structural_loss(name: str, config: dict, device) -> StructuralLoss:
    """Construct the auxiliary structural loss named by STRUCTURAL_LOSS in the config.

    "minkowski" reproduces the existing behaviour exactly, so an unset key changes nothing.
    """
    name = (name or "minkowski").lower()
    drizzle = float(config.get("DRIZZLE_THRESHOLD", 0.1))
    patch = int(config.get("PATCH_SIZE", 128))

    if name == "minkowski":
        from src.losses.minkowski import AnalyticalMinkowskiLoss
        from src.utils import load_physical_thresholds, load_persistence_thresholds
        topo = config.get("TOPOLOGY_MODE", "euler")
        fn = AnalyticalMinkowskiLoss(
            physical_thresholds=load_physical_thresholds(config),
            quantile_levels=config.get("QUANTILE_LEVELS"),
            pixel_size_km=config.get("PIXEL_SIZE_KM", 2.0),
            topology_mode=topo,
            area_mode="ste",
            persistence_thresh_b0=(load_persistence_thresholds(config)[0]
                                   if topo == "b0" else 0.0),
        )
        return StructuralLoss(fn, "minkowski", "phys", needs_gamma=True).to(device)

    if name == "spectral":
        fn = FFT2DKernelCRPSLoss(alpha=float(config.get("SPECTRAL_ALPHA", 1.0)))
        return StructuralLoss(fn, "spectral", "norm").to(device)

    if name == "ssim":
        fn = ImageSSIMLoss(window_size=int(config.get("SSIM_WINDOW", 11)))
        return StructuralLoss(fn, "ssim", "norm").to(device)

    if name == "wetarea":
        # Physical space: the threshold is the drizzle level in mm/h. The temperature is
        # half the threshold, so a dry pixel scores ~0.12 wet rather than 0.5.
        fn = WeightedSoftWetAreaLoss(
            threshold=drizzle,
            temperature=float(config.get("WETAREA_TEMPERATURE", 0.5 * drizzle)),
            false_positive_weight=float(config.get("WETAREA_FP_WEIGHT", 1.0)),
            false_negative_weight=float(config.get("WETAREA_FN_WEIGHT", 1.0)),
        )
        return StructuralLoss(fn, "wetarea", "phys").to(device)

    if name == "opticalflow":
        fn = OpticalFlowConsistencyLoss(
            x_dim=patch, y_dim=patch,
            patch_size=int(config.get("OF_PATCH_SIZE", 21)),
            search_radius=int(config.get("OF_SEARCH_RADIUS", 4)),
            downsample=int(config.get("OF_DOWNSAMPLE", 4)),
            rain_threshold=drizzle,
            mask_threshold=float(config.get("OF_MASK_THRESHOLD", drizzle)),
        )
        return StructuralLoss(fn, "opticalflow", "phys", needs_reference=True).to(device)

    raise ValueError(
        f"unknown STRUCTURAL_LOSS {name!r}; expected one of: "
        "minkowski, spectral, ssim, wetarea, opticalflow"
    )


# Study-1 training weights (lambda, the maximum after warm-up), chosen 2026-09-23. Each gives
# the auxiliary term the same share of the AdamW update as Minkowski has at its validated
# 1e-4 (active, at equilibrium, not hacking; DECISIONS §2), measured at the vanilla
# checkpoint with the checkpoint's own Adam second moments (tools/gradient_audit.py,
# eval_results/gradient_audit/2026-09-23; notes/gradient_audit.pdf appendix). They are the
# centres of the M4 bracket, not validated optima. SSIM must be recalibrated after its
# data-range/eps fix, which changes its gradient scale.
DEFAULT_STRUCTURAL_WEIGHTS = {
    "minkowski": 1e-4,     # validated working point; 1e-3 reward-hacks
    "spectral": 1e-1,      # Adam-matched 0.116 (was 5e-4: a bystander)
    "ssim": 2e-2,          # Adam-matched 0.0196 (was 1.6e-4: a bystander)
    "wetarea": 2e-2,       # Adam-matched 0.0198: already at the matched share
    "opticalflow": 2.5e-2,  # Adam-matched 0.0239 (was 4e-5: inert)
}


def resolve_structural_weight(config: dict, cli_weight=None) -> tuple:
    """The training weight for the configured STRUCTURAL_LOSS, and where it came from.

    Order: command line > config STRUCTURAL_LOSS_WEIGHTS[name] > (Minkowski only)
    MINKOWSKI_TARGET_WEIGHT > DEFAULT_STRUCTURAL_WEIGHTS[name]. MINKOWSKI_TARGET_WEIGHT is
    never applied to another loss: weights differ by orders of magnitude between losses.
    """
    name = (config.get("STRUCTURAL_LOSS") or "minkowski").lower()
    if cli_weight is not None:
        return float(cli_weight), "command line"
    table = config.get("STRUCTURAL_LOSS_WEIGHTS") or {}
    if name in table:
        return float(table[name]), "STRUCTURAL_LOSS_WEIGHTS"
    if name == "minkowski" and "MINKOWSKI_TARGET_WEIGHT" in config:
        return float(config["MINKOWSKI_TARGET_WEIGHT"]), "MINKOWSKI_TARGET_WEIGHT"
    if name in DEFAULT_STRUCTURAL_WEIGHTS:
        return DEFAULT_STRUCTURAL_WEIGHTS[name], "DEFAULT_STRUCTURAL_WEIGHTS"
    raise KeyError(f"no training weight known for structural loss {name!r}")

