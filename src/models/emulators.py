"""
Emulator architectures for Minkowski functional prediction.

Three models in an ablation hierarchy:
  - BaselineCNN: standard CNN, independent heads, no constraints
  - LipschitzCNN: spectral norm + GroupNorm + residual blocks
  - ConstrainedLipschitzCNN: adds monotonicity, isoperimetric, positivity

All models map a normalized precipitation field [B, 1, H, W] to a
gamma vector [B, 3, Q] containing (area, perimeter, topology) at Q
physical thresholds.

The `topology_mode` parameter controls the third output:
  - "b0": connected components (non-negative, via softplus)
  - "euler": Euler characteristic = B0 - B1 (signed, unconstrained)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------

class RobustBlock(nn.Module):
    """Pre-activation residual block with GroupNorm and spectral norm.

    GroupNorm is preferred over BatchNorm for physical emulators as
    it is independent of batch statistics, which can be unstable
    with sparse precipitation data.
    """

    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(x))
        out = self.conv1(out)
        out = F.gelu(self.gn2(out))
        out = self.conv2(out)
        return residual + out


# ---------------------------------------------------------------
# Model 1: Baseline CNN (unconstrained)
# ---------------------------------------------------------------

class BaselineCNN(nn.Module):
    """Standard CNN baseline with independent regression heads.

    Uses ReLU, BatchNorm, and global average pooling. No Lipschitz
    constraints or geometric coupling between heads.
    """

    def __init__(self, n_quantiles=30, input_shape=(1, 128, 128),
                 topology_mode="b0"):
        super().__init__()
        self.topology_mode = topology_mode

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head_A = nn.Linear(256, n_quantiles)
        self.head_P = nn.Linear(256, n_quantiles)
        self.head_T = nn.Linear(256, n_quantiles)

    def forward(self, x):
        feat = self.features(x).flatten(1)

        pred_A = F.relu(self.head_A(feat))
        pred_P = F.relu(self.head_P(feat))

        if self.topology_mode == "b0":
            pred_T = F.relu(self.head_T(feat))
        else:
            pred_T = self.head_T(feat)  # unconstrained for Euler

        return torch.stack([pred_A, pred_P, pred_T], dim=1)


# ---------------------------------------------------------------
# Model 2: Lipschitz CNN (unconstrained heads)
# ---------------------------------------------------------------

class LipschitzCNN(nn.Module):
    """Lipschitz-regularized CNN with independent regression heads.

    Uses spectral normalization on all convolutions (except the entry
    layer, which is left unconstrained to allow unrestricted input
    scaling — see Section 2.3.2 of the paper), GroupNorm, GELU
    activations, and residual connections.
    """

    def __init__(self, n_quantiles=30, input_shape=(1, 128, 128),
                 topology_mode="b0"):
        super().__init__()
        self.topology_mode = topology_mode

        # Entry conv: intentionally NOT spectrally normalized
        self.entry = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = RobustBlock(32)

        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(32, 64, 3, stride=1, padding=1)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = RobustBlock(64)

        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, 3, stride=1, padding=1)
        )
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = RobustBlock(128)

        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(128, 256, 3, stride=1, padding=1)
        )
        self.pool4 = nn.MaxPool2d(2)
        self.res4 = RobustBlock(256)

        self.res5 = RobustBlock(256)
        self.fc = nn.Linear(256, 256)

        self.head_A = nn.Linear(256, n_quantiles)
        self.head_P = nn.Linear(256, n_quantiles)
        self.head_T = nn.Linear(256, n_quantiles)

    def forward(self, x):
        x = F.gelu(self.entry(x))
        x = self.res1(self.pool1(x))

        x = F.gelu(self.conv2(x))
        x = self.res2(self.pool2(x))

        x = F.gelu(self.conv3(x))
        x = self.res3(self.pool3(x))

        x = F.gelu(self.conv4(x))
        x = self.res4(self.res5(self.pool4(x)))

        feat = x.mean(dim=(2, 3))
        latent = F.gelu(self.fc(feat))

        pred_A = F.softplus(self.head_A(latent))
        pred_P = F.softplus(self.head_P(latent))

        if self.topology_mode == "b0":
            pred_T = F.softplus(self.head_T(latent))
        else:
            pred_T = self.head_T(latent)

        return torch.stack([pred_A, pred_P, pred_T], dim=1)


# ---------------------------------------------------------------
# Model 3: Constrained Lipschitz CNN
# ---------------------------------------------------------------

class ConstrainedLipschitzCNN(nn.Module):
    """Lipschitz CNN with geometrically constrained output heads.

    Area head: sigmoid(total) * max_area → softmax → reverse cumsum.
        Enforces A(t_i) >= A(t_{i+1}) by construction.

    Perimeter head: P = sqrt(4*pi*A) * (1 + softplus(r)).
        Enforces isoperimetric inequality P^2 >= 4*pi*A.

    Topology head:
        "b0" mode: softplus (non-negative counting measure).
        "euler" mode: unconstrained linear (chi = B0 - B1 can be negative).

    Parameters
    ----------
    n_quantiles : int
        Number of threshold levels Q.
    input_shape : tuple
        (C, H, W) of input patches.
    quantile_levels : list
        Physical thresholds (unused in forward, stored for reference).
    pixel_area_km2 : float
        Physical area of a single pixel in km².
    topology_mode : str
        "b0" or "euler".
    """

    def __init__(
        self,
        n_quantiles=30,
        input_shape=(1, 128, 128),
        quantile_levels=None,
        pixel_area_km2=4.0,
        topology_mode="b0",
    ):
        super().__init__()
        if quantile_levels is None:
            raise ValueError("quantile_levels is required for ConstrainedLipschitzCNN.")

        self.topology_mode = topology_mode
        self.n_quantiles = n_quantiles

        max_physical_area = input_shape[1] * input_shape[2] * pixel_area_km2
        self.register_buffer(
            "max_total_area",
            torch.tensor(max_physical_area, dtype=torch.float32),
        )

        # --- Backbone (identical to LipschitzCNN) ---
        self.entry = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = RobustBlock(32)

        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(32, 64, 3, stride=1, padding=1)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = RobustBlock(64)

        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, 3, stride=1, padding=1)
        )
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = RobustBlock(128)

        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(128, 256, 3, stride=1, padding=1)
        )
        self.pool4 = nn.MaxPool2d(2)
        self.res4 = RobustBlock(256)

        self.res5 = RobustBlock(256)
        self.fc = nn.Linear(256, 256)

        # --- Constrained heads ---
        # Area: total area scalar + probability distribution over thresholds
        self.head_A_total = nn.Linear(256, 1)
        self.head_A_logits = nn.Linear(256, n_quantiles)

        # Perimeter: roughness coefficient conditioned on area
        self.head_P_roughness = nn.Linear(256, n_quantiles)

        # Topology
        self.head_T = nn.Linear(256, n_quantiles)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Bias area head toward small initial predictions
        nn.init.constant_(self.head_A_total.weight, 0)
        nn.init.constant_(self.head_A_total.bias, -2.0)
        nn.init.constant_(self.head_A_logits.weight, 0)
        nn.init.constant_(self.head_A_logits.bias, 0)

    def forward(self, x):
        eps = 1e-6

        # --- Backbone ---
        x = F.gelu(self.entry(x))
        x = self.res1(self.pool1(x))

        x = F.gelu(self.conv2(x))
        x = self.res2(self.pool2(x))

        x = F.gelu(self.conv3(x))
        x = self.res3(self.pool3(x))

        x = F.gelu(self.conv4(x))
        x = self.res4(self.res5(self.pool4(x)))

        feat = x.mean(dim=(2, 3))
        latent = F.gelu(self.fc(feat))

        # --- Area (monotonicity via reverse cumulative sum) ---
        pred_total_area = (
            torch.sigmoid(self.head_A_total(latent)) * self.max_total_area
        )
        probs = F.softmax(self.head_A_logits(latent), dim=1)
        pdf_scaled = probs * (pred_total_area + eps)
        # Reverse cumsum: A(t_i) = sum_{j >= i} pdf_j
        pred_A = torch.flip(
            torch.cumsum(torch.flip(pdf_scaled, dims=[1]), dim=1),
            dims=[1],
        )

        # --- Perimeter (isoperimetric inequality) ---
        P_min = torch.sqrt(4.0 * math.pi * (pred_A + eps))
        roughness = 1.0 + F.softplus(self.head_P_roughness(latent))
        pred_P = P_min * roughness

        # --- Topology ---
        if self.topology_mode == "b0":
            pred_T = F.softplus(self.head_T(latent))
        else:
            pred_T = self.head_T(latent)

        return torch.stack([pred_A, pred_P, pred_T], dim=1)
