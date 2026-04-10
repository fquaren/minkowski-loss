"""
Log-space residual U-Net for deterministic precipitation super-resolution.

Input:  [B, 2, H, W]  — channel 0: log-normalized interpolated precip,
                         channel 1: z-scored DEM
Output: [B, 1, H, W]  — log-normalized predicted precipitation

The model learns a residual in log-space:
    pred = clamp(interp_log + residual, min=0)

This means the DEM informs the residual but does not appear in the output.
The clamp enforces non-negative log-precipitation (physical constraint:
log1p(x) >= 0 for x >= 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two 3×3 convolutions with GroupNorm and ReLU."""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        # Ensure num_groups divides channel count
        g1 = min(num_groups, out_channels)
        while out_channels % g1 != 0:
            g1 -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LogSpaceResidualUNet(nn.Module):
    """U-Net with bilinear upsampling and log-space residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 2: precip + DEM).
    out_channels : int
        Number of output channels (default 1: precipitation).
    features : list of int
        Channel widths at each encoder stage.
    """

    def __init__(self, in_channels=2, out_channels=1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        features = list(features)

        # Encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_c = in_channels
        for f in features:
            self.downs.append(_DoubleConv(in_c, f))
            in_c = f

        # Bottleneck
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Decoder: each stage upsamples then concatenates with the skip
        self.ups = nn.ModuleList()
        prev_c = features[-1] * 2
        for f in reversed(features):
            self.ups.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
            # Input channels = upsampled (prev_c) + skip (f)
            self.ups.append(_DoubleConv(prev_c + f, f))
            prev_c = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Extract interpolated precip for residual connection
        interp_log = x[:, 0:1, :, :]

        # Encoder
        skip_connections = []
        out = x
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            out = self.ups[i](out)       # upsample
            skip = skip_connections[i // 2]
            # Handle spatial size mismatch from non-power-of-2 inputs
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(
                    out, size=skip.shape[2:],
                    mode="bilinear", align_corners=True,
                )
            out = torch.cat([skip, out], dim=1)
            out = self.ups[i + 1](out)   # double conv

        residual_log = self.final_conv(out)

        # Additive residual in log-space, clamped to [0, ∞)
        return torch.clamp(interp_log + residual_log, min=0.0)
