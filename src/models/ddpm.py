"""
Conditional U-Net for denoising diffusion probabilistic models (DDPM).

Architecture: encoder-decoder with time embedding injection,
self-attention at 32×32 and 16×16 resolution, and skip connections.

GroupNorm with num_groups=1 is used throughout, which is equivalent
to LayerNorm (normalizes over all channels jointly).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention on spatial features."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True,
        )
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        x_norm = self.ln(x_flat)
        attn, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        return x + attn.permute(0, 2, 1).view(B, C, H, W)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 residual=False):
        super().__init__()
        self.residual = residual
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.block(x))
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, in_ch, residual=True),
            _DoubleConv(in_ch, out_ch),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].expand_as(x)
        return x + emb


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch, in_ch // 2)
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # Pad to match skip connection size
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].expand_as(x)
        return x + emb


class ContextUnet(nn.Module):
    """Conditional U-Net for DDPM with time and spatial conditioning.

    Parameters
    ----------
    in_channels : int
        Channels of the noisy target (default 1 for precipitation).
    c_in_condition : int
        Channels of the conditioning input (default 2: precip + DEM).
    time_dim : int
        Dimensionality of sinusoidal time embeddings.
    """

    def __init__(self, in_channels=1, c_in_condition=2, time_dim=256,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        total_in = in_channels + c_in_condition

        # Encoder
        self.inc = _DoubleConv(total_in, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.sa1 = SelfAttention(256)
        self.down3 = _Down(256, 512)
        self.sa2 = SelfAttention(512)

        # Bottleneck
        self.bot1 = _DoubleConv(512, 1024)
        self.sa3 = SelfAttention(1024)
        self.bot2 = _DoubleConv(1024, 1024)
        self.bot3 = _DoubleConv(1024, 512)

        # Decoder
        self.up1 = _Up(768, 256)   # 512 + 256 skip
        self.sa4 = SelfAttention(256)
        self.up2 = _Up(384, 128)   # 256 + 128 skip
        self.up3 = _Up(192, 64)    # 128 + 64 skip
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def _pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pe_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pe_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pe_sin, pe_cos], dim=-1)

    def forward(self, x, t, condition):
        t = self._pos_encoding(t.unsqueeze(-1).float(), self.time_dim)
        x = torch.cat([x, condition], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.sa1(self.down2(x2, t))
        x4 = self.sa2(self.down3(x3, t))

        x4 = self.bot3(self.bot2(self.sa3(self.bot1(x4))))

        x = self.sa4(self.up1(x4, x3, t))
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        return self.outc(x)
