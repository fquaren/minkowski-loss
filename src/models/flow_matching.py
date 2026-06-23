"""
Conditional flow matching for residual super-resolution downscaling.

Linear-interpolant (rectified-flow) form: x_t = (1-t) x_0 + t x_1 with target
velocity u = x_1 - x_0; a network v_theta(x_t, t, c) regresses u under MSE.
Sampling integrates the probability-flow ODE dx/dt = v_theta from t=0 to t=1.

In the corrective (CorrDiff-style) configuration the field x_1 supplied here is the
*standardised residual* r = (x_hr - mu(c)) / sigma_r, where mu is a pretrained
deterministic backbone; the trainer handles mu and the (de)standardisation. The core
below is agnostic to that and operates on whatever x_1 it is given.

References:
  Lipman et al., Flow Matching for Generative Modeling, ICLR 2023, arXiv:2210.02747.
  Liu et al., Flow Straight and Fast (rectified flow), ICLR 2023, arXiv:2209.03003.
  Albergo & Vanden-Eijnden, stochastic interpolants, arXiv:2303.08797.
  Mardani et al., CorrDiff, Commun. Earth Environ. 6, 124 (2025),
    doi:10.1038/s43247-025-02042-5 (residual decomposition).
"""

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.models.ddpm import ContextUnet


def integrate_pf_ode(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int = 16,
    method: str = "heun",
) -> torch.Tensor:
    """Integrate dx/dt = v_fn(x, t) from t=0 to t=1 with a fixed step count.

    v_fn takes (x [B,...], t [B]) and returns the velocity. 'euler' is first order;
    'heun' is second order (one extra evaluation per step) and the recommended choice.
    """
    if method not in ("euler", "heun"):
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = i * dt
        t = torch.full((x.shape[0],), t0, device=x.device, dtype=x.dtype)
        v = v_fn(x, t)
        if method == "euler":
            x = x + dt * v
        else:  # heun (explicit trapezoidal)
            x_euler = x + dt * v
            t1 = torch.full_like(t, t0 + dt)
            v1 = v_fn(x_euler, t1)
            x = x + 0.5 * dt * (v + v1)
    return x


class FlowMatching(nn.Module):
    """Conditional flow-matching wrapper around a velocity network.

    Parameters
    ----------
    net : nn.Module, optional
        Velocity network with signature net(x, t, condition). Defaults to ContextUnet.
    in_channels : int
        Channels of the field being generated (default 1).
    c_in_condition : int
        Conditioning channels. Default 3 = [interp, dem, mu] for the CorrDiff residual;
        use 2 = [interp, dem] if not conditioning on the mean.
    time_scale : float
        Continuous t in [0,1] is multiplied by this before the network's sinusoidal time
        embedding, whose base (10000) is tuned for large integer diffusion steps; without
        rescaling the embedding of t in [0,1] collapses. Default 1000.
    """

    def __init__(
        self,
        net: Optional[nn.Module] = None,
        in_channels: int = 1,
        c_in_condition: int = 3,
        time_scale: float = 1000.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.net = net if net is not None else ContextUnet(
            in_channels=in_channels, c_in_condition=c_in_condition, device=device
        )
        self.time_scale = float(time_scale)

    def velocity(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """v_theta(x_t, t, c); t is [B] in [0,1] and is rescaled for the embedding."""
        return self.net(x, t * self.time_scale, c)

    def training_loss(self, x1: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE: ||v_theta(x_t,t,c) - (x_1 - x_0)||^2 over random t, x_0."""
        b = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=x1.device, dtype=x1.dtype)
        tb = t.view(b, 1, 1, 1)
        xt = (1.0 - tb) * x0 + tb * x1
        u = x1 - x0
        v = self.velocity(xt, t, c)
        return ((v - u) ** 2).mean()

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        n_steps: int = 16,
        method: str = "heun",
        x0: Optional[torch.Tensor] = None,
        in_channels: int = 1,
    ) -> torch.Tensor:
        """Draw a sample by integrating the probability-flow ODE from x_0 ~ N(0, I)."""
        if x0 is None:
            shape = (c.shape[0], in_channels, c.shape[2], c.shape[3])
            x0 = torch.randn(shape, device=c.device, dtype=c.dtype)
        return integrate_pf_ode(lambda x, t: self.velocity(x, t, c), x0, n_steps, method)

    def endpoint(self, xt: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Clean-field estimate x_hat_1 = x_t + (1-t) v_theta(x_t,t,c) (for the Phase-3 loss).

        Exact identity under the true velocity u = x_1 - x_0: x_t + (1-t) u = x_1.
        """
        v = self.velocity(xt, t, c)
        tb = t.view(-1, 1, 1, 1)
        return xt + (1.0 - tb) * v