"""
Diffusion process utilities: noise scheduling, forward diffusion, and sampling.

Implements the cosine variance schedule (Nichol & Dhariwal, 2021) and
both DDPM and DDIM (Song et al., 2020) reverse sampling.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Diffusion(nn.Module):
    """Manages the forward and reverse diffusion process.

    Parameters
    ----------
    noise_steps : int
        Total number of diffusion timesteps T.
    img_size : int
        Spatial dimension of generated images (assumes square).
    device : str
        Target device.
    """

    def __init__(self, noise_steps=1000, img_size=128, device="cuda"):
        super().__init__()
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        self.beta = self._cosine_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _cosine_schedule(self, s=0.008):
        """Cosine variance schedule (Nichol & Dhariwal, 2021)."""
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_images(self, x, t):
        """Forward diffusion: q(x_t | x_0)."""
        sqrt_ah = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_ah = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_ah * x + sqrt_one_minus_ah * eps, eps

    def sample_timesteps(self, n):
        """Sample uniform random timesteps for training."""
        return torch.randint(0, self.noise_steps, (n,), device=self.device)

    @torch.no_grad()
    def sample(self, model, n, conditions):
        """Full DDPM reverse sampling (T steps).

        Parameters
        ----------
        model : ContextUnet
        n : int
            Batch size.
        conditions : Tensor [n, C, H, W]

        Returns
        -------
        Tensor [n, 1, H, W] in [-1, 1]
        """
        model.eval()
        x = torch.randn(n, 1, self.img_size, self.img_size, device=self.device)

        for i in tqdm(reversed(range(self.noise_steps)), total=self.noise_steps,
                      leave=False):
            t = torch.full((n,), i, dtype=torch.long, device=self.device)
            pred_noise = model(x, t, conditions)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
            ) + torch.sqrt(beta) * noise

        model.train()
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_ddim(self, model, n, conditions, ddim_steps=50, eta=0.0):
        """DDIM sampling (deterministic when eta=0).

        Parameters
        ----------
        model : ContextUnet
        n : int
        conditions : Tensor [n, C, H, W]
        ddim_steps : int
            Number of sub-sampled timesteps.
        eta : float
            Stochasticity (0 = deterministic DDIM).

        Returns
        -------
        Tensor [n, 1, H, W] in [-1, 1]
        """
        model.eval()
        x = torch.randn(n, 1, self.img_size, self.img_size, device=self.device)

        # Sub-sampled timestep sequence: T → 0
        step_ratio = self.noise_steps // ddim_steps
        timesteps = list(reversed((np.arange(1, ddim_steps + 1) * step_ratio) - 1))
        timesteps.append(0)

        device_type = "cuda" if self.device == "cuda" else "cpu"

        for i in tqdm(range(len(timesteps) - 1), leave=False):
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]

            t = torch.full((n,), t_curr, dtype=torch.long, device=self.device)
            t_prev_t = torch.full((n,), t_prev, dtype=torch.long, device=self.device)

            with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                pred_noise = model(x, t, conditions)

            ah_t = self.alpha_hat[t][:, None, None, None]
            ah_prev = self.alpha_hat[t_prev_t][:, None, None, None]

            # Predict x_0
            pred_x0 = ((x - torch.sqrt(1 - ah_t) * pred_noise) / torch.sqrt(ah_t))
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            # DDIM variance
            sigma = eta * torch.sqrt(
                (1 - ah_prev) / (1 - ah_t) * (1 - ah_t / ah_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - ah_prev - sigma ** 2) * pred_noise

            # Stochastic noise (zero at final step or when eta=0)
            noise = (
                torch.randn_like(x) if t_prev > 0 and eta > 0
                else torch.zeros_like(x)
            )

            x = torch.sqrt(ah_prev) * pred_x0 + dir_xt + sigma * noise

        model.train()
        return x.clamp(-1.0, 1.0)
