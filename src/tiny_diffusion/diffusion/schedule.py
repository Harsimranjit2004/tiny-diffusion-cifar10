"""
src/tiny_diffusion/diffusion/schedule.py

The cosine noise schedule — split out from Phase 1's combined file.
See Phase 0, Section 4 for the full derivation of every coefficient
computed here, and the 7 sanity checks verify_schedule() runs.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class CosineNoiseSchedule(nn.Module):
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Precomputes all coefficients needed for training and sampling.
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__()
        self.T = T

        # ── Compute cosine schedule ────────────────────────────────────────
        # Steps: 0, 1, 2, ..., T (T+1 values)
        steps = torch.arange(T + 1, dtype=torch.float64)

        # f(t) = cos((t/T + s) / (1+s) * pi/2)^2
        # The offset s=0.008 prevents beta_t from being too large at t=0.
        # Without s: at t=0, f(0)=cos(0)=1, f(1)=cos(pi/2T) — fine.
        # The s offset slightly rounds the curve near t=0.
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        # Normalize so that alphas_cumprod[0] = 1.0 exactly
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Derive betas from alphas_cumprod:
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        # alphas_cumprod[1:] = alpha_bar for t=1..T
        # alphas_cumprod[:-1] = alpha_bar for t=0..T-1
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clip to prevent numerical issues near t=T where beta can approach 1
        betas = torch.clamp(betas, min=1e-5, max=0.999).float()

        # Now alphas_cumprod[1:] are the values for t=1..T (what we actually use)
        alphas_cumprod = alphas_cumprod[1:].float()

        # alpha_t = 1 - beta_t (single-step signal retention)
        alphas = 1.0 - betas

        # ── Register all coefficients as buffers ───────────────────────────
        # Shape of each buffer: [T] (one value per timestep)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # alpha_bar_{t-1}: needed for the reverse process posterior.
        # At t=1, alpha_bar_{t-1} = alpha_bar_0 = 1.0 (no noise before start).
        # We prepend 1.0 and take all but the last element.
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # ── Forward process coefficients ───────────────────────────────────
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))

        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # ── Reverse process coefficients ───────────────────────────────────
        # Used in DDPM sampling step:
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred)
        #           + sqrt(beta_t) * z

        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # beta_t / sqrt(1 - alpha_bar_t): the coefficient of eps_pred in the mean
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # ── Posterior q(x_{t-1} | x_t, x_0) coefficients ─────────────────
        # posterior_variance = beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # Log variance, clipped to prevent log(0).
        # We use log variance in the ELBO computation.
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20))
        )

        # Coefficients for the posterior mean:
        # mu_tilde_t = coef1 * x_0 + coef2 * x_t
        self.register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract values from array `arr` at timestep indices `t`,
        then reshape for broadcasting against tensors of shape x_shape.

        Example: arr=[1000 values], t=[B], x_shape=[B,3,32,32]
        Returns: [B, 1, 1, 1] — broadcasts against [B, 3, 32, 32]
        """
        # arr[t] indexes each of the B timesteps
        out = arr[t]  # [B]
        # Reshape to [B, 1, 1, ..., 1] with len(x_shape)-1 trailing dims
        # This allows broadcasting: scalar-per-sample × spatial tensor
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t given x_0 and timestep t.

        Math: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        This is the closed-form we derived in Phase 0, Section 1.

        Args:
            x_start: clean images x_0, shape [B, C, H, W]
            t:       timestep indices, shape [B], values in [0, T-1]
            noise:   optional pre-sampled noise (for reproducibility in tests)
        Returns:
            x_t:     noisy images at timestep t, shape [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Extract per-sample coefficients and broadcast to image shape
        sqrt_abar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_abar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_abar * x_start + sqrt_one_minus_abar * noise

    def verify_schedule(self) -> None:
        """Sanity checks from Phase 0 Section 4. Call once after init."""
        assert self.alphas_cumprod[0] < 1.0, "alpha_bar_0 should be < 1"
        assert self.alphas_cumprod[0] > 0.9, "alpha_bar_0 should be close to 1"
        assert self.alphas_cumprod[-1] < 0.02, "alpha_bar_T should be near 0"
        assert (self.betas > 0).all(), "all betas must be positive"
        assert (self.betas < 1).all(), "all betas must be < 1"
        # Monotonically decreasing
        assert (
            self.alphas_cumprod[1:] < self.alphas_cumprod[:-1]
        ).all(), "alpha_bar must be monotonically decreasing"
        print("  [Schedule] All sanity checks passed.")
        print(f"  alpha_bar_1   = {self.alphas_cumprod[0].item():.4f}  (should be ~0.9999)")
        print(f"  alpha_bar_500 = {self.alphas_cumprod[499].item():.4f} (should be ~0.10-0.30)")
        print(f"  alpha_bar_T   = {self.alphas_cumprod[-1].item():.6f} (should be <0.01)")
