"""Conditional diffusion for Hand2Body (M4, primary model).

Same-frame lifting like M2, but generative: denoise body[1..L] conditioned on hand[1..L].
Sampling a body (instead of regressing the mean) is what resolves the single-hand
ambiguity — the model commits to one plausible pose rather than averaging.

Pieces:
  * GaussianDiffusion — cosine schedule, q_sample (forward noising), x0-prediction loss,
    deterministic DDIM sampling (few steps for real-time; distill later if needed).
  * DiTDenoiser — a CAUSAL transformer denoiser (frame t attends <= t), conditioned per
    frame on the 12D hand and on the diffusion timestep. Predicts x0 (the clean body),
    which lets us add the geometric/hand-consistency losses directly on x0.

x0-prediction (not eps) is chosen so h2b.losses.compute_losses (FK, hand-consistency)
applies to the predicted clean motion with no reparameterization.
"""

from __future__ import annotations

import math

from ..representations.body import MOTION_DIM
from ..representations.frames import HAND12_DIM

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore


def cosine_beta_schedule(num_steps: int, s: float = 0.008):
    """Nichol & Dhariwal cosine schedule -> betas (num_steps,)."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    acp = f / f[0]
    betas = 1.0 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-8, 0.999).float()


if _HAS_TORCH:

    class TimestepEmbed(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
            a = t.float()[:, None] * freqs[None]
            emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
            if self.dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return self.mlp(emb)

    class DiTDenoiser(nn.Module):
        """Causal transformer denoiser: (x_t (B,L,M), t (B,), hand (B,L,12)) -> x0_hat (B,L,M)."""

        def __init__(self, motion_dim=MOTION_DIM, hand_dim=HAND12_DIM, hidden=256,
                     n_layers=4, n_heads=8, ffn=1024, max_len=256, dropout=0.0):
            super().__init__()
            self.in_proj = nn.Linear(motion_dim + hand_dim, hidden)
            self.t_embed = TimestepEmbed(hidden)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
            self.head = nn.Linear(hidden, motion_dim)

        def forward(self, x_t, t, hand):
            B, L, _ = x_t.shape
            h = self.in_proj(torch.cat([x_t, hand], dim=-1)) + self.pos[:, :L]
            h = h + self.t_embed(t)[:, None, :]
            mask = torch.triu(torch.ones(L, L, device=x_t.device, dtype=torch.bool), diagonal=1)
            return self.head(self.encoder(h, mask=mask))

    class GaussianDiffusion:
        """Cosine-schedule diffusion with x0-prediction and DDIM sampling."""

        def __init__(self, num_steps: int = 1000, device="cpu"):
            self.num_steps = num_steps
            betas = cosine_beta_schedule(num_steps)
            alphas = 1.0 - betas
            acp = torch.cumprod(alphas, dim=0)
            self.betas = betas.to(device)
            self.acp = acp.to(device)
            self.sqrt_acp = acp.sqrt().to(device)
            self.sqrt_one_minus_acp = (1.0 - acp).sqrt().to(device)

        def to(self, device):
            for k in ("betas", "acp", "sqrt_acp", "sqrt_one_minus_acp"):
                setattr(self, k, getattr(self, k).to(device))
            return self

        def q_sample(self, x0, t, noise):
            """Forward noising: x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) noise. t: (B,) long."""
            a = self.sqrt_acp[t].view(-1, 1, 1)
            b = self.sqrt_one_minus_acp[t].view(-1, 1, 1)
            return a * x0 + b * noise

        def sample_t(self, batch, device):
            return torch.randint(0, self.num_steps, (batch,), device=device)

        @torch.no_grad()
        def ddim_sample(self, model, shape, hand, steps: int = 8, device="cpu"):
            """Deterministic DDIM (eta=0). Returns x0_hat (shape)."""
            x = torch.randn(shape, device=device)
            ts = torch.linspace(self.num_steps - 1, 0, steps, device=device).round().long()
            x0_hat = x
            for i in range(steps):
                t = ts[i]
                tb = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
                x0_hat = model(x, tb, hand)
                acp_t = self.acp[t]
                eps = (x - acp_t.sqrt() * x0_hat) / (1.0 - acp_t).sqrt().clamp_min(1e-8)
                t_next = ts[i + 1] if i + 1 < steps else torch.tensor(0, device=device)
                acp_n = self.acp[t_next]
                x = acp_n.sqrt() * x0_hat + (1.0 - acp_n).sqrt() * eps
            return x0_hat

else:  # pragma: no cover

    class DiTDenoiser:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("DiTDenoiser requires torch")

    class GaussianDiffusion:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("GaussianDiffusion requires torch")
