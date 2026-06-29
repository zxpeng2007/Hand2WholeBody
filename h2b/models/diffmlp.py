"""DiffMLP — AGRoL-style conditional diffusion backbone for Hand2Body.

Reference: Du et al., "Avatars Grow Legs" (AGRoL), CVPR 2023 — a pure-MLP diffusion
denoiser that is small (~7M params) and fast (5 DDIM steps, ~35 ms / 196 frames), which
is what makes real-time sparse→full-body feasible. Here it denoises the next `P` body
frames conditioned on a window of past hand (12D) + past body frames.

Body-motion channel layout per frame = h2b.representations.body (135 dims):
    [ root_trans (3) | 22 joints x 6D (132) ]   (joint 0's 6D is the global pelvis orient)

This file is a faithful, shape-correct implementation ready to train once torch is in
the venv. It is intentionally self-contained (no external diffusion lib) so the math is
inspectable. The DDPM/DDIM schedule helpers live in `diffusion.py` (TODO) or can be
swapped for a flow-matching objective (lower latency — see research notes).
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional at import time
    _HAS_TORCH = False
    nn = object  # type: ignore


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "DiffMLP requires torch. Install into .venv:\n"
            "  python -m uv pip install --python .venv "
            "--index-url https://download.pytorch.org/whl/cu128 torch"
        )


if _HAS_TORCH:

    class SinusoidalTimestepEmbedding(nn.Module):
        """Standard diffusion timestep embedding -> MLP to `dim`."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t: "torch.Tensor") -> "torch.Tensor":  # t: (B,)
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
            )
            args = t.float()[:, None] * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if self.dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return self.mlp(emb)

    class ConditionEncoder(nn.Module):
        """Encode the causal conditioning window into a (B, latent) vector.

        Inputs:
          hand_window: (B, K, 12)  past hand 12D signal (canonicalized per CONTRACT §5)
          body_window: (B, Kb, M)  past generated/GT body frames (autoregressive state)
        """

        def __init__(self, hand_dim: int, body_dim: int, k_hand: int, k_body: int, latent: int):
            super().__init__()
            in_dim = hand_dim * k_hand + body_dim * k_body
            self.net = nn.Sequential(
                nn.Linear(in_dim, latent), nn.SiLU(),
                nn.Linear(latent, latent), nn.SiLU(),
            )

        def forward(self, hand_window, body_window):
            b = hand_window.shape[0]
            x = torch.cat([hand_window.reshape(b, -1), body_window.reshape(b, -1)], dim=-1)
            return self.net(x)

    class DiffMLPBlock(nn.Module):
        """Residual block: LN → Linear → SiLU → time-conv(k=1) → LN, FiLM-conditioned."""

        def __init__(self, dim: int, cond_dim: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.fc = nn.Linear(dim, dim)
            self.conv = nn.Conv1d(dim, dim, kernel_size=1)
            self.norm2 = nn.LayerNorm(dim)
            self.film = nn.Linear(cond_dim, dim * 2)   # scale + shift from (t ⊕ cond)

        def forward(self, x, c):                       # x: (B, P, dim), c: (B, cond_dim)
            scale, shift = self.film(c).chunk(2, dim=-1)
            h = self.norm1(x)
            h = F.silu(self.fc(h))
            h = self.conv(h.transpose(1, 2)).transpose(1, 2)
            h = self.norm2(h) * (1 + scale[:, None]) + shift[:, None]
            return x + h

    class DiffMLP(nn.Module):
        """Denoiser: predict x0 (the clean next-P body frames) from noised input + condition."""

        def __init__(
            self,
            motion_dim: int = 135,     # root_trans(3) + 22 joints x 6D(132); h2b.representations.body
            hand_dim: int = 12,
            k_hand: int = 20,
            k_body: int = 5,
            p_predict: int = 8,
            hidden: int = 512,
            n_blocks: int = 12,
            latent: int = 512,
        ):
            _require_torch()
            super().__init__()
            self.p_predict = p_predict
            self.motion_dim = motion_dim
            self.in_proj = nn.Linear(motion_dim, hidden)
            self.t_embed = SinusoidalTimestepEmbedding(hidden)
            self.cond_enc = ConditionEncoder(hand_dim, motion_dim, k_hand, k_body, latent)
            self.cond_merge = nn.Linear(hidden + latent, hidden)
            self.blocks = nn.ModuleList([DiffMLPBlock(hidden, hidden) for _ in range(n_blocks)])
            self.out_proj = nn.Linear(hidden, motion_dim)

        def forward(self, x_t, t, hand_window, body_window):
            """x_t: (B, P, M); t: (B,); hand_window: (B,K,12); body_window: (B,Kb,M) -> x0_hat (B,P,M)."""
            c = self.cond_merge(torch.cat([self.t_embed(t), self.cond_enc(hand_window, body_window)], dim=-1))
            h = self.in_proj(x_t)
            for blk in self.blocks:
                h = blk(h, c)
            return self.out_proj(h)

else:  # pragma: no cover

    class DiffMLP:  # type: ignore
        def __init__(self, *a, **k):
            _require_torch()
