"""Deterministic causal transformer regressor — the M2 baseline / latency floor.

Maps a hand signal sequence hand[1..L] (each frame the 12D vector) to a body motion
sequence body[1..L] (each frame the 135-D vector, see h2b.representations.body), with a
CAUSAL self-attention mask so frame t only attends to frames <= t. This makes it directly
usable in the streaming/online setting (CONTRACT §4).

AvatarPoser/AvatarJLM-style: fast, stable, deterministic. It regresses the most-likely
body, so it averages in ambiguous phases — the known single-hand failure mode. It is the
baseline and the speed floor, not the final model (that is the diffusion model, M4).
"""

from __future__ import annotations

from ..representations.body import MOTION_DIM
from ..representations.frames import HAND12_DIM

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore


if _HAS_TORCH:

    class RegressorHand2Body(nn.Module):
        """Causal Transformer encoder: (B, L, 12) hand -> (B, L, 135) body."""

        def __init__(
            self,
            hand_dim: int = HAND12_DIM,
            motion_dim: int = MOTION_DIM,
            hidden: int = 256,
            n_layers: int = 4,
            n_heads: int = 8,
            ffn: int = 1024,
            max_len: int = 256,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.in_proj = nn.Linear(hand_dim, hidden)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=ffn,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
            self.head = nn.Linear(hidden, motion_dim)

        def forward(self, hand):                       # hand: (B, L, 12) -> (B, L, 135)
            B, L, _ = hand.shape
            h = self.in_proj(hand) + self.pos[:, :L]
            mask = torch.triu(torch.ones(L, L, device=hand.device, dtype=torch.bool), diagonal=1)
            h = self.encoder(h, mask=mask)             # causal: frame t sees <= t
            return self.head(h)

else:  # pragma: no cover

    class RegressorHand2Body:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("RegressorHand2Body requires torch")
