"""Models for Hand2Body.

Primary:   diffmlp.DiffMLP        (AGRoL-style conditional diffusion backbone)
Streaming: streaming.StreamingHand2Body  (causal autoregressive rollout wrapper)
Fallback:  regressor.RegressorHand2Body  (deterministic transformer baseline, M2)

All modules import torch lazily-safely: importing this package does NOT require torch,
so the representation/data layers stay usable in a torch-free environment.
"""

__all__ = ["diffmlp", "streaming", "regressor"]
