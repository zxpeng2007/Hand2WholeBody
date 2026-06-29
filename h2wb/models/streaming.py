"""Causal online generation wrapper around the M4 diffusion model.

Maintains a sliding window of the most recent hand frames (world frame). Each new frame,
it canonicalizes the window by its first-frame hand position (the inference-time anchor,
matching training — see dataset.canonicalize_window), DDIM-samples the body over the
window, and emits the LATEST body frame de-canonicalized back to world. The causal
denoiser guarantees the latest frame depends only on frames <= t, so this is valid online.

This re-samples the window each step (simple + correct). A KV-cache / block-streaming
optimization (à la HoloMotion's ring buffer) is a later speed pass — TODO(perf).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..representations import frames as F
from ..representations import body as B


class DiffusionStreamer:
    """Online hand->body generator. push() one world-frame 12D, get one world-frame body."""

    def __init__(self, model, diffusion, window: int = 40, sample_steps: int = 8, device="cpu"):
        self.model = model
        self.diff = diffusion
        self.window = window
        self.sample_steps = sample_steps
        self.device = device
        self._hand = deque(maxlen=window)

    def reset(self):
        self._hand.clear()

    def push(self, hand12_world: np.ndarray):
        """Append one world-frame 12D hand sample and return the latest body frame (135,) world.

        Returns None until at least 2 frames are buffered (need a short context).
        """
        import torch
        self._hand.append(np.asarray(hand12_world, np.float32))
        if len(self._hand) < 2:
            return None
        hand = np.stack(self._hand)[None]                     # (1, L, 12)
        anchor = hand[:, 0:1, F.HAND12_POS].copy()            # (1,1,3) inference anchor
        hand_c = hand.copy()
        hand_c[..., F.HAND12_POS] -= anchor
        ht = torch.from_numpy(hand_c).to(self.device)
        shape = (1, ht.shape[1], B.MOTION_DIM)
        body_c = self.diff.ddim_sample(self.model, shape, ht, steps=self.sample_steps,
                                       device=self.device).cpu().numpy()[0]  # (L, 135)
        latest = body_c[-1].copy()
        latest[B.B_TRANS] += anchor[0, 0]                     # de-canonicalize trans -> world
        return latest
