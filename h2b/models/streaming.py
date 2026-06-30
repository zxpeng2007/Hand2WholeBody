"""Causal online generation wrapper around the M4 diffusion model.

Maintains a short sliding window of the most recent hand frames (world frame). Each step it
canonicalizes the window by its first-frame hand position (the inference-time anchor, matching
training -- see dataset.canonicalize_window), DDIM-samples the body over the window, and emits
the newest body frame(s) de-canonicalized back to world. The causal denoiser guarantees the
latest frame depends only on frames <= t, so this is valid online.

Two modes:
  * push(hand)            -> emit the single newest body frame (1 DDIM sample / output frame).
  * push_block(hand_blk)  -> append a block of B frames, DDIM-sample the window ONCE, emit the
                             newest B body frames. The sample is amortized over the block, so
                             per-frame cost drops ~B x -- the headroom for the downstream
                             GMR + HoloMotion stages.

Window/latency/quality (measured, RTX 5080, trained model):
  * Cost is launch-bound: the per-sample time is ~flat for window 5..32 (~5 ms at ddim=2 warm),
    so a LARGER window is effectively free. With block=4 that is ~1.2 ms/output-frame (~4% of
    the 133 ms 4-frame budget at 30 fps).
  * Quality, though, depends on the window: a very short window is jerky at the block seams
    (jitter ~12 at w=5 vs ~6 at w=16 vs offline ~3.6); wrist tracking stays tight (~8 mm) until
    the window grows long enough to re-introduce anchor drift (>~20). Sweet spot ~window=16.
  * Output smoothing: a 1-Euro filter on the WHOLE body lags the extended wrist badly
    (8 mm -> 100 mm+), so that was rejected. But the LEGS are unconstrained (no hand info) and are
    the jitteriest joints (~9.5 vs 6.4 whole-body), and GMR carries that straight to the G1's legs.
    The legs are a separate kinematic chain from the wrist (pelvis->leg vs pelvis->spine->arm->wrist),
    so a causal 1-Euro filter on the LEG joints only kills the leg shake at zero latency and leaves
    wrist tracking mathematically untouched -- that is `smooth_legs` below (on by default).

Speed pass (investigated 2026-06-30, RTX 5080 laptop / torch 2.11 / Windows):
  * The streamer is LAUNCH-bound -- per-sample time is ~flat for window 5..32 -- so a classic
    KV-cache (which cuts attention FLOPs) would NOT help. The realized win is the block emission
    above (one DDIM sample serves B frames -> ~B x fewer samples).
  * The launch-overhead killers are blocked on THIS stack: torch.compile(reduce-overhead) needs
    Triton (absent on Windows), and CUDA-graph capture of nn.TransformerEncoder is rejected
    (cudaErrorStreamCaptureInvalidated, even a single forward). Both are viable on a Linux deploy
    (the robot target), so they stay as deploy-time options, not code here.
  * ddim_sample was made host-sync-free (no int(t) per step) so it pipelines and is graph/compile
    -capturable where those work. On this box each push is already ~5 ms (ddim=2) = ~4% of the
    133 ms block budget, i.e. ~96% headroom -- at the practical hardware floor.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..representations import frames as F
from ..representations import body as B

# SMPL leg chain (hips, knees, ankles, feet) — disjoint from the pelvis->spine->arm->wrist chain,
# so smoothing these never moves the FK wrist. 135-D layout: trans(0:3) then 22 joints x 6D.
_LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
_LEG_COLS = np.array([3 + j * 6 + k for j in _LEG_JOINTS for k in range(6)])


class OneEuroFilter:
    """Causal 1-Euro filter (Casiez et al. 2012), per channel. Adaptive cutoff: smooths hard when
    slow (kills jitter), opens up when fast (no lag on real motion). No lookahead -> zero latency."""

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0, fps=30.0):
        self.min_cutoff, self.beta, self.d_cutoff, self.fps = min_cutoff, beta, d_cutoff, float(fps)
        self.x_prev = None
        self.dx_prev = None

    def reset(self):
        self.x_prev = self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.fps
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.fps
        a_d = self._alpha(self.d_cutoff)
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx_prev))   # per-channel
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev


class DiffusionStreamer:
    """Online hand->body generator. push() one frame, or push_block() a block of frames."""

    def __init__(self, model, diffusion, window: int = 16, block: int = 4,
                 sample_steps: int = 2, device="cpu", smooth_legs: bool = True,
                 leg_min_cutoff: float = 0.6, leg_beta: float = 0.1, fps: float = 30.0):
        self.model = model
        self.diff = diffusion
        self.window = window
        self.block = block
        self.sample_steps = sample_steps
        self.device = device
        self._hand = deque(maxlen=window)
        # causal leg-only smoother (kills G1 leg shake; wrist untouched). None disables.
        self._leg = OneEuroFilter(leg_min_cutoff, leg_beta, fps=fps) if smooth_legs else None

    def reset(self):
        self._hand.clear()
        if self._leg is not None:
            self._leg.reset()

    def _smooth(self, frame):
        """Causally smooth the leg-joint channels of one emitted (135,) frame, in place."""
        if self._leg is not None:
            frame[_LEG_COLS] = self._leg(frame[_LEG_COLS])
        return frame

    def _sample_window(self):
        """DDIM-sample the buffered window -> (L,135) world body for the whole window."""
        import torch
        hand = np.stack(self._hand)[None]                       # (1, L, 12)
        anchor = hand[:, 0:1, F.HAND12_POS].copy()              # (1,1,3) inference anchor
        hand_c = hand.copy()
        hand_c[..., F.HAND12_POS] -= anchor
        ht = torch.from_numpy(hand_c).to(self.device)
        body = self.diff.ddim_sample(self.model, (1, ht.shape[1], B.MOTION_DIM), ht,
                                     steps=self.sample_steps, device=self.device).cpu().numpy()[0]
        body[:, B.B_TRANS] += anchor[0, 0]                      # de-canonicalize trans -> world
        return body                                            # (L, 135)

    def push_block(self, hand_block):
        """Append a block of B world-frame 12D samples, DDIM-sample the window ONCE, and return
        the newest min(B, buffered) body frames (B,135) world. One sample serves the whole block
        -> ~B x lower per-frame cost. Returns None only if nothing is buffered."""
        hb = np.asarray(hand_block, np.float32).reshape(-1, 12)
        for h in hb:
            self._hand.append(h)
        if not self._hand:
            return None
        body = self._sample_window()
        k = min(hb.shape[0], body.shape[0])
        out = body[-k:].copy()                                 # (k, 135)
        for i in range(k):                                     # smooth legs in temporal order
            self._smooth(out[i])
        return out

    def push(self, hand12_world):
        """Single-frame online step: append one 12D frame, re-sample, emit the latest body
        frame (135,). Returns None until >=2 frames are buffered (needs a little context)."""
        self._hand.append(np.asarray(hand12_world, np.float32))
        if len(self._hand) < 2:
            return None
        return self._smooth(self._sample_window()[-1].copy())
