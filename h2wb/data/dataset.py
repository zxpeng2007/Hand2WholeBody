"""Windowed causal dataset for Hand2WholeBody training.

Given a per-clip pair (hand12 (T,12), body (T,M)), emit training samples that match the
streaming inference contract (CONTRACT §4/§5):

    sample t (valid for t in [K-1, T-P]):
        cond_hand  = hand12[t-K+1 : t+1]            (K past hand frames, incl. current)
        cond_body  = body[t-Kb+1 : t+1]             (Kb past body frames, autoregressive)
        target     = body[t+1 : t+1+P]              (P future body frames to predict)

`make_causal_windows` is pure NumPy and unit-tested. `Hand2BodyDataset` is a thin
torch wrapper (guarded import).
"""

from __future__ import annotations

import numpy as np


def make_causal_windows(hand12: np.ndarray, body: np.ndarray, k_hand: int, k_body: int, p_predict: int):
    """Return arrays (cond_hand, cond_body, target) stacked over all valid t in one clip.

    Shapes: cond_hand (N, K, 12), cond_body (N, Kb, M), target (N, P, M).
    """
    hand12 = np.asarray(hand12, np.float32)
    body = np.asarray(body, np.float32)
    T = hand12.shape[0]
    assert body.shape[0] == T, "hand and body must share the time axis"
    k = max(k_hand, k_body)
    # inclusive range of valid current-frame t: need k-1 past frames AND p future frames,
    # so target body[t+1:t+1+p] stays in-bounds => t <= T-1-p.
    lo, hi = k - 1, T - p_predict - 1
    if hi < lo:
        return (np.empty((0, k_hand, hand12.shape[1]), np.float32),
                np.empty((0, k_body, body.shape[1]), np.float32),
                np.empty((0, p_predict, body.shape[1]), np.float32))
    ch, cb, tg = [], [], []
    for t in range(lo, hi + 1):
        ch.append(hand12[t - k_hand + 1: t + 1])
        cb.append(body[t - k_body + 1: t + 1])
        tg.append(body[t + 1: t + 1 + p_predict])
    return np.stack(ch), np.stack(cb), np.stack(tg)


def make_sequence_windows(hand12: np.ndarray, body: np.ndarray, length: int, stride: int = 1):
    """Same-frame windows for the lifting task: hand[t..t+L] -> body[t..t+L].

    Returns (hand_w (N, L, 12), body_w (N, L, M)). This is the M2 regressor's view:
    the body at frame t is produced from the hand up to t (causal), NOT forecast.
    """
    hand12 = np.asarray(hand12, np.float32)
    body = np.asarray(body, np.float32)
    T = hand12.shape[0]
    assert body.shape[0] == T
    if T < length:
        return (np.empty((0, length, hand12.shape[1]), np.float32),
                np.empty((0, length, body.shape[1]), np.float32))
    hw, bw = [], []
    for t in range(0, T - length + 1, stride):
        hw.append(hand12[t:t + length])
        bw.append(body[t:t + length])
    return np.stack(hw), np.stack(bw)


def canonicalize_window(hand_w: np.ndarray, body_w: np.ndarray):
    """Subtract the window-start HAND position from both hand position and body translation.

    The anchor is taken from the HAND (not the pelvis) so it is available at inference time
    too — at test we only have the hand signal, not the body root we are predicting. Shifting
    body translation by a constant shifts the FK wrist by the same constant, so hand/body stay
    consistent. Orientation channels are untouched (CONTRACT §5: position is canonicalized, the
    global wrist orientation is preserved). hand_w (..., L, 12), body_w (..., L, 135).
    Returns (hand_c, body_c, anchor (...,3))."""
    from ..representations import frames as F
    from ..representations import body as Bd
    hand_w = np.asarray(hand_w, np.float32).copy()
    body_w = np.asarray(body_w, np.float32).copy()
    anchor = hand_w[..., 0:1, F.HAND12_POS].copy()           # (..., 1, 3) hand pos at t0
    hand_w[..., F.HAND12_POS] -= anchor
    body_w[..., Bd.B_TRANS] -= anchor
    return hand_w, body_w, anchor[..., 0, :]


try:
    import torch
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        """Same-frame windows for the M2 regressor. clips = list of (hand12, body) arrays."""

        def __init__(self, clips, length: int, stride: int = 1, canonicalize: bool = True):
            hs, bs = [], []
            for hand12, body in clips:
                hw, bw = make_sequence_windows(hand12, body, length, stride)
                if len(hw):
                    if canonicalize:
                        hw, bw, _ = canonicalize_window(hw, bw)
                    hs.append(hw); bs.append(bw)
            self.hand = np.concatenate(hs) if hs else np.empty((0, length, 12), np.float32)
            self.body = np.concatenate(bs) if bs else np.empty((0, length, 135), np.float32)

        def __len__(self):
            return len(self.hand)

        def __getitem__(self, i):
            return torch.from_numpy(self.hand[i]), torch.from_numpy(self.body[i])

    class Hand2BodyDataset(Dataset):
        """Concatenate windows from many clips. Pass lists of (hand12, body) numpy arrays."""

        def __init__(self, clips, k_hand: int, k_body: int, p_predict: int):
            chs, cbs, tgs = [], [], []
            for hand12, body in clips:
                ch, cb, tg = make_causal_windows(hand12, body, k_hand, k_body, p_predict)
                if len(ch):
                    chs.append(ch); cbs.append(cb); tgs.append(tg)
            self.cond_hand = np.concatenate(chs) if chs else np.empty((0,))
            self.cond_body = np.concatenate(cbs) if cbs else np.empty((0,))
            self.target = np.concatenate(tgs) if tgs else np.empty((0,))

        def __len__(self):
            return len(self.cond_hand)

        def __getitem__(self, i):
            return (torch.from_numpy(self.cond_hand[i]),
                    torch.from_numpy(self.cond_body[i]),
                    torch.from_numpy(self.target[i]))

except Exception:  # pragma: no cover
    Hand2BodyDataset = None  # type: ignore
