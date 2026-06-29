"""Tests for the causal windowing logic."""

import numpy as np

from h2b.data.dataset import make_causal_windows


def test_window_shapes_and_count():
    T, M = 50, 7
    hand = np.arange(T * 12).reshape(T, 12).astype(np.float32)
    body = np.arange(T * M).reshape(T, M).astype(np.float32)
    k_hand, k_body, p = 20, 5, 8
    ch, cb, tg = make_causal_windows(hand, body, k_hand, k_body, p)
    # valid current-frame t in [max(k_hand,k_body)-1, T-1-p]  =>  count = (T-p) - max(k) + 1
    n_expected = (T - p) - max(k_hand, k_body) + 1
    assert ch.shape == (n_expected, k_hand, 12)
    assert cb.shape == (n_expected, k_body, M)
    assert tg.shape == (n_expected, p, M)


def test_window_is_causal_and_aligned():
    T, M = 30, 3
    hand = np.zeros((T, 12), np.float32)
    body = (np.arange(T)[:, None] * np.ones((1, M))).astype(np.float32)  # body[t]=t
    k_hand, k_body, p = 4, 2, 3
    ch, cb, tg = make_causal_windows(hand, body, k_hand, k_body, p)
    # first valid current-frame t = max(k_hand,k_body)-1 = 3
    # cond_body last row is body[t]; target first row is body[t+1]
    assert cb[0, -1, 0] == 3.0
    assert tg[0, 0, 0] == 4.0
    # no target frame ever references the past (strictly future of current t)
    assert tg[0, 0, 0] > cb[0, -1, 0]


def test_too_short_clip_returns_empty():
    ch, cb, tg = make_causal_windows(np.zeros((5, 12)), np.zeros((5, 4)), 20, 5, 8)
    assert len(ch) == 0 and len(cb) == 0 and len(tg) == 0
