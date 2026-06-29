"""M2 torch tests: rotation/FK parity, regressor causality, canonicalization, overfit.

Skipped where torch isn't installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from h2b.representations import body as B
from h2b.representations import frames as F
from h2b.representations import rotations as Rnp
from h2b.representations import rotations_torch as Rt
from h2b.data import smpl_fk as FK
from h2b.data.dataset import make_sequence_windows, canonicalize_window
from h2b.models import fk_torch as FKt
from h2b.models.regressor import RegressorHand2Body
from h2b import training as TR


def test_torch_rotation_parity_with_numpy():
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((50, 3)) * 0.8
    Rn = Rnp.axis_angle_to_matrix(aa)
    Rtt = Rt.axis_angle_to_matrix(torch.tensor(aa)).numpy()
    assert np.allclose(Rn, Rtt, atol=1e-9)
    d6 = Rnp.matrix_to_rotation_6d(Rn, convention=Rnp.R6D_COLUMN)
    d6t = Rt.matrix_to_rotation_6d(torch.tensor(Rn)).numpy()
    assert np.allclose(d6, d6t, atol=1e-9)
    Rback = Rt.rotation_6d_to_matrix(torch.tensor(d6)).numpy()
    assert np.allclose(Rback, Rn, atol=1e-9)


def test_fk_torch_matches_numpy_extractor_for_left_wrist():
    # Build a synthetic clip; FK(body) left-wrist must equal the numpy-extracted 12D.
    rng = np.random.default_rng(1)
    T = 20
    poses = np.cumsum(rng.standard_normal((T, 72)) * 0.05, axis=0)
    trans = np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + [0, 0, 1.0]
    body = B.smpl72_to_motion(poses, trans)
    hand = FK.extract_hand12(poses, trans, np.zeros(10), FK.synthetic_joints_fn, fps=30.0)
    pos, rot6d = FKt.left_wrist_pose(torch.tensor(body)[None])      # (1,T,3),(1,T,6)
    assert np.allclose(pos[0].numpy(), hand[:, 0:3], atol=1e-4)      # position parity
    assert np.allclose(rot6d[0].numpy(), hand[:, 6:12], atol=1e-4)   # orientation parity


def test_regressor_forward_backward_shapes():
    m = RegressorHand2Body(hidden=64, n_layers=2)
    hand = torch.randn(3, 16, 12)
    out = m(hand)
    assert out.shape == (3, 16, B.MOTION_DIM)
    out.pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_regressor_is_causal():
    # Changing a FUTURE input frame must not change earlier outputs (streaming-safe).
    torch.manual_seed(0)
    m = RegressorHand2Body(hidden=64, n_layers=2).eval()
    hand = torch.randn(1, 12, 12)
    with torch.no_grad():
        out_a = m(hand)
        hand2 = hand.clone(); hand2[0, -1] += 5.0          # perturb the last frame
        out_b = m(hand2)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)   # earlier frames unchanged
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)  # last frame did change


def test_canonicalization_keeps_hand_body_consistent():
    rng = np.random.default_rng(2)
    T = 30
    poses = np.cumsum(rng.standard_normal((T, 72)) * 0.05, axis=0)
    trans = np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + [0, 0, 1.0]
    body = B.smpl72_to_motion(poses, trans).astype(np.float32)
    hand = FK.extract_hand12(poses, trans, np.zeros(10), FK.synthetic_joints_fn).astype(np.float32)
    hw, bw = make_sequence_windows(hand, body, length=16, stride=8)
    hc, bc, anchor = canonicalize_window(hw, bw)
    # orientation untouched; positions shifted by the same per-window anchor
    assert np.allclose(hc[..., F.HAND12_ROT6D], hw[..., F.HAND12_ROT6D])
    # FK(canon body) left wrist position still equals canon hand position
    pos, _ = FKt.left_wrist_pose(torch.tensor(bc))
    assert np.allclose(pos.numpy(), hc[..., 0:3], atol=1e-4)


def test_overfit_loop_reduces_loss():
    clips = TR.synthetic_clips(n_clips=4, T=48, seed=3)
    model, history = TR.train(clips, length=24, steps=250, batch_size=16, lr=3e-4,
                              device="cpu", hidden=64, n_layers=2, log_every=25, seed=0)
    assert len(history) >= 5
    first, last = history[0], history[-1]
    assert last["total"] < 0.5 * first["total"]            # loop clearly learns
    assert last["hand_consistency"] < first["hand_consistency"]  # conditioning is honored
