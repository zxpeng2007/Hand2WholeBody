"""Tests for foot-sliding fixes: A (post-process foot-lock) and B (foot-contact loss)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from h2b.representations import body as B
from h2b.export import footlock as FL
from h2b import training as TR
from h2b import losses as L


def test_detect_contacts():
    foot = np.zeros((10, 2, 3))
    foot[:, 1, 2] = 0.5                       # foot 1 high
    foot[:, 1, 0] = np.arange(10) * 0.1       # foot 1 moving (3 m/s)
    contact = FL.detect_contacts(foot, fps=30.0)
    assert contact[:, 0].all()                # foot 0 low + still -> contact
    assert not contact[1:, 1].any()           # foot 1 moving -> no contact


def test_footlock_runs_preserves_hand_and_shape():
    hand, body = TR.synthetic_clips(n_clips=1, T=40, seed=0)[0]   # hand == FK(body)
    locked, contact = FL.footlock(body, hand=hand, iters=60, device="cpu")
    assert locked.shape == body.shape and np.isfinite(locked).all()
    assert contact.shape[0] == body.shape[0]
    # the left wrist must still sit on the input hand (conditioning preserved)
    from h2b.models import fk_torch as FKt
    pos, _ = FKt.left_wrist_pose(torch.tensor(locked)[None])
    assert np.allclose(pos[0].numpy(), hand[:, 0:3], atol=0.03)


def test_foot_skate_metric_finite():
    _, body = TR.synthetic_clips(n_clips=1, T=30, seed=1)[0]
    s = FL.foot_skate(body)
    assert np.isfinite(s) and s >= 0.0


def test_foot_contact_loss_in_compute_losses():
    rng = torch.manual_seed(0)
    pred = torch.randn(2, 16, B.MOTION_DIM)
    gt = torch.randn(2, 16, B.MOTION_DIM)
    hand = torch.randn(2, 16, 12)
    w = {"trans": 1.0, "foot_contact": 0.5}
    total, parts = L.compute_losses(pred, gt, hand, w)
    assert "foot_contact" in parts and torch.isfinite(parts["foot_contact"])
    assert torch.isfinite(total)
