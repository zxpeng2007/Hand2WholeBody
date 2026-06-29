"""M4 torch tests: diffusion schedule, denoiser causality, DDIM sampling, overfit, streaming.

Skipped where torch isn't installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from h2b.representations import body as B
from h2b.representations import rotations as Rnp
from h2b.data.dataset import SequenceDataset
from h2b.models import fk_torch as FKt
from h2b.models.diffusion import GaussianDiffusion, DiTDenoiser, cosine_beta_schedule
from h2b.models.streaming import DiffusionStreamer
from h2b import training as TR


def test_cosine_schedule_sane():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas > 0).all() and (betas < 1).all()
    acp = torch.cumprod(1 - betas, 0)
    assert acp[0] > acp[-1] and acp[-1] >= 0           # cumprod decreasing toward 0


def test_q_sample_endpoints():
    diff = GaussianDiffusion(num_steps=100)
    x0 = torch.randn(2, 8, B.MOTION_DIM)
    noise = torch.randn_like(x0)
    near0 = diff.q_sample(x0, torch.zeros(2, dtype=torch.long), noise)
    assert torch.allclose(near0, x0, atol=0.2)         # t=0 ~ clean
    late = diff.q_sample(x0, torch.full((2,), 99, dtype=torch.long), noise)
    assert (late - x0).abs().mean() > (near0 - x0).abs().mean()   # late ~ noisier


def test_denoiser_forward_backward_and_causal():
    m = DiTDenoiser(hidden=64, n_layers=2)
    x = torch.randn(2, 12, B.MOTION_DIM)
    t = torch.randint(0, 1000, (2,))
    hand = torch.randn(2, 12, 12)
    out = m(x, t, hand)
    assert out.shape == (2, 12, B.MOTION_DIM)
    out.pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    # causal: perturbing the last frame of x AND hand must not change earlier outputs
    m.eval()
    with torch.no_grad():
        x2, h2 = x.clone(), hand.clone()
        x2[:, -1] += 3.0; h2[:, -1] += 3.0
        out2 = m(x2, t, h2)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5)


def test_ddim_sample_shape_finite():
    diff = GaussianDiffusion(num_steps=100)
    m = DiTDenoiser(hidden=64, n_layers=2).eval()
    hand = torch.randn(2, 16, 12)
    out = diff.ddim_sample(m, (2, 16, B.MOTION_DIM), hand, steps=5)
    assert out.shape == (2, 16, B.MOTION_DIM) and torch.isfinite(out).all()


def test_diffusion_overfit_reduces_loss():
    clips = TR.synthetic_clips(n_clips=4, T=40, seed=5)
    _, _, history = TR.train_diffusion(clips, length=20, steps=250, batch_size=16, lr=3e-4,
                                       device="cpu", hidden=64, n_layers=2, num_steps=100,
                                       log_every=25, seed=0)
    assert history[-1]["total"] < 0.7 * history[0]["total"]


def test_trained_sample_honors_hand_better_than_tpose():
    clips = TR.synthetic_clips(n_clips=4, T=40, seed=6)
    model, diff, _ = TR.train_diffusion(clips, length=20, steps=300, batch_size=16, lr=3e-4,
                                        device="cpu", hidden=64, n_layers=2, num_steps=100, seed=0)
    model.eval()
    ds = SequenceDataset(clips, length=20, stride=10, canonicalize=True)
    hand, _body = ds[0]
    hand = hand[None]                                   # (1, L, 12)
    sample = TR.sample_diffusion(model, diff, hand, steps=10)
    pos_m, rot_m = FKt.left_wrist_pose(sample)
    err_model = (pos_m - hand[..., 0:3]).pow(2).mean() + (rot_m - hand[..., 6:12]).pow(2).mean()
    # T-pose baseline: zero trans, identity rotations (column 6D of I = [1,0,0,0,1,0]).
    ident = torch.tensor(Rnp.matrix_to_rotation_6d(np.eye(3), Rnp.R6D_COLUMN), dtype=torch.float32)
    base = torch.zeros_like(sample)
    base[..., B.B_ROT6D] = ident.repeat(B.NUM_BODY_JOINTS)
    pos_b, rot_b = FKt.left_wrist_pose(base)
    err_base = (pos_b - hand[..., 0:3]).pow(2).mean() + (rot_b - hand[..., 6:12]).pow(2).mean()
    assert err_model < err_base                         # the sample honors the hand


def test_streamer_smoke():
    diff = GaussianDiffusion(num_steps=50)
    m = DiTDenoiser(hidden=64, n_layers=2).eval()
    s = DiffusionStreamer(m, diff, window=12, sample_steps=4)
    assert s.push(np.zeros(12, np.float32)) is None      # first frame: no output yet
    out = s.push(np.ones(12, np.float32))
    assert out.shape == (B.MOTION_DIM,) and np.isfinite(out).all()
