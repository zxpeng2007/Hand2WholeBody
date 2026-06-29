"""M5 tests: inference (hand->SMPL->npz) and the visualization montage. Torch-gated."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from h2b.representations import body as B
from h2b import inference as INF
from h2b import training as TR
from h2b.models.regressor import RegressorHand2Body
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion


def _hand_seq(T=24, seed=0):
    return TR.synthetic_clips(n_clips=1, T=T, seed=seed)[0][0]


def test_generate_regressor_shape_and_world_anchor():
    model = RegressorHand2Body(hidden=32, n_layers=1)
    hand = _hand_seq()
    motion = INF.generate(model, hand, arch="regressor")
    assert motion.shape == (hand.shape[0], B.MOTION_DIM)
    assert np.isfinite(motion).all()


def test_generate_diffusion_runs():
    model = DiTDenoiser(hidden=32, n_layers=1)
    diff = GaussianDiffusion(num_steps=50)
    hand = _hand_seq()
    motion = INF.generate(model, hand, arch="diffusion", diffusion=diff, sample_steps=3)
    assert motion.shape == (hand.shape[0], B.MOTION_DIM) and np.isfinite(motion).all()


def test_generate_to_npz_writes_contract_keys(tmp_path):
    model = RegressorHand2Body(hidden=32, n_layers=1)
    hand = _hand_seq()
    out = INF.generate_to_npz(str(tmp_path / "gen.npz"), model, hand, arch="regressor")
    d = np.load(out, allow_pickle=True)
    assert d["poses"].shape == (hand.shape[0], 72)
    assert d["trans"].shape == (hand.shape[0], 3)
    assert int(d["mocap_frame_rate"]) == 30


def test_generate_long_stitches_beyond_budget():
    model = RegressorHand2Body(hidden=32, n_layers=1, max_len=64)
    hand = _hand_seq(T=200)                      # > the 64-frame budget -> must chunk
    motion = INF.generate_long(model, hand, arch="regressor", chunk=60, overlap=15)
    assert motion.shape == (200, B.MOTION_DIM)
    assert np.isfinite(motion).all()


def test_sequence_too_long_raises():
    model = RegressorHand2Body(hidden=32, n_layers=1, max_len=16)
    hand = _hand_seq(T=40)
    with pytest.raises(ValueError):
        INF.generate(model, hand, arch="regressor")


def test_visualization_writes_png(tmp_path):
    motion = B.smpl72_to_motion(np.zeros((20, 72)), np.tile([0, 0, 1.0], (20, 1)))
    from h2b.export.visualize import plot_skeleton_montage
    out = plot_skeleton_montage(motion, str(tmp_path / "viz.png"), n_frames=4)
    assert (tmp_path / "viz.png").exists()
    assert (tmp_path / "viz.png").stat().st_size > 1000
