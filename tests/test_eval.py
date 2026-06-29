"""Tests for the held-out split + evaluation scaffolding."""

import numpy as np
import pytest

from h2wb.eval import split_clips

torch = pytest.importorskip("torch")

from h2wb import training as TR
from h2wb import eval as EV
from h2wb.models.regressor import RegressorHand2Body


def test_split_is_disjoint_and_deterministic():
    clips = [(np.zeros((10, 12)), np.zeros((10, 135))) for _ in range(10)]
    tr1, va1 = split_clips(clips, val_frac=0.2, seed=0)
    tr2, va2 = split_clips(clips, val_frac=0.2, seed=0)
    assert len(tr1) == 8 and len(va1) == 2
    assert len(tr1) + len(va1) == len(clips)
    assert [id(c) for c in tr1] == [id(c) for c in tr2]   # deterministic


def test_evaluate_returns_finite_metrics():
    clips = TR.synthetic_clips(n_clips=4, T=48, seed=1)
    model = RegressorHand2Body(hidden=32, n_layers=1)
    m = EV.evaluate(model, clips, length=24, device="cpu", arch="regressor")
    for k in ("val_mpjpe_mm", "val_wrist_pos_mm", "val_wrist_deg", "val_jitter", "val_windows"):
        assert k in m
    assert np.isfinite(m["val_mpjpe_mm"]) and m["val_windows"] > 0
    assert 0 <= m["val_wrist_deg"] <= 180


def test_training_logs_val_metrics():
    clips = TR.synthetic_clips(n_clips=6, T=48, seed=2)
    train_clips, val_clips = split_clips(clips, val_frac=0.34)
    model, history = TR.train(train_clips, length=24, steps=60, batch_size=16, device="cpu",
                              hidden=32, n_layers=1, log_every=30, val_clips=val_clips,
                              eval_every=30, seed=0)
    assert any("val_mpjpe_mm" in h for h in history)       # val metrics appear in history
