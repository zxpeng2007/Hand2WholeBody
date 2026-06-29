"""Torch-dependent smoke tests. Skipped automatically where torch isn't installed.

Validates that the DiffMLP denoiser constructs and produces correctly-shaped output,
and that the torch Dataset wrapper yields the right tensor shapes. These are shape/plumbing
checks, not training correctness.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from h2wb.models.diffmlp import DiffMLP
from h2wb.data.dataset import Hand2BodyDataset


def test_diffmlp_forward_shapes():
    B, K, Kb, P, M = 4, 20, 5, 8, 135
    model = DiffMLP(motion_dim=M, k_hand=K, k_body=Kb, p_predict=P,
                    hidden=128, n_blocks=4, latent=128)
    x_t = torch.randn(B, P, M)
    t = torch.randint(0, 1000, (B,))
    hand = torch.randn(B, K, 12)
    body = torch.randn(B, Kb, M)
    out = model(x_t, t, hand, body)
    assert out.shape == (B, P, M)
    assert torch.isfinite(out).all()


def test_diffmlp_backward_runs():
    B, K, Kb, P, M = 2, 10, 3, 4, 135
    model = DiffMLP(motion_dim=M, k_hand=K, k_body=Kb, p_predict=P,
                    hidden=64, n_blocks=2, latent=64)
    out = model(torch.randn(B, P, M), torch.randint(0, 1000, (B,)),
                torch.randn(B, K, 12), torch.randn(B, Kb, M))
    loss = out.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_torch_dataset_window_shapes():
    T, M = 60, 135
    hand = np.random.randn(T, 12).astype(np.float32)
    body = np.random.randn(T, M).astype(np.float32)
    ds = Hand2BodyDataset([(hand, body)], k_hand=20, k_body=5, p_predict=8)
    assert len(ds) > 0
    ch, cb, tg = ds[0]
    assert ch.shape == (20, 12) and cb.shape == (5, M) and tg.shape == (8, M)
