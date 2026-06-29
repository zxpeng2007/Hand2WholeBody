"""M2 training loop for the deterministic regressor — importable and testable.

`synthetic_clips` builds self-consistent (hand12, body) pairs: a random smooth SMPL motion
is turned into the 135-D body vector AND its left-wrist 12D via the same FK, so the hand
signal is exactly FK(body). That makes the loop verifiable today (the model can drive the
loss down, and the hand-consistency term is genuinely satisfiable) before real data lands.

`train` runs the regressor with the combined loss (h2wb.losses). Swap `synthetic_clips`
for real extracted pairs (scripts/extract_amass.py) with zero changes to `train`.
"""

from __future__ import annotations

import numpy as np

from .data import smpl_fk as FK
from .representations import body as B


def synthetic_clips(n_clips: int = 8, T: int = 64, seed: int = 0):
    """List of (hand12 (T,12), body (T,135)) self-consistent pairs (hand == FK(body))."""
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(n_clips):
        # smooth low-frequency pose: cumulative small steps then mild smoothing.
        steps = rng.standard_normal((T, 72)) * 0.03
        poses = np.cumsum(steps, axis=0)
        poses -= poses.mean(0, keepdims=True)
        trans = np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + np.array([0.0, 0.0, 1.0])
        body = B.smpl72_to_motion(poses, trans)
        hand = FK.extract_hand12(poses, trans, np.zeros(10), FK.synthetic_joints_fn, fps=30.0)
        clips.append((hand.astype(np.float32), body.astype(np.float32)))
    return clips


def default_loss_weights():
    return dict(trans=1.0, rot6d=1.0, velocity=0.5, fk_joint=1.0, hand_consistency=2.0)


def build_model(hidden=256, n_layers=4, n_heads=8):
    from .models.regressor import RegressorHand2Body
    return RegressorHand2Body(hidden=hidden, n_layers=n_layers, n_heads=n_heads)


def train(clips, length=40, steps=300, batch_size=64, lr=2e-4, device="cpu",
          weights=None, log_every=50, hidden=256, n_layers=4, seed=0):
    """Train the regressor on (hand, body) clips. Returns (model, history list of dicts)."""
    import torch
    from torch.utils.data import DataLoader
    from .data.dataset import SequenceDataset
    from . import losses as L

    torch.manual_seed(seed)
    weights = weights or default_loss_weights()
    ds = SequenceDataset(clips, length=length, stride=max(1, length // 2), canonicalize=True)
    if len(ds) == 0:
        raise ValueError("no training windows — clips shorter than `length`?")
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    model = build_model(hidden=hidden, n_layers=n_layers).to(device)
    rest = None  # default approx rest joints inside fk_torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history, step = [], 0
    model.train()
    while step < steps:
        for hand, body in dl:
            hand, body = hand.to(device), body.to(device)
            pred = model(hand)
            total, parts = L.compute_losses(pred, body, hand, weights, rest_joints=rest)
            opt.zero_grad(); total.backward(); opt.step()
            if step % log_every == 0:
                history.append({"step": step, "total": float(total.detach()),
                                **{k: float(v.detach()) for k, v in parts.items()}})
            step += 1
            if step >= steps:
                break
    return model, history


def train_diffusion(clips, length=40, steps=2000, batch_size=64, lr=2e-4, device="cpu",
                    weights=None, log_every=50, hidden=256, n_layers=4, num_steps=1000, seed=0):
    """Train the M4 conditional diffusion model (DiTDenoiser). Returns (model, diffusion, history).

    Per step: noise the GT body to a random diffusion time, denoise it conditioned on the
    hand, and supervise the predicted clean motion x0 with the SAME combined loss as M2
    (so FK + hand-consistency apply directly to x0).
    """
    import torch
    from torch.utils.data import DataLoader
    from .data.dataset import SequenceDataset
    from .models.diffusion import GaussianDiffusion, DiTDenoiser
    from . import losses as L

    torch.manual_seed(seed)
    weights = weights or default_loss_weights()
    ds = SequenceDataset(clips, length=length, stride=max(1, length // 2), canonicalize=True)
    if len(ds) == 0:
        raise ValueError("no training windows — clips shorter than `length`?")
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    model = DiTDenoiser(hidden=hidden, n_layers=n_layers).to(device)
    diff = GaussianDiffusion(num_steps=num_steps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history, step = [], 0
    model.train()
    while step < steps:
        for hand, body in dl:
            hand, body = hand.to(device), body.to(device)
            t = diff.sample_t(body.shape[0], device)
            x_t = diff.q_sample(body, t, torch.randn_like(body))
            x0_hat = model(x_t, t, hand)
            total, parts = L.compute_losses(x0_hat, body, hand, weights)
            opt.zero_grad(); total.backward(); opt.step()
            if step % log_every == 0:
                history.append({"step": step, "total": float(total.detach()),
                                **{k: float(v.detach()) for k, v in parts.items()}})
            step += 1
            if step >= steps:
                break
    return model, diff, history


def sample_diffusion(model, diff, hand, steps=8):
    """Generate body[1..L] from hand[1..L]. hand: (B, L, 12) tensor -> (B, L, 135)."""
    shape = (hand.shape[0], hand.shape[1], B.MOTION_DIM)
    return diff.ddim_sample(model, shape, hand, steps=steps, device=hand.device)
