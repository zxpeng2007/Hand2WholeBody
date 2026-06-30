"""Held-out evaluation for Hand2Body — interpretable metrics on UNSEEN sequences.

Splits clips by sequence (whole sequences held out, so no window leakage), then reports:
  * mpjpe_mm     : mean per-joint position error after FK (mm) — overall body accuracy.
  * wrist_pos_mm : generated left-wrist position vs the INPUT hand (mm) — held-out
                   hand-consistency (does the body follow the conditioning hand on unseen data?).
  * wrist_deg    : generated wrist orientation vs the input hand 6D (geodesic degrees).
  * jitter       : mean joint acceleration magnitude (smoothness; lower = more trackable).

Works for the regressor (forward) and the diffusion model (DDIM sample). torch-gated.
"""

from __future__ import annotations

import numpy as np

from .representations import body as B


def split_clips(clips, val_frac: float = 0.1, seed: int = 0):
    """Split a clip list into (train, val) by whole sequences. Deterministic."""
    n = len(clips)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_i = set(idx[:n_val].tolist())
    train = [c for k, c in enumerate(clips) if k not in val_i]
    val = [c for k, c in enumerate(clips) if k in val_i]
    return train, val


def _geodesic_deg(R1, R2):
    import torch
    Rrel = R1.transpose(-1, -2) @ R2
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def evaluate(model, val_clips, length=40, device="cpu", arch="regressor", diffusion=None,
             rest_joints=None, sample_steps=8, max_windows=256, fps=30):
    """Return a dict of held-out metrics (averaged over up to max_windows val windows)."""
    import torch
    from torch.utils.data import DataLoader
    from .data.dataset import SequenceDataset
    from .models import fk_torch as FK
    from .representations import rotations_torch as RT
    from .representations import frames as F

    feet = list(F.FOOT_JOINTS)
    ds = SequenceDataset(val_clips, length=length, stride=length, canonicalize=True)
    if len(ds) == 0:
        return {}
    rest = None if rest_joints is None else torch.as_tensor(rest_joints, dtype=torch.float32, device=device)
    dl = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False)

    model.eval()
    mpjpe, wpos, wdeg, jit, skate, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    seen = 0
    with torch.no_grad():
        for hand, body in dl:
            if seen >= max_windows:
                break
            hand, body = hand.to(device), body.to(device)
            if arch == "diffusion":
                pred = diffusion.ddim_sample(model, body.shape, hand, steps=sample_steps, device=device)
            else:
                pred = model(hand)
            _, pred_pos = FK.motion_to_joints(pred, rest)
            _, gt_pos = FK.motion_to_joints(body, rest)
            mpjpe += (pred_pos - gt_pos).norm(dim=-1).mean().item() * hand.shape[0]
            wp, wr = FK.left_wrist_pose(pred, rest)
            wpos += (wp - hand[..., 0:3]).norm(dim=-1).mean().item() * hand.shape[0]
            wdeg += _geodesic_deg(RT.rotation_6d_to_matrix(wr),
                                  RT.rotation_6d_to_matrix(hand[..., 6:12])).mean().item() * hand.shape[0]
            acc = pred_pos[:, 2:] - 2 * pred_pos[:, 1:-1] + pred_pos[:, :-2]
            jit += acc.norm(dim=-1).mean().item() * hand.shape[0] * (fps ** 2)
            # foot-skate: horizontal foot speed (m/s) over near-ground frames
            fpos = pred_pos[..., feet, :]
            fheight = fpos[..., 2]
            near = (fheight < fheight.min(dim=1, keepdim=True).values + 0.05).float()
            hspeed = torch.zeros_like(fheight)
            hspeed[:, 1:] = ((fpos[:, 1:, :, :2] - fpos[:, :-1, :, :2]) * fps).norm(dim=-1)
            skate += ((hspeed * near).sum() / (near.sum() + 1e-6)).item() * hand.shape[0]
            n += hand.shape[0]
            seen += hand.shape[0]
    return {
        "val_mpjpe_mm": 1000.0 * mpjpe / n,
        "val_wrist_pos_mm": 1000.0 * wpos / n,
        "val_wrist_deg": wdeg / n,
        "val_jitter": jit / n,
        "val_footskate_mm_s": 1000.0 * skate / n,
        "val_windows": n,
    }
