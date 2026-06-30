"""Post-process foot-locking (the AI4Animation 'contact detection + LegIK' analog).

Generated motion slides its feet because nothing enforces ground contact. This module:
  1. detects per-foot ground contact from the generated motion (foot near the ground AND slow),
  2. pins each contact segment to a fixed world point (the segment median), and
  3. solves a full-body IK (differentiable FK + Adam) that drives contacting feet to their pins
     WHILE keeping the left wrist on the input 12D hand (so the conditioning is preserved) and
     regularizing toward the original pose + smoothness.

Operates on our 135-D motion (h2b.representations.body) -> a foot-locked 135-D motion.
"""

from __future__ import annotations

import numpy as np

from ..representations import body as B
from ..representations import frames as F


def detect_contacts(foot_pos, fps=30.0, h_offset=0.05, v_thresh=0.30):
    """foot_pos (T, nf, 3) -> contact (T, nf) bool. Foot is in contact if it is within
    h_offset of its own clip-minimum height AND moving slower than v_thresh (m/s)."""
    fp = np.asarray(foot_pos, np.float64)
    T = fp.shape[0]
    height = fp[..., 2]                                   # (T, nf)  world z, ground = low z
    vel = np.zeros_like(fp)
    if T > 1:
        vel[1:] = (fp[1:] - fp[:-1]) * fps
    speed = np.linalg.norm(vel, axis=-1)                  # (T, nf) m/s
    ground = height.min(axis=0, keepdims=True)            # (1, nf) per-foot ground level
    return (height < ground + h_offset) & (speed < v_thresh)


def _lock_targets(foot_pos, contact):
    """Per foot, replace each contact run with its median position (the fixed pin)."""
    fp = np.asarray(foot_pos, np.float64).copy()
    T, nf = contact.shape
    for f in range(nf):
        t = 0
        while t < T:
            if contact[t, f]:
                s = t
                while t < T and contact[t, f]:
                    t += 1
                fp[s:t, f] = np.median(fp[s:t, f], axis=0)
            else:
                t += 1
    return fp


def footlock(motion, hand=None, rest_joints=None, fps=30.0, iters=300, lr=0.05,
             w_foot=2.5, w_hand=1.0, w_reg=0.02, w_root=0.02, w_smooth=0.03,
             h_offset=0.06, v_thresh=0.45, device="cpu"):
    """motion (T,135) -> (locked_motion (T,135), contact (T,nf)). hand (T,12) optional keeps
    the left wrist on the conditioning signal."""
    import torch
    from ..models import fk_torch as FKt

    feet = list(F.FOOT_JOINTS)                            # (7,8,10,11) ankles + toes, all < 22
    motion = np.asarray(motion, np.float32)
    rest = (FKt.default_rest_joints() if rest_joints is None
            else torch.as_tensor(rest_joints, dtype=torch.float32)).to(device)

    with torch.no_grad():
        _, pos0 = FKt.motion_to_joints(torch.tensor(motion, device=device)[None], rest)
    foot_pos0 = pos0[0, :, feet].cpu().numpy()            # (T, nf, 3)
    contact = detect_contacts(foot_pos0, fps, h_offset, v_thresh)
    targets = _lock_targets(foot_pos0, contact)

    target_t = torch.tensor(targets, dtype=torch.float32, device=device)        # (T,nf,3)
    cmask = torch.tensor(contact[..., None], dtype=torch.float32, device=device)  # (T,nf,1)
    n_contact = float(contact.sum()) or 1.0
    feet_idx = torch.tensor(feet, device=device)
    hand_t = None if hand is None else torch.tensor(np.asarray(hand, np.float32)[:, 0:3], device=device)

    rot = torch.tensor(motion[:, B.B_ROT6D], device=device).clone().requires_grad_(True)
    trans = torch.tensor(motion[:, B.B_TRANS], device=device).clone().requires_grad_(True)
    rot0, trans0 = rot.detach().clone(), trans.detach().clone()
    opt = torch.optim.Adam([rot, trans], lr=lr)

    for _ in range(iters):
        body = torch.cat([trans, rot], dim=-1)
        _, pos = FKt.motion_to_joints(body[None], rest)
        pos = pos[0]
        foot = pos[:, feet_idx]                                                  # (T,nf,3)
        loss = w_foot * (cmask * (foot - target_t) ** 2).sum() / n_contact
        loss = loss + w_reg * ((rot - rot0) ** 2).mean()
        loss = loss + w_root * ((trans - trans0) ** 2).mean()
        acc = body[2:] - 2 * body[1:-1] + body[:-2]
        loss = loss + w_smooth * (acc ** 2).mean()
        if hand_t is not None:
            loss = loss + w_hand * ((pos[:, F.LEFT_WRIST] - hand_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    locked = torch.cat([trans.detach(), rot.detach()], dim=-1).cpu().numpy().astype(np.float32)
    return locked, contact


def foot_skate(motion, rest_joints=None, fps=30.0, h_offset=0.05):
    """Foot-skate metric: mean horizontal foot speed (m/s) over ground-contact frames.
    Lower = less sliding. Contact here is height-only (so it doesn't pre-assume low speed)."""
    import torch
    from ..models import fk_torch as FKt
    rest = (FKt.default_rest_joints() if rest_joints is None
            else torch.as_tensor(rest_joints, dtype=torch.float32))
    with torch.no_grad():
        _, pos = FKt.motion_to_joints(torch.as_tensor(np.asarray(motion, np.float32))[None], rest)
    fp = pos[0, :, list(F.FOOT_JOINTS)].cpu().numpy()          # (T, nf, 3)
    T = fp.shape[0]
    if T < 2:
        return 0.0
    horiz = np.zeros((T, fp.shape[1]))
    horiz[1:] = np.linalg.norm((fp[1:, :, :2] - fp[:-1, :, :2]) * fps, axis=-1)   # m/s in xy
    height = fp[..., 2]
    near_ground = height < height.min(axis=0, keepdims=True) + h_offset
    m = near_ground.sum()
    return float((horiz * near_ground).sum() / m) if m else 0.0
