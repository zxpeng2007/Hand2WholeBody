"""Training objective for Hand2Body (weights in configs/default.yaml `loss`).

Operates on the model's 135-D motion output (h2b.representations.body) and the input 12D
hand signal. Terms:
  * trans            — root translation L2.
  * rot6d            — 6D joint-rotation L2 (all 22 joints incl. global_orient).
  * velocity         — first/second-difference penalty (smoothness -> trackable by HoloMotion).
  * fk_joint         — 3D joint-position L2 after differentiable FK (vs GT pose).
  * hand_consistency — THE key term: FK the predicted pose, recover the GLOBAL left-wrist
                       position + 6D orientation, and match them to the input 12D. Forces
                       the body to actually honor the conditioning hand.
  * foot_contact     — anti foot-skate: where the GT foot is in ground contact (near ground
                       and slow), penalize the PREDICTED foot's velocity so planted feet stay put.

torch-guarded so importing the package never requires torch.
"""

from __future__ import annotations

from .representations import body as B
from .representations import frames as F

try:
    import torch
    import torch.nn.functional as Fn
    from .models import fk_torch as FK
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def velocity_loss(pred_motion):
    """Penalize frame-to-frame velocity + acceleration. pred_motion: (B, L, 135)."""
    v = pred_motion[:, 1:] - pred_motion[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    return v.pow(2).mean() + a.pow(2).mean()


def foot_contact_loss(pred_pos, gt_pos, fps=30.0, h_offset=0.05, v_thresh=0.30):
    """Penalize predicted foot velocity where the GT foot is in ground contact.
    pred_pos/gt_pos: (B, L, 22, 3) world joint positions."""
    feet = list(F.FOOT_JOINTS)
    gtf = gt_pos[..., feet, :]                                  # (B,L,nf,3)
    height = gtf[..., 2]                                        # (B,L,nf)
    gv = (gtf[:, 1:] - gtf[:, :-1]) * fps
    speed = torch.zeros_like(height)
    speed[:, 1:] = gv.norm(dim=-1)
    ground = height.min(dim=1, keepdim=True).values            # (B,1,nf) per-clip ground level
    contact = ((height < ground + h_offset) & (speed < v_thresh)).float()[..., None]  # (B,L,nf,1)
    pv = (pred_pos[..., feet, :][:, 1:] - pred_pos[..., feet, :][:, :-1]) * fps        # (B,L-1,nf,3)
    return (contact[:, 1:] * pv.pow(2)).sum() / (contact[:, 1:].sum() * 3 + 1e-6)


def compute_losses(pred_motion, gt_motion, hand, weights, rest_joints=None, fps=30.0):
    """Return (total, parts dict). Shapes: pred/gt (B,L,135), hand (B,L,12).

    `weights` keys: trans, rot6d, velocity, fk_joint, hand_consistency, foot_contact. Missing -> 0.
    """
    parts = {}
    parts["trans"] = Fn.mse_loss(pred_motion[..., B.B_TRANS], gt_motion[..., B.B_TRANS])
    parts["rot6d"] = Fn.mse_loss(pred_motion[..., B.B_ROT6D], gt_motion[..., B.B_ROT6D])
    parts["velocity"] = velocity_loss(pred_motion)

    need_fk = weights.get("fk_joint", 0.0) > 0 or weights.get("foot_contact", 0.0) > 0
    if need_fk:
        _, pred_pos = FK.motion_to_joints(pred_motion, rest_joints)
        with torch.no_grad():
            _, gt_pos = FK.motion_to_joints(gt_motion, rest_joints)
        if weights.get("fk_joint", 0.0) > 0:
            parts["fk_joint"] = Fn.mse_loss(pred_pos, gt_pos)
        if weights.get("foot_contact", 0.0) > 0:
            parts["foot_contact"] = foot_contact_loss(pred_pos, gt_pos, fps=fps)

    if weights.get("hand_consistency", 0.0) > 0:
        wrist_pos, wrist_rot6d = FK.left_wrist_pose(pred_motion, rest_joints)
        tgt_pos = hand[..., 0:3]
        tgt_rot = hand[..., 6:12]
        parts["hand_consistency"] = Fn.mse_loss(wrist_pos, tgt_pos) + Fn.mse_loss(wrist_rot6d, tgt_rot)

    total = sum(weights.get(k, 0.0) * v for k, v in parts.items())
    return total, parts
