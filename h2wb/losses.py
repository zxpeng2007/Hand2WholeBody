"""Training objective for Hand2WholeBody (weights in configs/default.yaml `loss`).

Operates on the model's 135-D motion output (h2wb.representations.body) and the input 12D
hand signal. Terms:
  * trans            — root translation L2.
  * rot6d            — 6D joint-rotation L2 (all 22 joints incl. global_orient).
  * velocity         — first/second-difference penalty (smoothness -> trackable by HoloMotion).
  * fk_joint         — 3D joint-position L2 after differentiable FK (vs GT pose).
  * hand_consistency — THE key term: FK the predicted pose, recover the GLOBAL left-wrist
                       position + 6D orientation, and match them to the input 12D. Forces
                       the body to actually honor the conditioning hand.

torch-guarded so importing the package never requires torch.
"""

from __future__ import annotations

from .representations import body as B

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


def compute_losses(pred_motion, gt_motion, hand, weights, rest_joints=None):
    """Return (total, parts dict). Shapes: pred/gt (B,L,135), hand (B,L,12).

    `weights` keys: trans, rot6d, velocity, fk_joint, hand_consistency. Missing -> 0.
    """
    parts = {}
    parts["trans"] = Fn.mse_loss(pred_motion[..., B.B_TRANS], gt_motion[..., B.B_TRANS])
    parts["rot6d"] = Fn.mse_loss(pred_motion[..., B.B_ROT6D], gt_motion[..., B.B_ROT6D])
    parts["velocity"] = velocity_loss(pred_motion)

    if weights.get("fk_joint", 0.0) > 0:
        _, pred_pos = FK.motion_to_joints(pred_motion, rest_joints)
        with torch.no_grad():
            _, gt_pos = FK.motion_to_joints(gt_motion, rest_joints)
        parts["fk_joint"] = Fn.mse_loss(pred_pos, gt_pos)

    if weights.get("hand_consistency", 0.0) > 0:
        wrist_pos, wrist_rot6d = FK.left_wrist_pose(pred_motion, rest_joints)
        tgt_pos = hand[..., 0:3]
        tgt_rot = hand[..., 6:12]
        parts["hand_consistency"] = Fn.mse_loss(wrist_pos, tgt_pos) + Fn.mse_loss(wrist_rot6d, tgt_rot)

    total = sum(weights.get(k, 0.0) * v for k, v in parts.items())
    return total, parts
