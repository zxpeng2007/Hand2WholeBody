"""Export generated SMPL motion to an AMASS-style .npz for GMR → HoloMotion (Stage 3).

This is the Stage-2 → Stage-3 handoff (CONTRACT §3). HoloMotion does not consume SMPL
directly; GMR retargets human SMPL → Unitree G1. The documented entry artifact for that
pipeline is an AMASS-style SMPL .npz, which is exactly what we write here.

Implemented in NumPy (no torch needed) so it runs anywhere. The model produces rotations
in 6D internally; convert to axis-angle (via h2b.representations.rotations) before
calling this.
"""

from __future__ import annotations

import numpy as np

from ..representations import rotations as R
from ..representations import frames as Fr


def smpl_motion_to_amass_npz(
    path: str,
    poses_aa: np.ndarray,           # (T, 72) axis-angle: [global_orient(3) | body_pose(69)]
    trans: np.ndarray,              # (T, 3) world-frame pelvis translation, meters
    betas: np.ndarray,              # (10,) body shape
    fps: int = 30,
    gender: str = "neutral",
    contacts: np.ndarray | None = None,   # (T, 4) optional foot-contact flags
) -> str:
    """Write an AMASS-style .npz and return the path. Validates shapes against the contract."""
    poses_aa = np.asarray(poses_aa, np.float64)
    trans = np.asarray(trans, np.float64)
    betas = np.asarray(betas, np.float64).reshape(-1)
    T = poses_aa.shape[0]
    assert poses_aa.shape == (T, 72), f"poses must be (T,72), got {poses_aa.shape}"
    assert trans.shape == (T, 3), f"trans must be (T,3), got {trans.shape}"
    payload = dict(
        poses=poses_aa.astype(np.float32),
        trans=trans.astype(np.float32),
        betas=betas.astype(np.float32),
        gender=gender,
        mocap_frame_rate=np.array(fps),
    )
    if contacts is not None:
        payload["contacts"] = np.asarray(contacts, np.float32)
    np.savez(path, **payload)
    return path


def smpl_motion_to_smplx_npz(
    path: str,
    poses_aa: np.ndarray,           # (T, 72) SMPL axis-angle [global_orient(3) | body_pose(69)]
    trans: np.ndarray,              # (T, 3) world-frame pelvis translation, meters
    betas: np.ndarray,              # (10,) SMPL shape
    fps: int = 30,
    gender: str = "neutral",
    height_m: float | None = None,
) -> str:
    """Write a GMR-ready SMPL-X-style .npz (the Stage-3 ingest format).

    GMR's load_smplx_file does NOT read AMASS `poses (T,72)`; it reads
    {root_orient (T,3), pose_body (T,63), betas (16,), trans (T,3), gender, mocap_frame_rate}.
    We map our SMPL params directly (our hand joints are already zero, so dropping
    poses[:,66:72] is lossless). This is the preferred zero-glue path — no separate
    smpl_to_smplx.py pass needed.

    `betas[0]` drives GMR's auto-scale: human height = 1.66 + 0.1*betas[0], ratio = height/1.8.
    Pass `height_m` to set betas[0] = (height_m - 1.66)/0.1 (e.g. 1.8 -> betas[0]=1.4); else the
    padded SMPL betas are used (all-zero -> 1.66 m).
    """
    poses_aa = np.asarray(poses_aa, np.float64)
    trans = np.asarray(trans, np.float64)
    T = poses_aa.shape[0]
    assert poses_aa.shape == (T, 72), f"poses must be (T,72), got {poses_aa.shape}"
    betas16 = np.zeros(16, np.float32)
    betas16[:10] = np.asarray(betas, np.float64).reshape(-1)[:10]
    if height_m is not None:
        betas16[0] = (height_m - 1.66) / 0.1
    np.savez(
        path,
        root_orient=poses_aa[:, 0:3].astype(np.float32),     # (T,3)
        pose_body=poses_aa[:, 3:66].astype(np.float32),      # (T,63) = 21 body joints
        betas=betas16,                                       # (16,)
        trans=trans.astype(np.float32),                      # (T,3)
        gender=gender,
        mocap_frame_rate=np.array(fps),
    )
    return path


def motion6d_to_aa(root_orient_6d: np.ndarray, body_rot6d: np.ndarray) -> np.ndarray:
    """Convert model output (6D root + 6D body joints) to SMPL (T, 72) axis-angle.

    root_orient_6d: (T, 6)         body_rot6d: (T, 23, 6)  -> (T, 72)
    """
    conv = Fr.PROJECT_R6D
    root_R = R.rotation_6d_to_matrix(root_orient_6d, convention=conv)   # (T,3,3)
    body_R = R.rotation_6d_to_matrix(body_rot6d, convention=conv)       # (T,23,3,3)
    root_aa = R.matrix_to_axis_angle(root_R)                   # (T,3)
    body_aa = R.matrix_to_axis_angle(body_R)                   # (T,23,3)
    T = root_aa.shape[0]
    return np.concatenate([root_aa, body_aa.reshape(T, 69)], axis=-1)
