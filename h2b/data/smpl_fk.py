"""Derive the 12D left-wrist signal from whole-body SMPL motion (training-pair extraction).

This is the M1 milestone: with no paired (12D ↔ SMPL) table-tennis data in existence,
we BUILD the input signal by forward kinematics from whole-body mocap (AMASS, or the
upstream GVHMR-reconstructed table-tennis SMPL).

Two layers, deliberately decoupled so the rotation math is testable WITHOUT smplx:

  * `global_joint_rotations(poses)` — pure NumPy. Global orientation of every joint
    depends ONLY on the per-joint axis-angle pose and the kinematic tree, NOT on body
    shape or the mesh. R_global[j] = R_global[parent[j]] @ R_local[j]. This yields the
    GLOBAL wrist orientation that goes into the 12D 6D slot (CONTRACT §2).
  * `extract_hand12(...)` — needs global joint POSITIONS, which DO depend on the SMPL
    model (betas, J_regressor). We inject those via a `joints_fn` callback so this module
    has no hard smplx dependency. `smplx_joints_fn(...)` adapts the real model;
    `synthetic_joints_fn` lets tests/prototypes run today.

Everything is world-frame (CONTRACT §1) and meters; velocity is m/s at the target fps.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..representations import frames as F
from ..representations import rotations as R

# joints_fn signature: (poses (T,72) aa, trans (T,3), betas (10,)) -> joints (T, 24, 3) world
JointsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def global_joint_rotations(
    poses: np.ndarray,
    parents: np.ndarray = F.SMPL_PARENTS,
) -> np.ndarray:
    """(T, 72) SMPL axis-angle pose -> (T, 24, 3, 3) GLOBAL joint rotation matrices.

    poses[:, 0:3] is the global pelvis orient (world); poses[:, 3*j:3*j+3] is joint j's
    parent-relative rotation. Composes down the tree. Pure NumPy — no model needed.
    """
    poses = np.asarray(poses, np.float64)
    T = poses.shape[0]
    J = len(parents)
    aa = poses.reshape(T, J, 3)
    R_local = R.axis_angle_to_matrix(aa)                 # (T, J, 3, 3)
    R_global = np.zeros_like(R_local)
    R_global[:, 0] = R_local[:, 0]                       # root = global_orient
    for j in range(1, J):
        R_global[:, j] = R_global[:, parents[j]] @ R_local[:, j]
    return R_global


def wrist_orientation_6d(
    poses: np.ndarray,
    joint: int = F.PADDLE_HAND_JOINT,
    convention: str = F.PROJECT_R6D,
) -> np.ndarray:
    """(T, 72) -> (T, 6): the GLOBAL wrist orientation as the 6D stored in the 12D vector."""
    Rg = global_joint_rotations(poses)                   # (T, J, 3, 3)
    return R.matrix_to_rotation_6d(Rg[:, joint], convention=convention)


def extract_hand12(
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    joints_fn: JointsFn,
    fps: float = 30.0,
    joint: int = F.PADDLE_HAND_JOINT,
    convention: str = F.PROJECT_R6D,
) -> np.ndarray:
    """Build the (T, 12) left-wrist signal = [pos(3), lin_vel(3), rot6d(6)], world frame.

    `joints_fn` returns global joint positions for the SMPL model (inject smplx or a stub).
    Position = wrist joint world position; velocity = central finite difference (m/s);
    orientation = GLOBAL wrist rotation as 6D.
    """
    poses = np.asarray(poses, np.float64)
    trans = np.asarray(trans, np.float64)
    joints = np.asarray(joints_fn(poses, trans, betas), np.float64)   # (T, 24, 3)
    pos = joints[:, joint, :]                              # (T, 3) world
    vel = F.finite_diff_velocity(pos, fps)                 # (T, 3) m/s
    rot6d = wrist_orientation_6d(poses, joint=joint, convention=convention)
    return np.concatenate([pos, vel, rot6d], axis=-1)


# --------------------------------------------------------------------------- #
# joints_fn adapters
# --------------------------------------------------------------------------- #
def synthetic_joints_fn(
    poses: np.ndarray, trans: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    """A model-free stand-in: posed positions of a fixed rest skeleton.

    Uses a crude rest-pose offset table so prototypes/tests run before the real SMPL
    model lands. NOT anatomically exact — replace with `smplx_joints_fn` for real data.
    Positions are obtained by FK on the rest offsets through the global rotations + root
    translation, which is the correct posing operation given rest joints.
    """
    poses = np.asarray(poses, np.float64)
    trans = np.asarray(trans, np.float64)
    T = poses.shape[0]
    rest = _approx_rest_joints()                          # (24, 3) rest positions
    parents = F.SMPL_PARENTS
    R_global = global_joint_rotations(poses)              # (T, 24, 3, 3)
    # Standard SMPL posing of joints: p_j = R_global[parent] @ (rest_j - rest_parent) + p_parent
    pos = np.zeros((T, 24, 3))
    pos[:, 0] = trans + rest[0]
    for j in range(1, 24):
        p = parents[j]
        offset = rest[j] - rest[p]                        # (3,)
        pos[:, j] = (R_global[:, p] @ offset) + pos[:, p]
    return pos


def smplx_joints_fn_factory(model_dir: str, gender: str = "neutral", num_betas: int = 10):
    """Return a `joints_fn` backed by the real SMPL model (requires smplx + torch).

    Imports are lazy so importing this module never requires smplx. Wire `model_dir`
    to the upstream SMPL build once shipped (CONTRACT §6.4).
    """
    def _fn(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray) -> np.ndarray:
        import torch  # noqa: WPS433 (lazy)
        import smplx  # noqa: WPS433
        T = poses.shape[0]
        model = smplx.create(
            model_dir, model_type="smpl", gender=gender,
            num_betas=num_betas, batch_size=T,
        )
        out = model(
            global_orient=torch.tensor(poses[:, 0:3], dtype=torch.float32),
            body_pose=torch.tensor(poses[:, 3:72], dtype=torch.float32),
            betas=torch.tensor(np.broadcast_to(betas, (T, num_betas)).copy(), dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
        )
        return out.joints[:, :24, :].detach().cpu().numpy()
    return _fn


def _approx_rest_joints() -> np.ndarray:
    """Crude SMPL rest-pose joint offsets (meters), neutral-ish. Placeholder for tests.

    Only relative parent→child offsets matter for posing; absolute values are roughly
    anthropometric so cycle-consistency checks are meaningful. Replace with the model's
    true rest joints (model.J) for production extraction.
    """
    j = np.zeros((24, 3))
    # y-up-ish in SMPL's own rest frame; values are approximate, left/right on x.
    j[0] = [0.00, 0.00, 0.00]      # pelvis
    j[1] = [0.06, -0.09, 0.00]     # left_hip
    j[2] = [-0.06, -0.09, 0.00]    # right_hip
    j[3] = [0.00, 0.12, 0.00]      # spine1
    j[4] = [0.10, -0.46, 0.00]     # left_knee
    j[5] = [-0.10, -0.46, 0.00]    # right_knee
    j[6] = [0.00, 0.14, 0.00]      # spine2
    j[7] = [0.10, -0.86, 0.00]     # left_ankle
    j[8] = [-0.10, -0.86, 0.00]    # right_ankle
    j[9] = [0.00, 0.26, 0.00]      # spine3
    j[10] = [0.11, -0.91, 0.12]    # left_foot
    j[11] = [-0.11, -0.91, 0.12]   # right_foot
    j[12] = [0.00, 0.48, 0.00]     # neck
    j[13] = [0.07, 0.40, 0.00]     # left_collar
    j[14] = [-0.07, 0.40, 0.00]    # right_collar
    j[15] = [0.00, 0.58, 0.02]     # head
    j[16] = [0.18, 0.42, 0.00]     # left_shoulder
    j[17] = [-0.18, 0.42, 0.00]    # right_shoulder
    j[18] = [0.44, 0.42, 0.00]     # left_elbow
    j[19] = [-0.44, 0.42, 0.00]    # right_elbow
    j[20] = [0.69, 0.42, 0.00]     # left_wrist  (paddle hand)
    j[21] = [-0.69, 0.42, 0.00]    # right_wrist
    j[22] = [0.79, 0.42, 0.00]     # left_hand
    j[23] = [-0.79, 0.42, 0.00]    # right_hand
    return j
