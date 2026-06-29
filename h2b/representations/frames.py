"""World frame, SMPL skeleton, and the 12D-hand canonicalization for Hand2Body.

WORLD FRAME (pinned by assets/urdf/table.urdf — do not redefine elsewhere):
  * origin: on the FLOOR (z = 0), directly below the table center.
  * +x : along the table LENGTH (2.740 m).
  * +y : along the table WIDTH  (1.525 m).
  * +z : UP.   gravity = -z.
  * table top surface at z = 0.76 m; net is the plane x = 0.
The left player's paddle hand is the SMPL LEFT WRIST (joint 20). Forehand vs.
backhand is encoded by the *global* wrist orientation relative to this frame —
which is exactly why the 12D rotation is GLOBAL, not SMPL-local, and why we must
NOT yaw-canonicalize the orientation away.
"""

from __future__ import annotations

import numpy as np

from .rotations import matrix_to_rotation_6d, rotation_6d_to_matrix, R6D_COLUMN

# ---- PROJECT 6D CONVENTION — single source of truth ---------------------- #
# CONFIRMED 2026-06-29: the upstream 12D wrist orientation uses the Zhou et al. 2019
# COLUMN packing (first two columns of R). Everything in h2b (input extraction,
# decode, internal body rotations, export) reads this constant — change it in ONE
# place if the contract ever changes. Mirrors configs/default.yaml rot6d_convention.
PROJECT_R6D = R6D_COLUMN

# --------------------------------------------------------------------------- #
# World / table geometry (meters) — mirrors assets/urdf/table.urdf
# --------------------------------------------------------------------------- #
TABLE_LENGTH_X = 2.740
TABLE_WIDTH_Y = 1.525
TABLE_TOP_Z = 0.76
NET_PLANE_X = 0.0
GRAVITY = np.array([0.0, 0.0, -9.81])

# --------------------------------------------------------------------------- #
# SMPL skeleton (24 joints). Plain SMPL is sufficient (rigid wrist, no fingers).
# Kinematic parents from the SMPL body model; index 0 (pelvis) is the root.
# --------------------------------------------------------------------------- #
SMPL_NUM_JOINTS = 24
SMPL_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
    12, 13, 14, 16, 17, 18, 19, 20, 21,
], dtype=np.int64)

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

PADDLE_HAND_JOINT = 20            # left_wrist — the paddle hand for this project
LEFT_WRIST = 20
RIGHT_WRIST = 21

# SMPL body-model foot joints, for contact / no-slide losses later.
FOOT_JOINTS = (7, 8, 10, 11)     # ankles + feet


# --------------------------------------------------------------------------- #
# 12D hand signal layout  [ pos(3) | lin_vel(3) | rot6D(6) ]
# --------------------------------------------------------------------------- #
HAND12_POS = slice(0, 3)
HAND12_VEL = slice(3, 6)
HAND12_ROT6D = slice(6, 12)
HAND12_DIM = 12


def pack_hand12(pos: np.ndarray, vel: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """Concatenate the three sub-vectors into the (..., 12) signal."""
    pos, vel, rot6d = map(lambda a: np.asarray(a, np.float64), (pos, vel, rot6d))
    return np.concatenate([pos, vel, rot6d], axis=-1)


def unpack_hand12(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(h, np.float64)
    return h[..., HAND12_POS], h[..., HAND12_VEL], h[..., HAND12_ROT6D]


def finite_diff_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference linear velocity (m/s) for a (T, 3) position track.

    Endpoints use forward/backward difference. `fps` is the canonical generation
    rate (see configs/default.yaml: 30 Hz). Velocity is in world units/second so
    it is frame-rate independent — match this in the live generator.
    """
    pos = np.asarray(pos, np.float64)
    T = pos.shape[0]
    vel = np.zeros_like(pos)
    if T == 1:
        return vel
    vel[1:-1] = (pos[2:] - pos[:-2]) * (0.5 * fps)
    vel[0] = (pos[1] - pos[0]) * fps
    vel[-1] = (pos[-1] - pos[-2]) * fps
    return vel


# --------------------------------------------------------------------------- #
# Canonicalization for the network input.
#
# DESIGN DECISION (locked 2026-06-29): position is expressed relative to a chosen
# anchor (root XY or a fixed play-area origin) so the network is translation-robust,
# but ORIENTATION stays in the WORLD frame because forehand/backhand is defined by
# the paddle's global facing relative to the table. We therefore yaw-align position
# only, never the wrist orientation. `mode="none"` keeps everything global (the raw
# inter-stage contract); `mode="root_relative_pos"` subtracts a per-window anchor.
# --------------------------------------------------------------------------- #
def canonicalize_hand12(
    h: np.ndarray,
    anchor_xyz: np.ndarray | None = None,
    mode: str = "root_relative_pos",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (canonical_signal, anchor_used). Inverse: `decanonicalize_hand12`.

    The anchor is a single (3,) world point (e.g. the first-frame pelvis, or the
    fixed play-area origin). Velocity and rotation are unchanged — only position
    is shifted, and only if mode requests it. We deliberately do NOT remove global
    yaw from the orientation (see module docstring).
    """
    h = np.asarray(h, np.float64).copy()
    if mode == "none":
        return h, np.zeros(3)
    if mode == "root_relative_pos":
        if anchor_xyz is None:
            anchor_xyz = h[..., 0, HAND12_POS] if h.ndim > 1 else h[HAND12_POS]
        # .copy() is essential: a view here would alias into `h` and be zeroed by
        # the in-place subtraction below, silently corrupting the round-trip.
        anchor_xyz = np.array(anchor_xyz, np.float64)
        h[..., HAND12_POS] = h[..., HAND12_POS] - anchor_xyz
        return h, anchor_xyz
    raise ValueError(f"unknown canonicalization mode: {mode!r}")


def decanonicalize_hand12(h: np.ndarray, anchor_xyz: np.ndarray) -> np.ndarray:
    """Undo `canonicalize_hand12` to recover the world-frame signal."""
    h = np.asarray(h, np.float64).copy()
    h[..., HAND12_POS] = h[..., HAND12_POS] + np.asarray(anchor_xyz, np.float64)
    return h


def global_orientation_6d(R_global_wrist: np.ndarray) -> np.ndarray:
    """Helper: world-frame wrist rotation matrix -> the 6D stored in the 12D vector."""
    return matrix_to_rotation_6d(R_global_wrist, convention=PROJECT_R6D)


def matrix_from_hand12_rot(h: np.ndarray) -> np.ndarray:
    """Recover the global wrist rotation matrix from a 12D signal's 6D slot."""
    _, _, rot6d = unpack_hand12(h)
    return rotation_6d_to_matrix(rot6d, convention=PROJECT_R6D)
