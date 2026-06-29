"""Body-motion representation for Hand2WholeBody model I/O — single source of truth.

The model predicts, per frame, a 135-D vector:

    [ root_trans(3) | 22 joint rotations x 6D (132) ]   = 135

  * root_trans : pelvis world (or canonicalized) translation, meters.
  * 22 joints  : SMPL joints 0..21 (we DROP the two hand joints 22/23 — rigid wrist,
                 no fingers per CONTRACT §3). Joint 0's 6D is the GLOBAL pelvis
                 orientation (global_orient); joints 1..21 are parent-relative local
                 rotations, exactly as SMPL stores them. The left wrist (joint 20) is
                 included, so FK to the paddle hand stays inside this 22-joint subtree.
  * 6D packing : Zhou-2019 COLUMN convention (frames.PROJECT_R6D) everywhere.

`smpl72_to_motion` / `motion_to_smpl72` are the only places that convert between SMPL's
native 72-D axis-angle pose and this representation. Export pads joints 22/23 back to
identity (zero axis-angle).
"""

from __future__ import annotations

import numpy as np

from . import frames as F
from . import rotations as R

NUM_BODY_JOINTS = 22                 # SMPL joints 0..21 (drop hands 22, 23)
MOTION_DIM = 3 + NUM_BODY_JOINTS * 6  # 135
B_TRANS = slice(0, 3)
B_ROT6D = slice(3, MOTION_DIM)
# Parents for the 22-joint subtree (self-contained: every parent index < 22).
BODY_PARENTS = F.SMPL_PARENTS[:NUM_BODY_JOINTS].copy()


def smpl72_to_motion(poses72: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(T, 72) axis-angle + (T, 3) trans -> (T, 135) motion vector (PROJECT_R6D)."""
    poses72 = np.asarray(poses72, np.float64)
    trans = np.asarray(trans, np.float64)
    T = poses72.shape[0]
    aa = poses72[:, : NUM_BODY_JOINTS * 3].reshape(T, NUM_BODY_JOINTS, 3)
    rot6d = R.matrix_to_rotation_6d(R.axis_angle_to_matrix(aa), convention=F.PROJECT_R6D)
    return np.concatenate([trans, rot6d.reshape(T, NUM_BODY_JOINTS * 6)], axis=-1)


def motion_to_smpl72(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 135) -> (poses72 (T,72) axis-angle, trans (T,3)). Hand joints 22/23 -> 0."""
    motion = np.asarray(motion, np.float64)
    T = motion.shape[0]
    trans = motion[:, B_TRANS]
    rot6d = motion[:, B_ROT6D].reshape(T, NUM_BODY_JOINTS, 6)
    aa = R.matrix_to_axis_angle(R.rotation_6d_to_matrix(rot6d, convention=F.PROJECT_R6D))
    poses72 = np.zeros((T, 72), np.float64)
    poses72[:, : NUM_BODY_JOINTS * 3] = aa.reshape(T, NUM_BODY_JOINTS * 3)
    return poses72, trans


def unpack_motion(motion: np.ndarray):
    """Return (trans (...,3), rot6d (..., 22, 6))."""
    motion = np.asarray(motion, np.float64)
    trans = motion[..., B_TRANS]
    rot6d = motion[..., B_ROT6D].reshape(*motion.shape[:-1], NUM_BODY_JOINTS, 6)
    return trans, rot6d
