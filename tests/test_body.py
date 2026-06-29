"""Tests for the 135-D body-motion representation (h2b.representations.body)."""

import numpy as np

from h2b.representations import body as B
from h2b.representations import frames as F
from h2b.representations import rotations as R


def test_motion_dim_constants():
    assert B.NUM_BODY_JOINTS == 22
    assert B.MOTION_DIM == 135
    assert F.LEFT_WRIST < B.NUM_BODY_JOINTS          # paddle hand is inside the subtree
    assert len(B.BODY_PARENTS) == 22 and B.BODY_PARENTS.max() < 22


def test_smpl72_motion_roundtrip_rotations_and_trans():
    rng = np.random.default_rng(0)
    T = 16
    poses = rng.standard_normal((T, 72)) * 0.3
    poses[:, 66:72] = 0.0                            # hand joints unused -> keep zero
    trans = rng.standard_normal((T, 3))
    motion = B.smpl72_to_motion(poses, trans)
    assert motion.shape == (T, 135)
    poses2, trans2 = B.motion_to_smpl72(motion)
    assert np.allclose(trans2, trans, atol=1e-9)
    # compare rotations as matrices (axis-angle is non-unique) for the 22 modeled joints
    R1 = R.axis_angle_to_matrix(poses[:, :66].reshape(T, 22, 3))
    R2 = R.axis_angle_to_matrix(poses2[:, :66].reshape(T, 22, 3))
    assert np.allclose(R1, R2, atol=1e-7)
    # hand joints stay zero on the way back
    assert np.allclose(poses2[:, 66:72], 0.0)


def test_unpack_motion_shapes():
    motion = np.zeros((5, 135))
    trans, rot6d = B.unpack_motion(motion)
    assert trans.shape == (5, 3) and rot6d.shape == (5, 22, 6)
