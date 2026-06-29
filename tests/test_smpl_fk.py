"""Tests for the SMPL -> 12D forward-kinematics extractor (M1)."""

import numpy as np

from h2b.data import smpl_fk as FK
from h2b.representations import frames as F
from h2b.representations import rotations as R


def test_global_rotations_zero_pose_is_identity():
    poses = np.zeros((4, 72))
    Rg = FK.global_joint_rotations(poses)
    assert Rg.shape == (4, 24, 3, 3)
    assert np.allclose(Rg, np.broadcast_to(np.eye(3), (4, 24, 3, 3)), atol=1e-12)


def test_global_rotation_composes_down_the_tree():
    # Rotate only the root; every joint's global rotation must equal the root's.
    T = 3
    poses = np.zeros((T, 72))
    root_aa = np.array([0.0, 0.0, 0.5])
    poses[:, 0:3] = root_aa
    Rg = FK.global_joint_rotations(poses)
    R_root = R.axis_angle_to_matrix(root_aa)
    for j in range(24):
        assert np.allclose(Rg[:, j], R_root, atol=1e-12)


def test_global_rotation_chain_two_joints():
    # Root rotation composed with a child rotation: R_global[child] = R_root @ R_child.
    poses = np.zeros((1, 72))
    poses[0, 0:3] = [0.0, 0.3, 0.0]                 # pelvis
    poses[0, 3 * 16:3 * 16 + 3] = [0.0, 0.0, 0.4]   # left_shoulder (parent chain -> pelvis)
    Rg = FK.global_joint_rotations(poses)
    # left_wrist (20) inherits the shoulder; its global = product along 0->...->20.
    # Just assert it is a valid rotation and not identity.
    assert R.is_rotation_matrix(Rg[0, 20], atol=1e-9)
    assert not np.allclose(Rg[0, 20], np.eye(3))


def test_wrist_orientation_6d_decodes_to_global_rotation():
    rng = np.random.default_rng(0)
    poses = rng.standard_normal((8, 72)) * 0.3
    d6 = FK.wrist_orientation_6d(poses, joint=F.PADDLE_HAND_JOINT)
    Rg = FK.global_joint_rotations(poses)[:, F.PADDLE_HAND_JOINT]
    Rdec = R.rotation_6d_to_matrix(d6, convention=F.PROJECT_R6D)
    assert np.allclose(Rg, Rdec, atol=1e-9)         # cycle-consistency under project convention


def test_extract_hand12_shapes_and_consistency():
    rng = np.random.default_rng(1)
    T = 30
    poses = rng.standard_normal((T, 72)) * 0.2
    trans = np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + [0.0, 0.0, 1.0]
    betas = np.zeros(10)
    h = FK.extract_hand12(poses, trans, betas, FK.synthetic_joints_fn, fps=30.0)
    assert h.shape == (T, F.HAND12_DIM)
    # the stored 6D must equal the independently computed global wrist orientation
    Rfrom12 = F.matrix_from_hand12_rot(h)
    Rg = FK.global_joint_rotations(poses)[:, F.PADDLE_HAND_JOINT]
    assert np.allclose(Rfrom12, Rg, atol=1e-9)


def test_extract_hand12_static_pose_zero_velocity():
    T = 20
    poses = np.tile(np.zeros(72), (T, 1))
    trans = np.tile(np.array([0.1, 0.2, 1.0]), (T, 1))   # no motion
    h = FK.extract_hand12(poses, trans, np.zeros(10), FK.synthetic_joints_fn, fps=30.0)
    _, vel, _ = F.unpack_hand12(h)
    assert np.allclose(vel, 0.0, atol=1e-9)


def test_synthetic_joints_root_translation_tracks():
    # Pelvis position must follow `trans` (plus the rest pelvis offset, which is 0).
    T = 5
    poses = np.zeros((T, 72))
    trans = np.arange(T)[:, None] * np.array([0.1, 0.0, 0.0]) + [0.0, 0.0, 1.0]
    joints = FK.synthetic_joints_fn(poses, trans, np.zeros(10))
    assert np.allclose(joints[:, 0, :], trans, atol=1e-9)
