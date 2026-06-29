"""Tests for h2b.representations.frames: 12D packing, velocity, canonicalization."""

import numpy as np

from h2b.representations import frames as F
from h2b.representations import rotations as R


def test_hand12_pack_unpack_roundtrip():
    pos = np.array([0.1, -0.2, 0.9])
    vel = np.array([1.0, 0.0, -0.5])
    rot6d = np.array([1.0, 0, 0, 0, 1.0, 0])
    h = F.pack_hand12(pos, vel, rot6d)
    assert h.shape == (12,)
    p, v, r = F.unpack_hand12(h)
    assert np.allclose(p, pos) and np.allclose(v, vel) and np.allclose(r, rot6d)


def test_finite_diff_velocity_constant_motion():
    fps = 30.0
    T = 10
    step = np.array([0.1, 0.0, 0.0])              # 0.1 m/frame -> 3.0 m/s
    pos = np.cumsum(np.broadcast_to(step, (T, 3)), axis=0)
    vel = F.finite_diff_velocity(pos, fps)
    assert np.allclose(vel, np.array([3.0, 0.0, 0.0]), atol=1e-9)


def test_finite_diff_single_frame():
    vel = F.finite_diff_velocity(np.array([[0.0, 0.0, 1.0]]), 30.0)
    assert np.allclose(vel, 0.0)


def test_canonicalize_pos_only_keeps_orientation():
    rng = np.random.default_rng(0)
    T = 16
    pos = rng.standard_normal((T, 3)) + np.array([0.0, 0.0, 1.2])
    vel = rng.standard_normal((T, 3))
    rot6d = np.tile(np.array([1.0, 0, 0, 0, 1.0, 0]), (T, 1))
    h = np.concatenate([pos, vel, rot6d], axis=-1)
    hc, anchor = F.canonicalize_hand12(h, mode="root_relative_pos")
    # orientation + velocity untouched
    assert np.allclose(hc[..., F.HAND12_ROT6D], h[..., F.HAND12_ROT6D])
    assert np.allclose(hc[..., F.HAND12_VEL], h[..., F.HAND12_VEL])
    # first-frame position becomes the origin
    assert np.allclose(hc[0, F.HAND12_POS], 0.0, atol=1e-9)
    # round-trip
    hr = F.decanonicalize_hand12(hc, anchor)
    assert np.allclose(hr, h, atol=1e-12)


def test_canonicalize_none_is_identity():
    h = np.arange(12, dtype=float)
    hc, anchor = F.canonicalize_hand12(h, mode="none")
    assert np.allclose(hc, h) and np.allclose(anchor, 0.0)


def test_global_orientation_6d_roundtrip():
    Rm = R.axis_angle_to_matrix(np.array([0.2, -0.4, 0.1]))
    d6 = F.global_orientation_6d(Rm)
    h = F.pack_hand12(np.zeros(3), np.zeros(3), d6)
    Rr = F.matrix_from_hand12_rot(h)
    assert np.allclose(Rm, Rr, atol=1e-10)


def test_smpl_parents_wellformed():
    assert len(F.SMPL_PARENTS) == F.SMPL_NUM_JOINTS == 24
    assert F.SMPL_PARENTS[0] == -1
    assert F.SMPL_JOINT_NAMES[F.PADDLE_HAND_JOINT] == "left_wrist"
    # every non-root parent precedes its child (topological order)
    for j in range(1, 24):
        assert F.SMPL_PARENTS[j] < j


def test_project_convention_is_zhou_column():
    # CONFIRMED contract (2026-06-29). If this flips, the upstream 12D no longer matches.
    assert F.PROJECT_R6D == R.R6D_COLUMN


def test_hand12_decodes_under_column_and_row_would_be_wrong():
    # The 12D 6D is Zhou-columns: decoding it as columns recovers the true rotation,
    # and decoding it as pytorch3d-rows gives a DIFFERENT matrix — so a convention slip
    # cannot pass silently.
    Rm = R.axis_angle_to_matrix(np.array([0.3, -0.5, 0.2]))
    d6 = R.matrix_to_rotation_6d(Rm, convention=R.R6D_COLUMN)
    h = F.pack_hand12(np.zeros(3), np.zeros(3), d6)
    assert np.allclose(F.matrix_from_hand12_rot(h), Rm, atol=1e-9)       # column = correct
    wrong = R.rotation_6d_to_matrix(d6, convention=R.R6D_PYTORCH3D)
    assert not np.allclose(wrong, Rm, atol=1e-6)                          # row = wrong


def test_world_constants_match_urdf():
    assert F.TABLE_TOP_Z == 0.76
    assert abs(F.TABLE_LENGTH_X - 2.740) < 1e-9
    assert abs(F.TABLE_WIDTH_Y - 1.525) < 1e-9
