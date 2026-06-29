"""Round-trip and convention tests for h2b.representations.rotations.

These pin the rotation contract. If the upstream 6D basis differs, a
failure here is the first place it surfaces.
"""

import numpy as np
import pytest

from h2b.representations import rotations as R


def _random_rotations(n, seed=0):
    """n random SO(3) matrices via QR of a Gaussian (det fixed to +1)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, 3, 3))
    Q, _ = np.linalg.qr(M)
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1.0           # force det +1
    return Q


def test_matrices_are_valid_so3():
    Rs = _random_rotations(50)
    for Rm in Rs:
        assert R.is_rotation_matrix(Rm)


@pytest.mark.parametrize("conv", [R.R6D_PYTORCH3D, R.R6D_COLUMN])
def test_6d_matrix_roundtrip(conv):
    Rs = _random_rotations(200, seed=1)
    d6 = R.matrix_to_rotation_6d(Rs, convention=conv)
    Rr = R.rotation_6d_to_matrix(d6, convention=conv)
    assert np.allclose(Rs, Rr, atol=1e-10)


def test_6d_pytorch3d_packs_first_two_rows():
    Rs = _random_rotations(10, seed=2)
    d6 = R.matrix_to_rotation_6d(Rs, convention=R.R6D_PYTORCH3D)
    # first 3 == row 0, next 3 == row 1
    assert np.allclose(d6[..., 0:3], Rs[..., 0, :])
    assert np.allclose(d6[..., 3:6], Rs[..., 1, :])


def test_6d_column_packs_first_two_columns():
    Rs = _random_rotations(10, seed=3)
    d6 = R.matrix_to_rotation_6d(Rs, convention=R.R6D_COLUMN)
    assert np.allclose(d6[..., 0:3], Rs[..., :, 0])
    assert np.allclose(d6[..., 3:6], Rs[..., :, 1])


def test_6d_gram_schmidt_robust_to_nonorthogonal_input():
    # A perturbed (non-orthonormal) 6D should still decode to a valid rotation.
    rng = np.random.default_rng(4)
    d6 = rng.standard_normal((20, 6))
    Rm = R.rotation_6d_to_matrix(d6)
    for m in Rm:
        assert R.is_rotation_matrix(m, atol=1e-8)


def test_axis_angle_matrix_roundtrip():
    rng = np.random.default_rng(5)
    aa = rng.standard_normal((300, 3)) * 1.2          # angles up to ~ a few rad
    Rm = R.axis_angle_to_matrix(aa)
    for m in Rm:
        assert R.is_rotation_matrix(m, atol=1e-9)
    aa2 = R.matrix_to_axis_angle(Rm)
    Rm2 = R.axis_angle_to_matrix(aa2)               # compare rotations, not raw aa
    assert np.allclose(Rm, Rm2, atol=1e-8)


def test_axis_angle_zero():
    Rm = R.axis_angle_to_matrix(np.zeros((4, 3)))
    assert np.allclose(Rm, np.broadcast_to(np.eye(3), (4, 3, 3)), atol=1e-12)


def test_axis_angle_near_pi():
    # 180-degree rotation about a tilted axis — the tricky branch.
    axis = R.normalize(np.array([0.3, -0.7, 0.5]))
    aa = axis * np.pi
    Rm = R.axis_angle_to_matrix(aa)
    aa2 = R.matrix_to_axis_angle(Rm)
    assert np.allclose(R.axis_angle_to_matrix(aa2), Rm, atol=1e-6)


def test_quaternion_matrix_roundtrip_wxyz():
    Rs = _random_rotations(200, seed=6)
    q = R.matrix_to_quaternion(Rs)
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-9)
    Rr = R.quaternion_to_matrix(q)
    assert np.allclose(Rs, Rr, atol=1e-9)


def test_quaternion_identity_is_w1():
    q = R.matrix_to_quaternion(np.eye(3))
    assert np.allclose(np.abs(q), np.array([1.0, 0, 0, 0]), atol=1e-9)


def test_look_roundtrip():
    # forward/up pair -> matrix -> forward/up (AI4Animation adapter).
    fwd = R.normalize(np.array([1.0, 0.2, -0.3]))
    up = np.array([0.0, 0.0, 1.0])
    Rm = R.look_to_matrix(fwd, up)
    assert R.is_rotation_matrix(Rm, atol=1e-9)
    f2, u2 = R.matrix_to_look(Rm)
    assert np.allclose(f2, fwd, atol=1e-9)


def test_batched_shapes_preserved():
    d6 = np.zeros((2, 5, 6)); d6[..., 0] = 1.0; d6[..., 4] = 1.0
    Rm = R.rotation_6d_to_matrix(d6)
    assert Rm.shape == (2, 5, 3, 3)
    assert R.matrix_to_rotation_6d(Rm).shape == (2, 5, 6)
