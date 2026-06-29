"""Rotation conversions for Hand2Body — pure NumPy, batched, dependency-light.

Everything here is intentionally written without torch so the representation core
can be unit-tested in any environment (the model/training code will mirror these
with torch ops). Conventions are made *explicit* because the project bridges three
ecosystems that each pick a different one:

  * Zhou et al. 2019 / PyTorch3D 6D rotation  -> our internal default ("r6d").
    PyTorch3D packs the first two ROWS of R (not columns). We match it byte-for-byte
    so values are interchangeable once torch is in the stack.
  * AI4Animation (the upstream ai4animationpy generator) stores orientation as a
    forward/up pair AxisZ,AxisY decoded via Rotation.Look(z, y). See `look_to_matrix`
    / `matrix_to_look`. The upstream 12D wrist 6D MAY use this basis — keep an
    adapter, do not assume.
  * GMR / HoloMotion consume quaternions in (w, x, y, z) order. See
    `matrix_to_quaternion` / `quaternion_to_matrix`.

All functions accept arbitrary leading batch dims; the rotation lives in the last
1-2 axes. Inputs are coerced to float64 for numerical stability.

THE 6D CONVENTION IS A CONTRACT, NOT A DETAIL. If a round-trip test ever fails after
the upstream pins their convention, fix it HERE and nowhere else.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8

# Rotation-representation registry — keep names stable; they appear in configs.
R6D_PYTORCH3D = "pytorch3d_row"   # first two rows of R (PyTorch3D / our default)
R6D_COLUMN = "zhou_column"        # first two columns of R (literal Zhou et al. 2019)


def normalize(v: np.ndarray, axis: int = -1, eps: float = _EPS) -> np.ndarray:
    """L2-normalize along `axis`, guarding against zero-length vectors."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


# --------------------------------------------------------------------------- #
# 6D continuous rotation  <->  rotation matrix  (Zhou et al. 2019)
# --------------------------------------------------------------------------- #
def rotation_6d_to_matrix(d6: np.ndarray, convention: str = R6D_PYTORCH3D) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt.

    For `pytorch3d_row`, the 6 numbers are [m1 (3) | m2 (3)] = first two ROWS of R;
    the recovered rows are b1, b2, b3 = b1 x b2. For `zhou_column` they are the first
    two COLUMNS instead (transpose of the above). Both are continuous on SO(3).
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = normalize(a1)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(a2_proj)
    b3 = np.cross(b1, b2)
    if convention == R6D_PYTORCH3D:
        # b1, b2, b3 are the ROWS of R.
        return np.stack((b1, b2, b3), axis=-2)
    if convention == R6D_COLUMN:
        # b1, b2, b3 are the COLUMNS of R.
        return np.stack((b1, b2, b3), axis=-1)
    raise ValueError(f"unknown 6D convention: {convention!r}")


def matrix_to_rotation_6d(R: np.ndarray, convention: str = R6D_PYTORCH3D) -> np.ndarray:
    """(..., 3, 3) -> (..., 6). Inverse of `rotation_6d_to_matrix` (up to GS)."""
    R = np.asarray(R, dtype=np.float64)
    if convention == R6D_PYTORCH3D:
        rows = R[..., 0:2, :]                      # first two rows
        return rows.reshape(*R.shape[:-2], 6)
    if convention == R6D_COLUMN:
        cols = R[..., :, 0:2]                       # first two columns
        return np.swapaxes(cols, -1, -2).reshape(*R.shape[:-2], 6)
    raise ValueError(f"unknown 6D convention: {convention!r}")


# --------------------------------------------------------------------------- #
# axis-angle (SMPL pose params)  <->  rotation matrix  (Rodrigues)
# --------------------------------------------------------------------------- #
def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3). SMPL stores pose as axis-angle."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)          # (..., 1)
    k = aa / np.maximum(theta, _EPS)                            # unit axis
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1)
    K = K.reshape(*k.shape[:-1], 3, 3)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s = np.sin(theta)[..., None]                                # (..., 1, 1)
    c = np.cos(theta)[..., None]
    return I + s * K + (1.0 - c) * (K @ K)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) axis-angle (rotation vector). Stable at all angles.

    Routes through a quaternion (Shepperd's method) so it is correct near theta=pi,
    where the skew-symmetric part vanishes and the naive formula loses the axis sign.
    """
    q = matrix_to_quaternion(R)                  # (..., 4) wxyz, unit
    # Canonicalize to the short rotation: force w >= 0 so the angle is in [0, pi].
    sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
    q = q * sign
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)                    # (...), in [0, pi]
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    small = sin_half < 1e-8
    # ratio = angle / sin_half, with the small-angle limit -> 2 (since angle ~ 2*sin_half)
    ratio = np.where(small, 2.0, angle / np.maximum(sin_half, _EPS))
    return q[..., 1:4] * ratio[..., None]


# --------------------------------------------------------------------------- #
# quaternion (w, x, y, z)  <->  rotation matrix   (GMR / HoloMotion order)
# --------------------------------------------------------------------------- #
def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) quaternion in (w, x, y, z). GMR expects wxyz.

    Shepperd's method: build four candidate quaternions (one per diagonal pivot)
    and select the one whose pivot magnitude is largest. Robust at theta=pi where
    the single-formula approach is numerically unstable.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    q_abs = np.sqrt(np.maximum(np.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], axis=-1), 0.0))                                    # (..., 4)

    # Candidate quaternions (w, x, y, z) for each pivot, unnormalized.
    cand = np.stack([
        np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
        np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
        np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
        np.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], axis=-1),
    ], axis=-2)                                           # (..., 4, 4)
    cand = cand / (2.0 * np.maximum(q_abs[..., None], 0.1))

    best = np.argmax(q_abs, axis=-1)                      # (...)
    q = np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :]
    return normalize(q)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = normalize(np.asarray(q, dtype=np.float64))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


# --------------------------------------------------------------------------- #
# AI4Animation forward/up "Look" convention (upstream generator adapter)
# --------------------------------------------------------------------------- #
def look_to_matrix(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation.Look(z=forward, y=up) as used by ai4animationpy ReadRotation3D.

    Builds R whose columns are (right, up, forward) for a left-handed-ish look
    basis: z'=normalize(forward), x'=normalize(cross(up, z')), y'=cross(z', x').
    This is the ADAPTER for the upstream stored 6D if it turns out to be the
    AxisZ/AxisY pair rather than Gram-Schmidt r6d. Verify empirically before use.
    """
    z = normalize(forward)
    x = normalize(np.cross(np.asarray(up, dtype=np.float64), z))
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)        # columns = x, y, z


def matrix_to_look(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `look_to_matrix`: return (forward=AxisZ, up=AxisY) columns."""
    R = np.asarray(R, dtype=np.float64)
    forward = R[..., :, 2]
    up = R[..., :, 1]
    return forward, up


def is_rotation_matrix(R: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Boolean test that R is in SO(3) (orthonormal, det +1)."""
    R = np.asarray(R, dtype=np.float64)
    eye = np.broadcast_to(np.eye(3), R.shape)
    orth = np.allclose(np.swapaxes(R, -1, -2) @ R, eye, atol=atol)
    det = np.allclose(np.linalg.det(R), 1.0, atol=atol)
    return bool(orth and det)
