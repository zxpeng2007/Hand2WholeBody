"""Differentiable (torch) rotation ops mirroring h2b.representations.rotations.

Hardcoded to the project's Zhou-2019 COLUMN convention (frames.PROJECT_R6D) — the model
and losses operate entirely in that convention, so there is no convention argument to get
wrong. Parity with the NumPy reference is unit-tested.

torch is imported lazily-safely so importing the package never requires torch.
"""

from __future__ import annotations

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _check():
    if not _HAS_TORCH:
        raise ImportError("rotations_torch requires torch")


def axis_angle_to_matrix(aa):
    """(..., 3) -> (..., 3, 3) via Rodrigues. Differentiable."""
    _check()
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    k = aa / theta.clamp_min(1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1)
    K = K.reshape(*k.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * K + (1.0 - c) * (K @ K)


def rotation_6d_to_matrix(d6):
    """(..., 6) COLUMN-packed 6D -> (..., 3, 3). Gram-Schmidt; columns = b1, b2, b3."""
    _check()
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)        # columns


def matrix_to_rotation_6d(R):
    """(..., 3, 3) -> (..., 6): first two COLUMNS flattened."""
    _check()
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
