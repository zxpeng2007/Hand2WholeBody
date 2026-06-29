"""Tests for the AMASS-style SMPL export (Stage-2 → Stage-3 handoff)."""

import numpy as np

from h2b.export import to_amass_npz as EX
from h2b.representations import frames as F
from h2b.representations import rotations as R


def test_motion6d_to_aa_shapes_and_roundtrip():
    rng = np.random.default_rng(0)
    T = 12
    root_R = R.axis_angle_to_matrix(rng.standard_normal((T, 3)) * 0.3)
    body_R = R.axis_angle_to_matrix(rng.standard_normal((T, 23, 3)) * 0.3)
    # encode with the project convention so it matches motion6d_to_aa's decode
    root6d = R.matrix_to_rotation_6d(root_R, convention=F.PROJECT_R6D)
    body6d = R.matrix_to_rotation_6d(body_R, convention=F.PROJECT_R6D)
    poses = EX.motion6d_to_aa(root6d, body6d)
    assert poses.shape == (T, 72)
    # re-encode and compare rotations (axis-angle is not unique, matrices are)
    assert np.allclose(R.axis_angle_to_matrix(poses[:, 0:3]), root_R, atol=1e-7)
    body_back = R.axis_angle_to_matrix(poses[:, 3:72].reshape(T, 23, 3))
    assert np.allclose(body_back, body_R, atol=1e-7)


def test_smplx_npz_has_gmr_keys_and_shapes(tmp_path):
    T = 10
    poses = np.zeros((T, 72)); poses[:, 0:3] = 0.1; poses[:, 3:66] = 0.2  # body filled, hands 0
    trans = np.tile([0.0, 0.0, 1.0], (T, 1))
    out = EX.smpl_motion_to_smplx_npz(str(tmp_path / "c.npz"), poses, np.asarray(trans),
                                      betas=np.zeros(10), fps=30, gender="neutral", height_m=1.8)
    d = np.load(out, allow_pickle=True)
    assert set(["root_orient", "pose_body", "betas", "trans", "gender", "mocap_frame_rate"]).issubset(d.files)
    assert "poses" not in d.files                          # SMPL-X format, no 'poses' key
    assert d["root_orient"].shape == (T, 3)
    assert d["pose_body"].shape == (T, 63)                  # 21 body joints, hands dropped
    assert d["betas"].shape == (16,)
    assert abs(float(d["betas"][0]) - (1.8 - 1.66) / 0.1) < 1e-4   # height -> betas[0] scale
    assert np.allclose(d["root_orient"], poses[:, 0:3])
    assert np.allclose(d["pose_body"], poses[:, 3:66])


def test_npz_written_with_contract_keys(tmp_path):
    T = 8
    poses = np.zeros((T, 72))
    trans = np.tile([0.0, 0.0, 1.0], (T, 1))
    betas = np.zeros(10)
    out = EX.smpl_motion_to_amass_npz(
        str(tmp_path / "clip.npz"), poses, trans, betas, fps=30, gender="neutral",
        contacts=np.zeros((T, 4)),
    )
    d = np.load(out, allow_pickle=True)
    assert set(["poses", "trans", "betas", "gender", "mocap_frame_rate", "contacts"]).issubset(d.files)
    assert d["poses"].shape == (T, 72)
    assert d["trans"].shape == (T, 3)
    assert int(d["mocap_frame_rate"]) == 30
    assert str(d["gender"]) == "neutral"
