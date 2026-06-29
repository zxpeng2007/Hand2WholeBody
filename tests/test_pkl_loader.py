"""Tests for the train.pkl loader using a synthetic joblib pkl in the REAL nested format
(list of {motion: {trans, poses(T,66), joints(T,22,3), betas, gender}})."""

import numpy as np
import pytest

joblib = pytest.importorskip("joblib")

from h2b.data import pkl_loader as PL
from h2b.data import smpl_fk as FK
from h2b.representations import body as B
from h2b.representations import frames as F


def _make_pkl(tmp_path, n_seq=3, seed=0):
    """Self-consistent: joints are FK(poses, trans) so calibration can recover the skeleton."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_seq):
        T = 20 + i * 5
        poses = np.cumsum(rng.standard_normal((T, 66)) * 0.05, axis=0).astype(np.float32)
        trans = (np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + [0, 0, 0.9]).astype(np.float32)
        poses72 = PL._poses_to_72(poses.astype(np.float64))
        joints = FK.synthetic_joints_fn(poses72, trans, np.zeros(10))[:, :22].astype(np.float32)
        items.append({"motion": {"trans": trans, "poses": poses, "joints": joints,
                                 "betas": np.zeros(10, np.float32), "gender": "male",
                                 "pelvis_delta": np.zeros(3, np.float32)},
                      "seq_name": f"seq_{i}"})
    p = tmp_path / "train.pkl"
    joblib.dump(items, str(p))
    return str(p)


def test_iter_sequences_real_nested_format(tmp_path):
    seqs = list(PL.iter_sequences(PL.load_smpl_pkl(_make_pkl(tmp_path, 3))))
    assert len(seqs) == 3
    assert seqs[0]["poses"].shape[1] == 66
    assert seqs[0]["joints"].shape[1:] == (22, 3)
    assert seqs[0]["gender"] == "male"


def test_hand_position_uses_real_joints(tmp_path):
    seqs = list(PL.iter_sequences(PL.load_smpl_pkl(_make_pkl(tmp_path, 1))))
    hand, body = PL.sequence_to_pair(seqs[0])
    T = hand.shape[0]
    assert hand.shape == (T, F.HAND12_DIM) and body.shape == (T, B.MOTION_DIM)
    # position channel == the real left-wrist joint (index 20)
    assert np.allclose(hand[:, F.HAND12_POS], seqs[0]["joints"][:, F.LEFT_WRIST, :], atol=1e-4)
    # 6D channel decodes to the global wrist rotation
    Rg = FK.global_joint_rotations(PL._poses_to_72(seqs[0]["poses"]))[:, F.PADDLE_HAND_JOINT]
    assert np.allclose(F.matrix_from_hand12_rot(hand), Rg, atol=1e-4)


def test_calibrated_rest_reproduces_joints(tmp_path):
    torch = pytest.importorskip("torch")
    from h2b.models import fk_torch as FKt
    seqs = list(PL.iter_sequences(PL.load_smpl_pkl(_make_pkl(tmp_path, 1))))
    s = seqs[0]
    poses72 = PL._poses_to_72(s["poses"])
    rest = PL.calibrate_rest_joints(poses72, s["joints"], s["trans"])     # (22,3)
    body = B.smpl72_to_motion(poses72, s["trans"])
    _, pos = FKt.motion_to_joints(torch.tensor(body)[None],
                                  rest_joints=torch.tensor(rest, dtype=torch.float32))
    # FK with the calibrated skeleton reproduces the real joint positions
    assert np.allclose(pos[0].numpy(), s["joints"], atol=1e-4)


def test_load_clips_returns_clips_and_rest(tmp_path):
    clips, rest = PL.load_clips(_make_pkl(tmp_path, 3), min_frames=8)
    assert len(clips) == 3
    assert rest is not None and rest.shape == (22, 3)


def test_limit_caps_sequences(tmp_path):
    clips, _ = PL.load_clips(_make_pkl(tmp_path, 5), limit=2, min_frames=8)
    assert len(clips) == 2


def test_legacy_dict_of_lists_still_works(tmp_path):
    T = 16
    payload = {"poses": [np.zeros((T, 66), np.float32)],
               "trans": [np.tile([0, 0, 0.9], (T, 1)).astype(np.float32)]}
    p = tmp_path / "legacy.pkl"
    joblib.dump(payload, str(p))
    clips = PL.pkl_to_clips(str(p), min_frames=8)
    assert clips[0][1].shape == (T, B.MOTION_DIM)
