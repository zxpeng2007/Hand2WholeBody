"""ARCTIC (SMPL-X bimanual) ingestion -> (hand24, body135) clips. Validated on synthetic ARCTIC
data with a stub joints_fn (the real SMPL-X FK path is exercised only when the dataset is present)."""

import numpy as np
import pytest

from h2b.representations import frames as F

torch = pytest.importorskip("torch")

from h2b.data import arctic_loader as AL
from h2b.data import smpl_fk as SF
from h2b.data.pkl_loader import calibrate_rest_joints
from h2b import losses as L


def _synthetic_arctic(T=20, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "global_orient": (rng.standard_normal((T, 3)) * 0.1).astype(np.float32),
        "body_pose": (rng.standard_normal((T, 63)) * 0.1).astype(np.float32),
        "transl": np.cumsum(np.ones((T, 3)) * 0.01, 0).astype(np.float32),
        "betas": np.zeros(10, np.float32),
        "gender": "neutral",
    }


def test_normalize_key_variants():
    d = _synthetic_arctic(8)
    alt = {"root_orient": d["global_orient"],
           "pose_body": np.concatenate([d["body_pose"], np.zeros((8, 3))], 1),   # extra jaw col dropped
           "trans": d["transl"], "betas": d["betas"]}
    sx = AL.normalize_arctic_smplx(alt)
    assert sx["global_orient"].shape == (8, 3)
    assert sx["body_pose"].shape == (8, 63) and sx["transl"].shape == (8, 3)


def test_arctic_seq_to_pair_bimanual_cycle():
    sx = AL.normalize_arctic_smplx(_synthetic_arctic(24, 1))
    hand, body, joints = AL.arctic_seq_to_pair(sx, SF.synthetic_joints_fn, wrists=F.WRIST_JOINTS)
    assert hand.shape == (24, 24) and body.shape == (24, 135) and joints.shape[1] == 24
    # with the rest skeleton calibrated from (poses, joints, trans), FK(body)'s wrists reproduce
    # the 24D input -> the extraction is self-consistent (the real training invariant).
    rest = calibrate_rest_joints(AL._poses72(sx), joints, sx["transl"])
    bt, ht = torch.tensor(body)[None], torch.tensor(hand)[None]
    _, parts = L.compute_losses(bt, bt, ht, {"hand_consistency": 1.0},
                                rest_joints=torch.tensor(rest, dtype=torch.float32))
    assert parts["hand_consistency"] < 1e-4


def test_arctic_left_only_is_12d():
    sx = AL.normalize_arctic_smplx(_synthetic_arctic(10, 2))
    hand, body, _ = AL.arctic_seq_to_pair(sx, SF.synthetic_joints_fn, wrists=(F.LEFT_WRIST,))
    assert hand.shape == (10, 12) and body.shape == (10, 135)


def test_load_arctic_clips_from_files(tmp_path):
    for i in range(2):
        np.save(str(tmp_path / f"s{i}_use.smplx"), _synthetic_arctic(30, i), allow_pickle=True)
    clips, rest = AL.load_arctic_clips(str(tmp_path), joints_fn=SF.synthetic_joints_fn,
                                       wrists=F.WRIST_JOINTS)
    assert len(clips) == 2
    assert clips[0][0].shape[1] == 24 and clips[0][1].shape[1] == 135
    assert rest is not None and rest.shape == (22, 3)
