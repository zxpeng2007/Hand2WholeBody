"""Loader for the upstream whole-body SMPL data (train.pkl) -> training pairs.

REAL format (verified on train.pkl, 2026-06-29): a JOBLIB pickle of a **list of sequence
dicts**. Each item: {motion, data_source, seq_name, feat_p, frame_labels}. The `motion` dict:
  * trans  : [T, 3] root translation (world, z-up = our table frame).
  * poses  : [T, 66] axis-angle = root(3) + 21 body joints(63)  (smpl_22, hands dropped).
  * joints : [T, 22, 3] REAL world joint positions (absolute, incl. translation).
  * betas  : [10], gender : str, pelvis_delta : [3].
  * frame_labels : per-frame {start_t,end_t,proc_label('ball_cond_...'),act_cat} (ball-conditioned).

We use the REAL `joints[:, LEFT_WRIST]` for the 12D position, FK-on-poses for the global
wrist orientation, and `smpl_22_to_motion` for the 135-D body. We also calibrate the FK rest
skeleton from (poses, joints, trans) so the differentiable FK in the loss reproduces the data
(no approximate skeleton, no SMPL model files needed).

A legacy dict-of-lists format (the separate bvh_data_split_smpl_22.pkl) is still handled.
"""

from __future__ import annotations

import sys

import numpy as np

from . import smpl_fk as FK
from ..representations import body as B
from ..representations import frames as F


def load_smpl_pkl(path: str):
    """joblib.load with the numpy._core compat shim newer pickles need."""
    import joblib
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") != "numpy._core":
            raise
        import numpy.core as np_core
        sys.modules.setdefault("numpy._core", np_core)
        if hasattr(np_core, "multiarray"):
            sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        return joblib.load(path)


def _to_seq_list(value, key):
    if isinstance(value, list):
        return [np.asarray(v) for v in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [np.asarray(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [value]
    raise ValueError(f"`{key}` must be a list/obj-array/ndarray, got {type(value)}")


def _motion_of(item):
    """Return the per-sequence motion dict (unwrap the 'motion' key if present)."""
    if isinstance(item, dict) and isinstance(item.get("motion"), dict):
        return item["motion"]
    return item


def iter_sequences(payload):
    """Yield {idx, poses (T,66+), trans (T,3), joints (T,22,3)|None, betas, gender, seq_name}."""
    if isinstance(payload, list):                                  # REAL train.pkl format
        for i, item in enumerate(payload):
            m = _motion_of(item)
            if not isinstance(m, dict) or "poses" not in m or "trans" not in m:
                continue
            j = m.get("joints")
            yield {
                "idx": i,
                "poses": np.asarray(m["poses"], np.float64),
                "trans": np.asarray(m["trans"], np.float64),
                "joints": None if j is None else np.asarray(j, np.float64),
                "betas": np.asarray(m.get("betas", np.zeros(10)), np.float64),
                "gender": m.get("gender", "neutral"),
                "seq_name": item.get("seq_name") if isinstance(item, dict) else None,
            }
    elif isinstance(payload, dict) and "poses" in payload:         # legacy dict-of-lists
        trans_list = _to_seq_list(payload["trans"], "trans")
        poses_list = _to_seq_list(payload["poses"], "poses")
        for i, (tr, po) in enumerate(zip(trans_list, poses_list)):
            yield {"idx": i, "poses": np.asarray(po, np.float64),
                   "trans": np.asarray(tr, np.float64), "joints": None,
                   "betas": np.zeros(10), "gender": "neutral", "seq_name": None}
    else:
        raise ValueError(f"unrecognized payload type {type(payload)}")


def _poses_to_72(poses: np.ndarray) -> np.ndarray:
    """Pad/truncate axis-angle poses to the 72-dim (24-joint) layout for the FK helpers."""
    T, D = poses.shape
    if D >= 72:
        return poses[:, :72]
    out = np.zeros((T, 72), np.float64)
    out[:, :D] = poses
    return out


def sequence_to_pair(seq: dict, fps: float = 30.0):
    """One sequence -> (hand12 (T,12), body (T,135)). Uses REAL joints for the wrist position."""
    poses72 = _poses_to_72(seq["poses"])
    trans = seq["trans"]
    if seq.get("joints") is not None:
        pos = np.asarray(seq["joints"])[:, F.LEFT_WRIST, :]        # real world wrist position
    else:
        pos = FK.synthetic_joints_fn(poses72, trans, seq["betas"])[:, F.LEFT_WRIST, :]
    vel = F.finite_diff_velocity(pos, fps)
    rot6d = FK.wrist_orientation_6d(poses72)                       # global, column convention
    hand = F.pack_hand12(pos, vel, rot6d)
    body = B.smpl72_to_motion(poses72, trans)
    return hand.astype(np.float32), body.astype(np.float32)


def calibrate_rest_joints(poses72: np.ndarray, joints: np.ndarray, trans: np.ndarray,
                          parents=B.BODY_PARENTS) -> np.ndarray:
    """Recover the 22 rest-joint offsets from real (poses, joints, trans) so FK reproduces joints.

    Bones are rigid in SMPL's joint computation, so per joint:
        joints[:,j] - joints[:,parent] = R_global[parent] @ (rest[j] - rest[parent]).
    Solve rest[j]-rest[parent] = mean_t R_global[parent]^T (joints[:,j]-joints[:,parent]); set
    rest[0] = mean_t (joints[:,0] - trans) (the pelvis offset). Averaging over frames removes noise.
    """
    J = len(parents)
    joints = np.asarray(joints, np.float64)[:, :J]
    Rg = FK.global_joint_rotations(poses72)[:, :J]                 # (T,J,3,3)
    rest = np.zeros((J, 3))
    rest[0] = (joints[:, 0] - np.asarray(trans, np.float64)).mean(0)
    for j in range(1, J):
        p = parents[j]
        dj = joints[:, j] - joints[:, p]                          # (T,3)
        off = np.einsum("tij,tj->ti", np.swapaxes(Rg[:, p], -1, -2), dj).mean(0)
        rest[j] = rest[p] + off
    return rest


def load_clips(path: str, fps: float = 30.0, limit: int | None = None, min_frames: int = 8,
               calibrate: bool = True, calib_seqs: int = 16):
    """Load train.pkl -> (clips, rest_joints). clips = list of (hand12, body); rest_joints (22,3)
    calibrated from the real joints (or None if unavailable)."""
    payload = load_smpl_pkl(path)
    clips, rests = [], []
    for seq in iter_sequences(payload):
        if seq["poses"].shape[0] < min_frames:
            continue
        clips.append(sequence_to_pair(seq, fps=fps))
        if calibrate and seq.get("joints") is not None and len(rests) < calib_seqs:
            rests.append(calibrate_rest_joints(_poses_to_72(seq["poses"]),
                                               seq["joints"], seq["trans"]))
        if limit is not None and len(clips) >= limit:
            break
    rest = np.mean(rests, axis=0) if rests else None
    return clips, rest


def pkl_to_clips(path: str, fps: float = 30.0, limit: int | None = None, min_frames: int = 8):
    """Convenience: just the clips (no rest calibration)."""
    return load_clips(path, fps=fps, limit=limit, min_frames=min_frames, calibrate=False)[0]
