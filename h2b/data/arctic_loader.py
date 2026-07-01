"""Load ARCTIC (SMPL-X bimanual manipulation) -> Hand2Body (hand, body) clips.

ARCTIC (Fan et al., CVPR 2023) provides full-body SMPL-X + both MANO hands + articulated objects.
For the two-wrist v1 we use only the SMPL-X BODY: joints 0..21 are the SAME kinematic tree as SMPL
(left_wrist=20, right_wrist=21), so we FK-derive the 24D two-wrist signal and the 135-D body exactly
like `pkl_loader` does for train.pkl. Hands/fingers and objects are ignored (rigid-wrist body model);
they are available for a later object-conditioning / grasp phase.

ARCTIC per-sequence SMPL-X params live in `arctic_data/data/raw_seqs/<sid>/<obj>_<action>.smplx.npy`
(np.load(..., allow_pickle=True).item() -> dict of (T,·) arrays). Key names vary across releases, so
we normalize {global_orient|root_orient, body_pose|pose_body, transl|trans, betas}. Body-joint
POSITIONS (for the wrist signal + rest calibration) come from a `joints_fn` — the real SMPL-X model
by default (accurate), or an injected stub for tests.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ..representations import frames as F
from . import smpl_fk as SF
from . import pkl_loader as PK


def _first(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} present; have {list(d.keys())}")


def normalize_arctic_smplx(d: dict) -> dict:
    """ARCTIC dict-of-arrays -> {global_orient (T,3), body_pose (T,63), transl (T,3), betas, gender}."""
    go = np.asarray(_first(d, "global_orient", "root_orient"), np.float64).reshape(-1, 3)
    bp = np.asarray(_first(d, "body_pose", "pose_body"), np.float64)
    bp = bp.reshape(bp.shape[0], -1)[:, :63]                     # 21 body joints x 3 (drop any hand/jaw)
    tr = np.asarray(_first(d, "transl", "trans"), np.float64).reshape(-1, 3)
    T = min(len(go), len(bp), len(tr))
    return {"global_orient": go[:T], "body_pose": bp[:T], "transl": tr[:T],
            "betas": np.asarray(d.get("betas", np.zeros(10)), np.float64).reshape(-1),
            "gender": str(d.get("gender", "neutral"))}


def _poses72(sx: dict) -> np.ndarray:
    """SMPL 22-joint axis-angle padded to 72 (hand joints 22/23 = 0), from SMPL-X body params."""
    poses66 = np.concatenate([sx["global_orient"], sx["body_pose"]], axis=-1)   # (T,66) == SMPL 22-joint
    return np.concatenate([poses66, np.zeros((len(poses66), 6))], axis=-1)


def smplx_joints_fn_factory(model_dir: str, gender: str = "neutral", num_betas: int = 10):
    """joints_fn (poses72, trans, betas) -> (T,24,3) world, via the real SMPL-X model (lazy smplx+torch).
    `model_dir` is the folder CONTAINING `smplx/SMPLX_<GENDER>.{npz,pkl}` (e.g. repo assets/models)."""
    def _fn(poses, trans, betas):
        import torch
        import smplx
        T = poses.shape[0]
        model = smplx.create(model_dir, model_type="smplx", gender=gender,
                             num_betas=num_betas, use_pca=False, batch_size=T)
        b = np.broadcast_to(np.asarray(betas).reshape(-1)[:num_betas], (T, num_betas)).copy()
        out = model(global_orient=torch.tensor(poses[:, 0:3], dtype=torch.float32),
                    body_pose=torch.tensor(poses[:, 3:66], dtype=torch.float32),
                    transl=torch.tensor(trans, dtype=torch.float32),
                    betas=torch.tensor(b, dtype=torch.float32))
        return out.joints[:, :24, :].detach().cpu().numpy()     # 0..21 body (wrists 20/21) + jaw/eyes
    return _fn


def arctic_seq_to_pair(sx: dict, joints_fn, fps: float = 30.0, wrists=F.WRIST_JOINTS):
    """Normalized SMPL-X dict -> (hand (T,12*len(wrists)), body (T,135), joints (T,24,3))."""
    poses72 = _poses72(sx)
    joints = np.asarray(joints_fn(poses72, sx["transl"], sx["betas"]), np.float64)   # (T,24,3) world
    seq = {"poses": poses72, "trans": sx["transl"], "betas": sx["betas"],
           "gender": sx["gender"], "joints": joints}
    hand, body = PK.sequence_to_pair(seq, fps=fps, wrists=wrists)
    return hand, body, joints


def load_arctic_clips(arctic_dir: str, model_dir: str = "", gender: str = "neutral", fps: float = 30.0,
                      wrists=F.WRIST_JOINTS, limit: int | None = None, min_frames: int = 8,
                      calibrate: bool = True, calib_seqs: int = 16, joints_fn=None):
    """Glob `<arctic_dir>/**/*.smplx.npy` -> (clips, rest_joints). Default `joints_fn` = the real
    SMPL-X model at `model_dir` (falls back to the approx skeleton if `model_dir` is empty).
    `wrists=(20,21)` -> 24D bimanual; `(20,)` -> 12D left only (respects --wrist-count)."""
    files = sorted(glob.glob(os.path.join(arctic_dir, "**", "*.smplx.npy"), recursive=True))
    if joints_fn is None:
        joints_fn = smplx_joints_fn_factory(model_dir, gender) if model_dir else SF.synthetic_joints_fn
    clips, rests = [], []
    for fp in files:
        sx = normalize_arctic_smplx(np.load(fp, allow_pickle=True).item())
        if len(sx["global_orient"]) < min_frames:
            continue
        try:
            hand, body, joints = arctic_seq_to_pair(sx, joints_fn, fps=fps, wrists=wrists)
        except Exception as e:                                  # skip a bad sequence, keep going
            print(f"skip {os.path.basename(fp)}: {e}")
            continue
        clips.append((hand, body))
        if calibrate and len(rests) < calib_seqs:
            rests.append(PK.calibrate_rest_joints(_poses72(sx), joints, sx["transl"]))
        if limit and len(clips) >= limit:
            break
    return clips, (np.mean(rests, axis=0) if rests else None)
