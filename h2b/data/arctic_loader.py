"""Load ARCTIC (SMPL-X bimanual manipulation) -> Hand2Body (hand, body) clips.

ARCTIC (Fan et al., CVPR 2023) provides full-body SMPL-X + both MANO hands + articulated objects.
For the two-wrist v1 we use only the SMPL-X BODY: joints 0..21 are the SAME kinematic tree as SMPL
(left_wrist=20, right_wrist=21), so we FK-derive the 24D two-wrist signal and the 135-D body exactly
like `pkl_loader` does for train.pkl. Hands/fingers and objects are ignored (rigid-wrist body model);
they are available for a later object-conditioning / grasp phase.

VERIFIED against github.com/zc-alexfan/arctic (docs/data + common/body_models.py, 2026-07-01):
  * layout: `arctic_data/data/raw_seqs/s01..s10/<obj>_<action>_NN.smplx.npy` (np.load(...).item()).
    We glob `raw_seqs/**/*.smplx.npy` and read the subject id from the parent folder (s01..s10).
  * keys/shapes: `transl (T,3)`, `global_orient (T,3)`, `body_pose (T,63)` (+ jaw/eye/hand ignored).
    Axis-angle, WORLD frame, meters. fps = 30 (common/viewer.py) -> matches our pipeline, no resample.
  * per-subject SHAPE + GENDER (fidelity): `meta/misc.json[sid]["gender"]` and a personalized
    v-template `meta/subject_vtemplates/{sid}.obj` (10475x3), passed as SMPL-X `v_template` (this is
    exactly `common/body_models.py::construct_layers`). If `meta/` is absent we fall back to
    neutral/mean-shape -- still SELF-CONSISTENT (rest calibrated from the same FK), just not the
    subject's true anthropometry.
Body-joint POSITIONS come from a `joints_fn` (poses72, trans, betas) -> (T,24,3): the real SMPL-X
model by default, or an injected stub for tests.
"""

from __future__ import annotations

import glob
import json
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
    bp = bp.reshape(bp.shape[0], -1)[:, :63]                     # 21 body joints x 3 (drop hand/jaw)
    tr = np.asarray(_first(d, "transl", "trans"), np.float64).reshape(-1, 3)
    T = min(len(go), len(bp), len(tr))
    return {"global_orient": go[:T], "body_pose": bp[:T], "transl": tr[:T],
            "betas": np.asarray(d.get("betas", np.zeros(10)), np.float64).reshape(-1),
            "gender": str(d.get("gender", "neutral"))}


def _poses72(sx: dict) -> np.ndarray:
    """SMPL 22-joint axis-angle padded to 72 (hand joints 22/23 = 0), from SMPL-X body params."""
    poses66 = np.concatenate([sx["global_orient"], sx["body_pose"]], axis=-1)   # (T,66) == SMPL 22-joint
    return np.concatenate([poses66, np.zeros((len(poses66), 6))], axis=-1)


def subject_id_from_path(fp: str) -> str:
    """.../raw_seqs/s01/box_grab_01.smplx.npy -> 's01'."""
    return os.path.basename(os.path.dirname(fp))


def load_arctic_meta(meta_dir: str):
    """meta/ -> {genders: {sid: 'male'|'female'|'neutral'}, vtemplate_dir}, or None if absent."""
    misc_p = os.path.join(meta_dir, "misc.json")
    if not os.path.exists(misc_p):
        return None
    misc = json.load(open(misc_p))
    return {"genders": {sid: misc[sid].get("gender", "neutral") for sid in misc},
            "vtemplate_dir": os.path.join(meta_dir, "subject_vtemplates")}


def _load_vtemplate(vtemplate_dir: str, sid: str):
    """meta/subject_vtemplates/{sid}.obj -> (10475,3) SMPL-X v_template, or None."""
    p = os.path.join(vtemplate_dir, f"{sid}.obj")
    if not os.path.exists(p):
        return None
    import trimesh
    return np.asarray(trimesh.load(p, process=False).vertices, np.float64)


def smplx_joints_fn_factory(model_dir: str, gender: str = "neutral", v_template=None, num_betas: int = 10):
    """joints_fn (poses72, trans, betas) -> (T,24,3) world via the real SMPL-X model. `gender` and the
    personalized `v_template` (10475x3, overrides betas) are baked in per subject (as ARCTIC does).
    `model_dir` contains `smplx/SMPLX_<GENDER>.{npz,pkl}` (e.g. repo assets/models)."""
    def _fn(poses, trans, betas):
        import torch
        import smplx
        T = poses.shape[0]
        kw = dict(model_type="smplx", gender=gender, num_betas=num_betas, use_pca=False, batch_size=T)
        if v_template is not None:
            kw["v_template"] = torch.tensor(np.asarray(v_template), dtype=torch.float32)
        model = smplx.create(model_dir, **kw)
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


def planted_feet_fraction(body: np.ndarray, rest_joints=None, fps: float = 30.0,
                          h_offset: float = 0.06, v_thresh: float = 0.15) -> float:
    """Fraction of frames where BOTH feet are near the ground AND slow — a proxy for seated /
    planted-feet manipulation (vs walking). Used to filter ARCTIC to the low-locomotion regime
    where the unconstrained lower body is least harmful."""
    import torch
    from ..models import fk_torch as FKt
    rest = (FKt.default_rest_joints() if rest_joints is None
            else torch.as_tensor(rest_joints, dtype=torch.float32))
    with torch.no_grad():
        _, pos = FKt.motion_to_joints(torch.as_tensor(np.asarray(body, np.float32))[None], rest)
    fp = pos[0, :, list(F.FOOT_JOINTS)].cpu().numpy()           # (T, 4, 3)
    T = fp.shape[0]
    if T < 2:
        return 1.0
    height = fp[..., 2]
    speed = np.zeros((T, fp.shape[1]))
    speed[1:] = np.linalg.norm((fp[1:] - fp[:-1]) * fps, axis=-1)
    planted = (height < height.min(0, keepdims=True) + h_offset) & (speed < v_thresh)   # (T,4)
    return float(planted.all(axis=1).mean())                   # frames with all feet planted


def load_arctic_clips(arctic_dir: str, model_dir: str = "", meta_dir: str = "", gender: str = "neutral",
                      fps: float = 30.0, wrists=F.WRIST_JOINTS, limit: int | None = None,
                      min_frames: int = 8, calibrate: bool = True, calib_seqs: int = 16,
                      seated_min: float = 0.0, joints_fn=None):
    """Glob `<arctic_dir>/**/*.smplx.npy` -> (clips, rest_joints).

    `model_dir`: SMPL-X model dir (real FK); empty -> approx skeleton stub. `meta_dir`: ARCTIC meta/
    for per-subject gender + v-template (fidelity); empty -> `gender`/mean-shape for all. `seated_min`:
    drop sequences whose planted-feet fraction < this (0 = keep all). `joints_fn`: override for tests."""
    files = sorted(glob.glob(os.path.join(arctic_dir, "**", "*.smplx.npy"), recursive=True))
    meta = load_arctic_meta(meta_dir) if meta_dir else None
    fn_cache: dict = {}

    def _joints_fn_for(sid: str):
        if joints_fn is not None:
            return joints_fn
        if sid not in fn_cache:
            g = meta["genders"].get(sid, gender) if meta else gender
            vt = _load_vtemplate(meta["vtemplate_dir"], sid) if meta else None
            fn_cache[sid] = (smplx_joints_fn_factory(model_dir, g, vt) if model_dir
                             else SF.synthetic_joints_fn)
        return fn_cache[sid]

    clips, rests, n_seat = [], [], 0
    for fp in files:
        sid = subject_id_from_path(fp)
        sx = normalize_arctic_smplx(np.load(fp, allow_pickle=True).item())
        if len(sx["global_orient"]) < min_frames:
            continue
        try:
            hand, body, joints = arctic_seq_to_pair(sx, _joints_fn_for(sid), fps=fps, wrists=wrists)
        except Exception as e:                                  # skip a bad sequence, keep going
            print(f"skip {os.path.basename(fp)}: {e}")
            continue
        if calibrate and len(rests) < calib_seqs:
            rests.append(PK.calibrate_rest_joints(_poses72(sx), joints, sx["transl"]))
        if seated_min > 0:
            rest_now = np.mean(rests, axis=0) if rests else None
            if planted_feet_fraction(body, rest_now) < seated_min:
                n_seat += 1
                continue
        clips.append((hand, body))
        if limit and len(clips) >= limit:
            break
    if seated_min > 0:
        print(f"seated filter: dropped {n_seat} sequences (planted-feet fraction < {seated_min})")
    return clips, (np.mean(rests, axis=0) if rests else None)
