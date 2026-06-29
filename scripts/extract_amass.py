"""M1: extract paired (12D left-wrist signal, whole-body SMPL) training data from AMASS.

Usage (once smplx + the SMPL model + an AMASS dir are available):
    python scripts/extract_amass.py --amass DIR --smpl-model DIR --out data/pairs --fps 30

Without the SMPL model you can still smoke-test the pipeline on synthetic joints:
    python scripts/extract_amass.py --amass DIR --synthetic --out data/pairs

Each input AMASS .npz (keys: poses (T,156) SMPL-H, trans, betas, mocap_framerate) is
resampled to --fps, the left-wrist 12D is derived by FK (h2b.data.smpl_fk), and the
(hand12, body) pair is saved. `body` packs root_trans + root_orient_6d + 22 body-joint
rot6D to match the model output layout (see configs/default.yaml).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from h2b.data import smpl_fk as FK
from h2b.representations import body as BODY
from h2b.representations import frames as F
from h2b.representations import rotations as R


def resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear resample along time axis 0 (causal-safe nearest for rotations is TODO)."""
    if abs(src_fps - dst_fps) < 1e-6:
        return arr
    T = arr.shape[0]
    dst_T = max(1, int(round(T * dst_fps / src_fps)))
    src_t = np.linspace(0, 1, T)
    dst_t = np.linspace(0, 1, dst_T)
    out = np.stack([np.interp(dst_t, src_t, arr[:, i]) for i in range(arr.shape[1])], axis=1)
    return out


def smpl_body_vector(poses72: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Pack the 135-D body target via the shared representation (h2b.representations.body)."""
    return BODY.smpl72_to_motion(poses72, trans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", required=True, help="dir of AMASS .npz files")
    ap.add_argument("--smpl-model", default="", help="SMPL model dir (smplx). Omit with --synthetic")
    ap.add_argument("--synthetic", action="store_true", help="use the model-free joints stub")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        joints_fn = FK.synthetic_joints_fn
    else:
        joints_fn = FK.smplx_joints_fn_factory(args.smpl_model)

    files = sorted(glob.glob(os.path.join(args.amass, "**", "*.npz"), recursive=True))
    print(f"found {len(files)} AMASS files")
    for i, fp in enumerate(files):
        d = np.load(fp)
        src_fps = float(d["mocap_framerate"]) if "mocap_framerate" in d else 60.0
        poses = np.asarray(d["poses"])[:, :72]          # SMPL-H -> SMPL body (first 24 joints)
        trans = np.asarray(d["trans"])
        betas = np.asarray(d["betas"])[:10]
        poses = resample(poses, src_fps, args.fps)
        trans = resample(trans, src_fps, args.fps)
        hand12 = FK.extract_hand12(poses, trans, betas, joints_fn, fps=args.fps)
        body = smpl_body_vector(poses, trans)
        out = os.path.join(args.out, f"pair_{i:05d}.npz")
        np.savez(out, hand12=hand12.astype(np.float32), body=body.astype(np.float32),
                 betas=betas.astype(np.float32), fps=args.fps)
    print(f"wrote pairs to {args.out}")


if __name__ == "__main__":
    main()
