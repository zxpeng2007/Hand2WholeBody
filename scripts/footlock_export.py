"""Generate a clip, foot-lock it, and export before/after as AMASS-style npz for aitviewer.

    python scripts/footlock_export.py --seconds 30 --device cuda \
        --out-before before.npz --out-after after.npz

Then render each with scripts/render_aitviewer.py --input ... and hstack the two mp4s.
Prints the foot-skate metric (mm/s) before vs after.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from h2b.data.cache import load_pairs_cache, clip_wrist_activity
from h2b.eval import split_clips
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
from h2b import inference as INF
from h2b.export.footlock import footlock, foot_skate
from h2b.representations import body as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-before", required=True)
    ap.add_argument("--out-after", required=True)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    target = int(args.seconds * 30)
    clips, rest = load_pairs_cache(args.cache)
    _, val = split_clips(clips, val_frac=0.1, seed=0)
    lens = np.array([len(c[0]) for c in val])
    acts = np.array([clip_wrist_activity(c) for c in val])
    elig = np.where(lens >= target)[0]
    idx = int(elig[np.argmax(acts[elig])]) if len(elig) else int(np.argmax(lens))
    hand = val[idx][0][:target]

    model = DiTDenoiser(hidden=256, n_layers=4).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    diff = GaussianDiffusion(device=dev)
    motion = INF.generate_long(model, hand, arch="diffusion", diffusion=diff, sample_steps=8, device=dev)
    locked, contact = footlock(motion, hand=hand, rest_joints=rest, iters=args.iters, device=dev)

    s0, s1 = foot_skate(motion, rest_joints=rest), foot_skate(locked, rest_joints=rest)
    print(f"clip {idx}: {len(hand)} frames, contact frames {int(contact.sum())}")
    print(f"foot-skate  before = {1000*s0:.1f} mm/s   after = {1000*s1:.1f} mm/s   "
          f"({100*(1-s1/max(s0,1e-9)):.0f}% reduction)")

    for path, m in ((args.out_before, motion), (args.out_after, locked)):
        poses, trans = B.motion_to_smpl72(m)
        np.savez(path, poses=np.asarray(poses, np.float32), trans=np.asarray(trans, np.float32))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
