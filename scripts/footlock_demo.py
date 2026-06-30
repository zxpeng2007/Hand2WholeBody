"""Demo A — post-process foot-locking: generate a clip, foot-lock it, render before/after.

    python scripts/footlock_demo.py --cache data/cache/pairs_full.npz \
        --checkpoint checkpoints/diffusion_full.pt --out footlock.mp4

Prints the foot-skate metric (mm/s, lower = less sliding) before vs after, and renders a
side-by-side skeleton video [before foot-lock | after foot-lock].
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from h2b.data.cache import load_pairs_cache, clip_wrist_activity
from h2b.eval import split_clips
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
from h2b.models import fk_torch as FKt
from h2b import inference as INF
from h2b.export.footlock import footlock, foot_skate
from h2b.export.visualize import animate_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--out", default="footlock.mp4")
    ap.add_argument("--seq", type=int, default=-1)
    ap.add_argument("--max-frames", type=int, default=180)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    clips, rest = load_pairs_cache(args.cache)
    _, val = split_clips(clips, val_frac=0.1, seed=0)
    if args.seq >= 0:
        idx = args.seq
    else:
        acts = np.array([clip_wrist_activity(c) for c in val])
        lens = np.minimum([len(c[0]) for c in val], args.max_frames)
        idx = int(np.argmax(acts * lens))
    hand, _ = val[idx]
    hand = hand[:args.max_frames]

    model = DiTDenoiser(hidden=256, n_layers=4).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    diff = GaussianDiffusion(device=args.device)
    motion = INF.generate(model, hand, arch="diffusion", diffusion=diff, sample_steps=8, device=args.device)

    locked, contact = footlock(motion, hand=hand, rest_joints=rest, iters=args.iters, device=args.device)

    s0 = foot_skate(motion, rest_joints=rest)
    s1 = foot_skate(locked, rest_joints=rest)
    print(f"clip {idx}: {len(hand)} frames, contact frames {int(contact.sum())}")
    print(f"foot-skate  before = {1000*s0:.1f} mm/s   after = {1000*s1:.1f} mm/s   "
          f"({100*(1-s1/max(s0,1e-9)):.0f}% reduction)")

    rest_t = torch.as_tensor(rest, dtype=torch.float32)
    _, p0 = FKt.motion_to_joints(torch.tensor(motion)[None], rest_t)
    _, p1 = FKt.motion_to_joints(torch.tensor(locked)[None], rest_t)
    out = animate_comparison([p0[0].numpy(), p1[0].numpy()], args.out, fps=30,
                             titles=["before foot-lock", "after foot-lock"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
