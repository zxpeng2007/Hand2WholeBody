"""Render a video of generated whole-body motion (generated vs ground truth) from a checkpoint.

    python scripts/render_video.py --cache data/cache/pairs_full.npz \
        --checkpoint checkpoints/diffusion_full.pt --out demo.mp4

Picks (by default) the most wrist-active held-out clip — an actual swing — generates the body
from its hand 12D, and renders a side-by-side 3D skeleton animation (accurate calibrated skeleton).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from h2b.data.cache import load_pairs_cache, clip_wrist_activity
from h2b.eval import split_clips
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
from h2b.models.regressor import RegressorHand2Body
from h2b.models import fk_torch as FKt
from h2b import inference as INF
from h2b.export.visualize import animate_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--arch", default="diffusion", choices=["diffusion", "regressor"])
    ap.add_argument("--out", default="h2b_result.mp4")
    ap.add_argument("--seq", type=int, default=-1, help="val clip index; -1 = most active")
    ap.add_argument("--max-frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.max_frames = min(args.max_frames, 256)            # model positional budget (single-shot)
    device = args.device if torch.cuda.is_available() else "cpu"
    clips, rest = load_pairs_cache(args.cache)
    _, val = split_clips(clips, val_frac=0.1, seed=0)
    if args.seq >= 0:
        idx = args.seq
    else:                                                  # favor clips that are long AND active
        acts = np.array([clip_wrist_activity(c) for c in val])
        lens = np.minimum([len(c[0]) for c in val], args.max_frames)
        idx = int(np.argmax(acts * lens))
    hand, gt_body = val[idx]
    hand, gt_body = hand[:args.max_frames], gt_body[:args.max_frames]
    print(f"val clip {idx}: {hand.shape[0]} frames, wrist activity {clip_wrist_activity((hand, gt_body)):.2f} m/s")

    if args.arch == "diffusion":
        model = DiTDenoiser(hidden=256, n_layers=4).to(device)
        diff = GaussianDiffusion(device=device)
    else:
        model = RegressorHand2Body(hidden=256, n_layers=4).to(device)
        diff = None
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    motion = INF.generate(model, hand, arch=args.arch, diffusion=diff, sample_steps=8, device=device)
    rest_t = torch.as_tensor(rest, dtype=torch.float32)
    _, gen_pos = FKt.motion_to_joints(torch.tensor(motion)[None], rest_t)
    _, gt_pos = FKt.motion_to_joints(torch.tensor(gt_body)[None], rest_t)

    out = animate_comparison([gen_pos[0].numpy(), gt_pos[0].numpy()], args.out, fps=args.fps,
                             titles=["GENERATED (from hand only)", "GROUND TRUTH"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
