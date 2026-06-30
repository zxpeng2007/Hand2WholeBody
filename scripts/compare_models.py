"""Evaluate trained checkpoints on the SAME held-out val set (apples-to-apples).

    python scripts/compare_models.py --cache data/cache/pairs_full.npz

Loads the full cache, reproduces the deterministic val split (seed 0), and reports held-out
metrics for each checkpoint on that identical val set — so the activity-filtered model (whose
own training split differs) is compared fairly against the full-data models.
"""

from __future__ import annotations

import argparse
import os

import torch

from h2b.data.cache import load_pairs_cache
from h2b.eval import split_clips, evaluate
from h2b.models.regressor import RegressorHand2Body
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--max-windows", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    clips, rest = load_pairs_cache(args.cache)
    _, val = split_clips(clips, val_frac=0.1, seed=0)        # the SAME full val set
    print(f"evaluating on {len(val)} held-out sequences (identical for all models)\n")

    specs = [
        ("diffusion_full", "checkpoints/diffusion_full.pt", "diffusion"),
        ("diffusion_footloss", "checkpoints/diffusion_footloss.pt", "diffusion"),
        ("regressor_full", "checkpoints/regressor_full.pt", "regressor"),
        ("diffusion_active", "checkpoints/diffusion_active.pt", "diffusion"),
    ]
    rows = []
    for name, ckpt, arch in specs:
        if not os.path.exists(ckpt):
            print(f"skip {name}: {ckpt} missing"); continue
        if arch == "diffusion":
            model = DiTDenoiser(hidden=256, n_layers=4).to(device)
            diff = GaussianDiffusion(device=device)
        else:
            model = RegressorHand2Body(hidden=256, n_layers=4).to(device)
            diff = None
        model.load_state_dict(torch.load(ckpt, map_location=device))
        m = evaluate(model, val, length=args.length, device=device, arch=arch,
                     diffusion=diff, rest_joints=rest, sample_steps=8, max_windows=args.max_windows)
        rows.append((name, m))
        print(f"{name:20s}  mpjpe={m['val_mpjpe_mm']:6.1f}mm  wrist={m['val_wrist_pos_mm']:6.1f}mm "
              f" {m['val_wrist_deg']:.2f}deg  jitter={m['val_jitter']:.2f}  "
              f"footskate={m['val_footskate_mm_s']:6.1f}mm/s  (n={m['val_windows']})")

    print("\n| model | MPJPE (mm) | wrist pos (mm) | wrist orient | jitter | foot-skate (mm/s) |")
    print("|---|---|---|---|---|---|")
    for name, m in rows:
        print(f"| {name} | {m['val_mpjpe_mm']:.1f} | {m['val_wrist_pos_mm']:.1f} | "
              f"{m['val_wrist_deg']:.2f}° | {m['val_jitter']:.2f} | {m['val_footskate_mm_s']:.1f} |")


if __name__ == "__main__":
    main()
