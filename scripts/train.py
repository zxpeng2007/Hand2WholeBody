"""Train the M2 regressor.

    # smoke-test the whole loop on synthetic self-consistent data (no real data needed):
    python scripts/train.py --synthetic --steps 300

    # train on extracted pairs (scripts/extract_amass.py output):
    python scripts/train.py --pairs data/pairs --config configs/default.yaml --steps 20000
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import yaml

from h2wb.training import train, train_diffusion, synthetic_clips, default_loss_weights


def load_pairs(pairs_dir: str):
    clips = []
    for fp in sorted(glob.glob(os.path.join(pairs_dir, "*.npz"))):
        d = np.load(fp)
        clips.append((d["hand12"], d["body"]))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="regressor", choices=["regressor", "diffusion"])
    ap.add_argument("--pairs", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    weights = cfg.get("loss", default_loss_weights())

    if args.synthetic or not args.pairs:
        print("using synthetic self-consistent clips")
        clips = synthetic_clips(n_clips=16, T=96)
    else:
        clips = load_pairs(args.pairs)
        print(f"loaded {len(clips)} clips from {args.pairs}")

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    w = {k: weights[k] for k in
         ("trans", "rot6d", "velocity", "fk_joint", "hand_consistency") if k in weights} or None

    if args.arch == "diffusion":
        model, _diff, history = train_diffusion(clips, length=args.length, steps=args.steps,
                                                device=device, weights=w)
    else:
        model, history = train(clips, length=args.length, steps=args.steps, device=device, weights=w)

    for h in history:
        print(h)
    out = args.out or f"checkpoints/{args.arch}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
