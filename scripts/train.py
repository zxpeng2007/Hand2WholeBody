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
    ap.add_argument("--pkl", default="", help="coworker train.pkl (SMPL) — FK-extracts the 12D")
    ap.add_argument("--pairs", default="", help="dir of pre-extracted pair_*.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="cap #sequences from --pkl (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction (by sequence)")
    ap.add_argument("--eval-every", type=int, default=0, help="run held-out eval every N steps (0=off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    weights = cfg.get("loss", default_loss_weights())

    rest_joints = None
    if args.pkl:
        from h2wb.data.pkl_loader import load_clips
        clips, rest_joints = load_clips(args.pkl, fps=cfg["frame"]["fps"],
                                        limit=(args.limit or None))
        print(f"loaded {len(clips)} sequences from {args.pkl}; "
              f"rest_joints {'calibrated' if rest_joints is not None else 'approx'}")
    elif args.pairs:
        clips = load_pairs(args.pairs)
        print(f"loaded {len(clips)} clips from {args.pairs}")
    else:
        print("using synthetic self-consistent clips")
        clips = synthetic_clips(n_clips=16, T=96)

    import torch
    from h2wb.eval import split_clips
    device = args.device if torch.cuda.is_available() else "cpu"
    w = {k: weights[k] for k in
         ("trans", "rot6d", "velocity", "fk_joint", "hand_consistency") if k in weights} or None

    if args.val_frac > 0 and len(clips) > 1:
        train_clips, val_clips = split_clips(clips, val_frac=args.val_frac)
        print(f"split: {len(train_clips)} train / {len(val_clips)} val sequences")
    else:
        train_clips, val_clips = clips, None

    if args.arch == "diffusion":
        model, _diff, history = train_diffusion(train_clips, length=args.length, steps=args.steps,
                                                device=device, weights=w, rest_joints=rest_joints,
                                                val_clips=val_clips, eval_every=args.eval_every)
    else:
        model, history = train(train_clips, length=args.length, steps=args.steps, device=device,
                               weights=w, rest_joints=rest_joints,
                               val_clips=val_clips, eval_every=args.eval_every)

    for h in history:
        print(h)
    out = args.out or f"checkpoints/{args.arch}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
