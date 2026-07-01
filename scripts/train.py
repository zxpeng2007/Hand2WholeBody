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

from h2b.training import train, train_diffusion, synthetic_clips, default_loss_weights


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
    ap.add_argument("--cache", default="", help="fast pairs cache from scripts/cache_pairs.py")
    ap.add_argument("--keep-labels", default="", help="comma act_cat to KEEP (else all)")
    ap.add_argument("--drop-labels", default="", help="comma act_cat to DROP")
    ap.add_argument("--top-activity-frac", type=float, default=0.0,
                    help="keep only the most wrist-active fraction (0=off); proxy for striking motions")
    ap.add_argument("--pkl", default="", help="upstream train.pkl (SMPL) — FK-extracts the 12D/24D")
    ap.add_argument("--arctic", default="", help="ARCTIC raw_seqs dir (SMPL-X) -> bimanual clips")
    ap.add_argument("--smplx-models", default="",
                    help="dir with smplx/SMPLX_*.{npz,pkl} for ARCTIC FK (e.g. assets/models)")
    ap.add_argument("--wrist-count", type=int, default=0,
                    help="1 (left paddle) or 2 (bimanual); 0 = use config hand_signal.wrist_count")
    ap.add_argument("--pairs", default="", help="dir of pre-extracted pair_*.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="cap #sequences from --pkl (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction (by sequence)")
    ap.add_argument("--eval-every", type=int, default=0, help="run held-out eval every N steps (0=off)")
    ap.add_argument("--log-every", type=int, default=200, help="print train metrics every N steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    weights = cfg.get("loss", default_loss_weights())
    from h2b.representations import frames as F
    wrist_count = args.wrist_count or int(cfg.get("hand_signal", {}).get("wrist_count", 1))
    wrists = F.WRIST_JOINTS[:wrist_count]

    rest_joints = None
    if args.cache:
        from h2b.data.cache import load_pairs_cache
        clips, rest_joints = load_pairs_cache(
            args.cache,
            keep_labels=[s for s in args.keep_labels.split(",") if s] or None,
            drop_labels=[s for s in args.drop_labels.split(",") if s] or None)
        if args.limit:
            clips = clips[:args.limit]
        print(f"loaded {len(clips)} sequences from cache {args.cache}; "
              f"rest_joints {'calibrated' if rest_joints is not None else 'approx'}")
    elif args.pkl:
        from h2b.data.pkl_loader import load_clips
        clips, rest_joints = load_clips(args.pkl, fps=cfg["frame"]["fps"],
                                        limit=(args.limit or None), wrists=wrists)
        print(f"loaded {len(clips)} sequences from {args.pkl}; "
              f"rest_joints {'calibrated' if rest_joints is not None else 'approx'}")
    elif args.arctic:
        from h2b.data.arctic_loader import load_arctic_clips
        clips, rest_joints = load_arctic_clips(args.arctic, model_dir=args.smplx_models,
                                               fps=cfg["frame"]["fps"], wrists=wrists,
                                               limit=(args.limit or None))
        print(f"loaded {len(clips)} ARCTIC (SMPL-X) sequences; wrists={wrists}; "
              f"rest_joints {'calibrated' if rest_joints is not None else 'approx'}")
    elif args.pairs:
        clips = load_pairs(args.pairs)
        print(f"loaded {len(clips)} clips from {args.pairs}")
    else:
        print("using synthetic self-consistent clips")
        clips = synthetic_clips(n_clips=16, T=96)

    import torch
    from h2b.eval import split_clips
    device = args.device if torch.cuda.is_available() else "cpu"
    w = {k: weights[k] for k in
         ("trans", "rot6d", "velocity", "fk_joint", "hand_consistency", "foot_contact")
         if k in weights} or None

    if args.top_activity_frac > 0:
        from h2b.data.cache import filter_by_activity
        n0 = len(clips)
        clips, acts = filter_by_activity(clips, top_frac=args.top_activity_frac)
        print(f"activity filter: kept {len(clips)}/{n0} most-active clips "
              f"(wrist speed >= {float(np.quantile(acts, 1 - args.top_activity_frac)):.2f} m/s)")

    if args.val_frac > 0 and len(clips) > 1:
        train_clips, val_clips = split_clips(clips, val_frac=args.val_frac)
        print(f"split: {len(train_clips)} train / {len(val_clips)} val sequences")
    else:
        train_clips, val_clips = clips, None

    hand_dim = int(np.asarray(train_clips[0][0]).shape[-1]) if train_clips else 12 * wrist_count
    print(f"wrist_count={wrist_count}  hand_dim={hand_dim}  (12 = 1 wrist, 24 = bimanual)")

    if args.arch == "diffusion":
        model, _diff, history = train_diffusion(train_clips, length=args.length, steps=args.steps,
                                                device=device, weights=w, rest_joints=rest_joints,
                                                val_clips=val_clips, eval_every=args.eval_every,
                                                log_every=args.log_every, hand_dim=hand_dim)
    else:
        model, history = train(train_clips, length=args.length, steps=args.steps, device=device,
                               weights=w, rest_joints=rest_joints,
                               val_clips=val_clips, eval_every=args.eval_every,
                               log_every=args.log_every, hand_dim=hand_dim)

    out = args.out or f"checkpoints/{args.arch}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
