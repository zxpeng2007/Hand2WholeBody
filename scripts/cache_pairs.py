"""One-time: extract (hand12, body) pairs from train.pkl -> fast numpy cache.

    python scripts/cache_pairs.py --pkl train.pkl --out data/cache/pairs_full.npz

Also prints the act_cat label distribution so we can choose a label filter for training.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

from h2b.data.pkl_loader import load_smpl_pkl, _motion_of, _poses_to_72, sequence_to_pair, calibrate_rest_joints
from h2b.data.cache import save_pairs_cache


def _seq_labels(item):
    """Union of act_cat strings across the sequence's frame_labels."""
    cats = set()
    for fl in (item.get("frame_labels") or []) if isinstance(item, dict) else []:
        for c in (fl.get("act_cat") or []):
            cats.add(str(c))
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="data/cache/pairs_full.npz")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--calib-seqs", type=int, default=32)
    args = ap.parse_args()

    payload = load_smpl_pkl(args.pkl)
    if not isinstance(payload, list):
        raise SystemExit("expected the real list-of-sequences train.pkl format")

    clips, labels, names, rests = [], [], [], []
    cat_counter = Counter()
    for i, item in enumerate(payload):
        if args.limit and len(clips) >= args.limit:
            break
        m = _motion_of(item)
        if not isinstance(m, dict) or "poses" not in m or "trans" not in m:
            continue
        if np.asarray(m["poses"]).shape[0] < args.min_frames:
            continue
        seq = {"poses": np.asarray(m["poses"], np.float64), "trans": np.asarray(m["trans"], np.float64),
               "joints": (np.asarray(m["joints"], np.float64) if m.get("joints") is not None else None),
               "betas": np.asarray(m.get("betas", np.zeros(10)), np.float64)}
        clips.append(sequence_to_pair(seq, fps=args.fps))
        cats = _seq_labels(item)
        cat_counter.update(cats)
        labels.append(",".join(sorted(cats)))
        names.append(item.get("seq_name", "") if isinstance(item, dict) else "")
        if seq["joints"] is not None and len(rests) < args.calib_seqs:
            rests.append(calibrate_rest_joints(_poses_to_72(seq["poses"]), seq["joints"], seq["trans"]))

    rest = np.mean(rests, axis=0) if rests else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_pairs_cache(args.out, clips, labels, rest, names=names)

    total_frames = sum(len(c[0]) for c in clips)
    print(f"cached {len(clips)} sequences / {total_frames} frames -> {args.out}")
    print(f"rest_joints: {'calibrated' if rest is not None else 'none'}")
    print("act_cat distribution (top 30):")
    for cat, n in cat_counter.most_common(30):
        print(f"  {n:6d}  {cat}")


if __name__ == "__main__":
    main()
