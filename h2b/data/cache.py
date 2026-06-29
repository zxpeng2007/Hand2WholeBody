"""Fast on-disk cache of extracted (hand12, body) training pairs + labels + rest skeleton.

Building the pairs from train.pkl costs a 2.5 GB joblib load + FK over ~4M frames. Cache the
result once as flat numpy arrays so subsequent runs load in seconds (np.load, optionally mmap)
and support post-hoc label filtering without re-extraction.

Layout (np.savez): hand (N,12) f32, body (N,135) f32, offsets (n_seq+1,) int64 (sequence
boundaries), labels (object: per-seq comma-joined act_cat set), names (object), rest (22,3) f32.
"""

from __future__ import annotations

import numpy as np


def save_pairs_cache(path, clips, labels, rest, names=None):
    """clips: list of (hand (T,12), body (T,135)); labels: list[str]; rest: (22,3) or None."""
    hand = np.concatenate([c[0] for c in clips]).astype(np.float32)
    body = np.concatenate([c[1] for c in clips]).astype(np.float32)
    offsets = np.zeros(len(clips) + 1, np.int64)
    offsets[1:] = np.cumsum([len(c[0]) for c in clips])
    np.savez(
        path,
        hand=hand, body=body, offsets=offsets,
        labels=np.array(labels, dtype=object),
        names=np.array(names if names is not None else [""] * len(clips), dtype=object),
        rest=(np.zeros((0, 3), np.float32) if rest is None else np.asarray(rest, np.float32)),
    )
    return path


def clip_wrist_activity(clip) -> float:
    """Mean wrist speed (m/s) of a clip — the velocity channel of the 12D signal.

    Motion-based proxy for 'striking vs idle/locomotion', used instead of act_cat (which is
    uniformly 'walk' in train.pkl and so useless for filtering). High = dynamic swing.
    """
    h = np.asarray(clip[0])
    return float(np.linalg.norm(h[:, 3:6], axis=1).mean())


def filter_by_activity(clips, top_frac=None, min_speed=None):
    """Keep clips by wrist activity. top_frac keeps the most-active fraction; min_speed thresholds.
    Returns (filtered_clips, activities)."""
    acts = np.array([clip_wrist_activity(c) for c in clips]) if clips else np.zeros(0)
    keep = np.ones(len(clips), bool)
    if min_speed is not None:
        keep &= acts >= min_speed
    if top_frac is not None and 0 < top_frac < 1 and len(clips):
        keep &= acts >= np.quantile(acts, 1.0 - top_frac)
    return [c for c, k in zip(clips, keep) if k], acts


def load_pairs_cache(path, keep_labels=None, drop_labels=None, mmap=True):
    """Return (clips, rest). Optionally keep/drop sequences whose act_cat set matches."""
    d = np.load(path, allow_pickle=True, mmap_mode=("r" if mmap else None))
    hand, body, offsets = d["hand"], d["body"], d["offsets"]
    labels = d["labels"]
    rest = d["rest"]
    rest = None if getattr(rest, "shape", (0,))[0] == 0 else np.asarray(rest)
    keep = set(keep_labels) if keep_labels else None
    drop = set(drop_labels) if drop_labels else None
    clips = []
    for i in range(len(offsets) - 1):
        lab = set(str(labels[i]).split(",")) if labels[i] else set()
        if keep is not None and not (lab & keep):
            continue
        if drop is not None and (lab & drop):
            continue
        s, e = int(offsets[i]), int(offsets[i + 1])
        clips.append((np.asarray(hand[s:e]), np.asarray(body[s:e])))
    return clips, rest
