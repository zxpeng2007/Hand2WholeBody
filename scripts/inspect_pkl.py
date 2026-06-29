"""Inspect an arbitrary .pkl (e.g. the upstream train.pkl of SMPL data) and print its
structure: types, dict keys, array shapes/dtypes, lengths, and small value samples.

    python scripts/inspect_pkl.py path/to/train.pkl

Use this the moment train.pkl arrives to learn its schema, then wire a loader in
h2b/data/ that maps it to (poses (T,72) axis-angle, trans (T,3), betas, fps) so
scripts/extract_amass.py-style FK extraction can derive the 12D left-wrist signal.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def _load(path):
    """Prefer the joblib loader (with numpy._core shim); fall back to plain pickle."""
    try:
        from h2b.data.pkl_loader import load_smpl_pkl
        return load_smpl_pkl(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def describe(obj, name="root", depth=0, max_depth=4, max_keys=40):
    pad = "  " * depth
    t = type(obj).__name__
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict ({len(obj)} keys)")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_keys:
                print(f"{pad}  ... (+{len(obj) - max_keys} more keys)")
                break
            if depth < max_depth:
                describe(v, repr(k), depth + 1, max_depth, max_keys)
            else:
                print(f"{pad}  {k!r}: {type(v).__name__}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {t} (len {len(obj)})")
        if obj and depth < max_depth:
            describe(obj[0], "[0]", depth + 1, max_depth, max_keys)
    elif isinstance(obj, np.ndarray):
        rng = ""
        if obj.size and np.issubdtype(obj.dtype, np.number):
            rng = f"  min={np.nanmin(obj):.4g} max={np.nanmax(obj):.4g}"
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}{rng}")
    elif hasattr(obj, "shape"):                       # torch tensor etc.
        print(f"{pad}{name}: {t} shape={tuple(obj.shape)} dtype={getattr(obj, 'dtype', '?')}")
    else:
        s = repr(obj)
        print(f"{pad}{name}: {t} = {s[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()
    obj = _load(args.path)
    describe(obj, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
