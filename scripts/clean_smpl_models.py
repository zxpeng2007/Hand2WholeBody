"""Convert the official SMPL v1.1.0 models (chumpy, numpy-1) -> smplx/aitviewer-ready files.

    python scripts/clean_smpl_models.py \
        --src "C:/Users/ZixuanPeng/Downloads/SMPL_python_v.1.1.0/smpl/models" \
        --out "C:/Users/ZixuanPeng/Downloads/smpl_models"

Writes <out>/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl (the layout smplx.create / aitviewer expect).
The official .pkl store chumpy arrays and use removed numpy aliases; we patch the aliases so
chumpy imports under numpy 2.x, then convert chumpy/scipy-sparse values to plain numpy so the
output needs neither chumpy nor scipy at load time.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import inspect

import numpy as np

# chumpy was written for old numpy + py<=3.10; patch the bits removed since.
for _name, _t in [("bool", np.bool_), ("int", np.int_), ("float", np.float64),
                  ("complex", np.complex128), ("object", np.object_),
                  ("str", np.str_), ("unicode", np.str_)]:
    if not hasattr(np, _name):
        setattr(np, _name, _t)
if not hasattr(inspect, "getargspec"):           # removed in Python 3.11
    inspect.getargspec = inspect.getfullargspec

import chumpy  # noqa: E402  (import after the shims)

GENDER = {"f": "FEMALE", "m": "MALE", "neutral": "NEUTRAL"}


def to_numpy(v):
    m = getattr(type(v), "__module__", "") or ""
    if m.startswith("chumpy"):
        return np.array(v)
    if "scipy.sparse" in m:
        return np.asarray(v.todense())
    return v


def clean_one(src, dst):
    with open(src, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    out = {k: to_numpy(v) for k, v in data.items()}
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with basicmodel_*_lbs_*.pkl")
    ap.add_argument("--out", required=True, help="output models root (creates <out>/smpl/)")
    args = ap.parse_args()
    smpl_dir = os.path.join(args.out, "smpl")
    os.makedirs(smpl_dir, exist_ok=True)
    srcs = glob.glob(os.path.join(args.src, "basicmodel_*_lbs_*.pkl"))
    if not srcs:
        raise SystemExit(f"no basicmodel_*_lbs_*.pkl in {args.src}")
    for src in srcs:
        g = os.path.basename(src).split("_")[1]
        gname = GENDER.get(g)
        if not gname:
            print("skip", os.path.basename(src)); continue
        dst = os.path.join(smpl_dir, f"SMPL_{gname}.pkl")
        d = clean_one(src, dst)
        print(f"{os.path.basename(src)} -> {dst}  v_template={np.asarray(d['v_template']).shape}")


if __name__ == "__main__":
    main()
