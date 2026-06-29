"""Offline generation: a hand 12D sequence -> whole-body SMPL -> AMASS-style .npz.

Single-shot over the whole sequence (the causal transformer handles any length up to its
positional budget). For true online/streaming use h2b.models.streaming.DiffusionStreamer.
The exported .npz is the Stage-3 handoff (CONTRACT §3) for GMR -> HoloMotion.
"""

from __future__ import annotations

import numpy as np

from .representations import body as B
from .representations import frames as F


def generate(model, hand12, arch="regressor", diffusion=None, sample_steps=8, device="cpu"):
    """hand12 (T,12) world -> motion (T,135) world. Canonicalizes by the first-frame hand pos."""
    import torch
    hand = np.asarray(hand12, np.float32)
    T = hand.shape[0]
    max_len = getattr(getattr(model, "pos", None), "shape", [0, 999])[1]
    if T > max_len:
        raise ValueError(f"sequence length {T} exceeds model positional budget {max_len}; "
                         "use h2b.models.streaming.DiffusionStreamer for long/online sequences")
    anchor = hand[0:1, F.HAND12_POS].copy()                 # (1,3) inference anchor
    hand_c = hand.copy()
    hand_c[:, F.HAND12_POS] -= anchor
    ht = torch.from_numpy(hand_c)[None].to(device)          # (1,T,12)
    model.eval()
    with torch.no_grad():
        if arch == "diffusion":
            motion = diffusion.ddim_sample(model, (1, T, B.MOTION_DIM), ht,
                                           steps=sample_steps, device=device)
        else:
            motion = model(ht)
    motion = motion[0].cpu().numpy()                        # (T,135)
    motion[:, B.B_TRANS] += anchor                          # de-canonicalize -> world
    return motion


def generate_long(model, hand12, chunk=250, overlap=50, **kw):
    """Generate arbitrarily long motion by overlapping chunks (the model is capped at ~256
    frames single-shot). Each chunk is generated in world coords (generate() de-canonicalizes),
    then cross-faded across overlaps for continuity. hand12 (T,12) -> motion (T,135)."""
    hand12 = np.asarray(hand12, np.float32)
    T = hand12.shape[0]
    if T <= chunk:
        return generate(model, hand12, **kw)
    step = chunk - overlap
    starts = list(range(0, T - chunk + 1, step))
    if starts[-1] != T - chunk:
        starts.append(T - chunk)
    out = np.zeros((T, B.MOTION_DIM), np.float32)
    out[0:chunk] = generate(model, hand12[0:chunk], **kw)
    prev_end = chunk
    for s in starts[1:]:
        seg = generate(model, hand12[s:s + chunk], **kw)             # covers [s, s+chunk)
        ov = prev_end - s                                            # overlap length
        if ov > 0:
            a = np.linspace(1.0, 0.0, ov, dtype=np.float32)[:, None]
            out[s:prev_end] = a * out[s:prev_end] + (1.0 - a) * seg[:ov]
            out[prev_end:s + chunk] = seg[ov:]
        else:
            out[s:s + chunk] = seg
        prev_end = s + chunk
    return out


def generate_to_npz(path, model, hand12, betas=None, fps=30, gender="neutral", **kw):
    """Generate and write the AMASS-style SMPL .npz. Returns the path."""
    from .export.to_amass_npz import smpl_motion_to_amass_npz
    motion = generate(model, hand12, **kw)
    poses72, trans = B.motion_to_smpl72(motion)
    betas = np.zeros(10, np.float32) if betas is None else np.asarray(betas, np.float32)
    return smpl_motion_to_amass_npz(path, poses72, trans, betas, fps=fps, gender=gender)


def generate_to_smplx_npz(path, model, hand12, betas=None, fps=30, gender="neutral",
                          height_m=1.75, **kw):
    """Generate and write the GMR-ready SMPL-X .npz (Stage-3 ingest). Returns the path."""
    from .export.to_amass_npz import smpl_motion_to_smplx_npz
    motion = generate(model, hand12, **kw)
    poses72, trans = B.motion_to_smpl72(motion)
    betas = np.zeros(10, np.float32) if betas is None else np.asarray(betas, np.float32)
    return smpl_motion_to_smplx_npz(path, poses72, trans, betas, fps=fps, gender=gender,
                                    height_m=height_m)
