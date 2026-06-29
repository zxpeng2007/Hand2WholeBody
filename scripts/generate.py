"""Generate whole-body SMPL motion from a hand 12D sequence and export for Stage 3.

    # from a trained checkpoint + a hand sequence file (.npy or .npz with key 'hand12', (T,12)):
    python scripts/generate.py --arch diffusion --checkpoint checkpoints/diffusion.pt \
        --hand my_hand.npy --out out.npz --viz out.png

    # quick demo on a synthetic hand sequence (no real input needed):
    python scripts/generate.py --arch regressor --checkpoint checkpoints/regressor.pt --synthetic --viz demo.png

Output is an AMASS-style SMPL .npz (Stage-3 handoff for GMR -> HoloMotion) plus an optional
skeleton montage PNG for a quick eyeball.
"""

from __future__ import annotations

import argparse

import numpy as np


def load_hand(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        d = np.load(path)
        return np.asarray(d["hand12"] if "hand12" in d else d[d.files[0]], np.float32)
    return np.asarray(np.load(path), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="regressor", choices=["regressor", "diffusion"])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--hand", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="generated.npz", help="AMASS-style SMPL .npz")
    ap.add_argument("--gmr-out", default="", help="also write a GMR-ready SMPL-X .npz for Stage 3")
    ap.add_argument("--height", type=float, default=1.75, help="subject height (m) for GMR betas[0] scale")
    ap.add_argument("--viz", default="")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from h2b.inference import generate_to_npz, generate
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.synthetic or not args.hand:
        from h2b.training import synthetic_clips
        hand = synthetic_clips(n_clips=1, T=64)[0][0]
    else:
        hand = load_hand(args.hand)

    diffusion = None
    if args.arch == "diffusion":
        from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
        model = DiTDenoiser(hidden=args.hidden, n_layers=args.n_layers).to(device)
        diffusion = GaussianDiffusion(device=device)
    else:
        from h2b.models.regressor import RegressorHand2Body
        model = RegressorHand2Body(hidden=args.hidden, n_layers=args.n_layers).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    path = generate_to_npz(args.out, model, hand, arch=args.arch, diffusion=diffusion,
                           sample_steps=args.steps, device=device)
    print(f"wrote SMPL motion: {path}  (frames={len(hand)})")

    if args.gmr_out:
        from h2b.inference import generate_to_smplx_npz
        gp = generate_to_smplx_npz(args.gmr_out, model, hand, arch=args.arch, diffusion=diffusion,
                                   sample_steps=args.steps, device=device, height_m=args.height)
        print(f"wrote GMR-ready SMPL-X: {gp}")

    if args.viz:
        from h2b.export.visualize import plot_skeleton_montage
        motion = generate(model, hand, arch=args.arch, diffusion=diffusion,
                          sample_steps=args.steps, device=device)
        plot_skeleton_montage(motion, args.viz, title=f"{args.arch} generated")
        print(f"wrote viz: {args.viz}")


if __name__ == "__main__":
    main()
