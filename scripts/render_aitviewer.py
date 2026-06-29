"""Headless aitviewer render -> mp4 of the SMPL MESH (proper body, not a skeleton).

Requires the licensed SMPL model files on disk (smpl.is.tue.mpg.de). Convert the official
release with scripts/clean_smpl_models.py, then pass --smpl-models <dir> (folder containing
`smpl/SMPL_{GENDER}.pkl`).

    # 30-second mesh video from a checkpoint, elevated wide camera:
    python scripts/render_aitviewer.py --cache data/cache/pairs_full.npz \
        --checkpoint checkpoints/diffusion_full.pt --smpl-models C:/.../smpl_models \
        --seconds 30 --out Downloads/h2b_mesh.mp4

aitviewer is Y-up internally while our data is Z-up (z_up=True remaps data-z -> viewer-y), so
the camera is set in the VIEWER frame: x = table length, y = height, z = -table width.
aitviewer's save_video(video_dir=X.mp4) appends a suffix -> the file is X_0.mp4.
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="our AMASS-style .npz (poses (T,72), trans)")
    ap.add_argument("--cache", default="", help="generate from a held-out cache clip instead")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--arch", default="diffusion", choices=["diffusion", "regressor"])
    ap.add_argument("--smpl-models", default="", help="dir with SMPL model files (sets aitviewer config)")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--model-type", default="smpl", choices=["smpl", "smplx"])
    ap.add_argument("--out", default="h2b_mesh.mp4")
    ap.add_argument("--seconds", type=float, default=0.0, help="target duration; 0 = use --max-frames")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--fps", type=int, default=30)
    # camera (viewer frame, y-up): x along table length, y up, z = -table width
    ap.add_argument("--cam-height", type=float, default=5.0)
    ap.add_argument("--cam-dist", type=float, default=7.5)
    ap.add_argument("--cam-target-x", type=float, default=-0.3, help="center between player(~-2) and table(0)")
    args = ap.parse_args()

    target_frames = int(args.seconds * args.fps) if args.seconds > 0 else args.max_frames

    # Set config BEFORE the renderer is constructed: playback_fps must equal output fps or
    # aitviewer subsamples (duration = n_frames / playback_fps; default 60 halves a 30fps export).
    from aitviewer.configuration import CONFIG as C
    conf = {"playback_fps": float(args.fps)}
    if args.smpl_models:
        conf["smplx_models"] = args.smpl_models
    try:
        C.update_conf(conf)
    except Exception:
        C.playback_fps = float(args.fps)
        if args.smpl_models:
            C.smplx_models = args.smpl_models

    if args.input:
        d = np.load(args.input, allow_pickle=True)
        poses, trans = np.asarray(d["poses"]), np.asarray(d["trans"])
    elif args.cache:
        import torch
        from h2b.data.cache import load_pairs_cache, clip_wrist_activity
        from h2b.eval import split_clips
        from h2b.representations import body as B
        from h2b import inference as INF
        from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
        from h2b.models.regressor import RegressorHand2Body
        clips, _ = load_pairs_cache(args.cache)
        _, val = split_clips(clips, val_frac=0.1, seed=0)
        lens = np.array([len(c[0]) for c in val])
        acts = np.array([clip_wrist_activity(c) for c in val])
        elig = np.where(lens >= target_frames)[0]
        idx = int(elig[np.argmax(acts[elig])]) if len(elig) else int(np.argmax(lens))
        hand = val[idx][0][:target_frames]
        print(f"val clip {idx}: {len(hand)} frames, activity {clip_wrist_activity((hand, hand)):.2f} m/s")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if args.arch == "diffusion":
            model = DiTDenoiser(hidden=256, n_layers=4).to(dev); diff = GaussianDiffusion(device=dev)
        else:
            model = RegressorHand2Body(hidden=256, n_layers=4).to(dev); diff = None
        model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
        motion = INF.generate_long(model, hand, arch=args.arch, diffusion=diff,
                                   sample_steps=8, device=dev)
        poses, trans = B.motion_to_smpl72(motion)
    else:
        raise SystemExit("pass --input <npz> or --cache <npz> + --checkpoint")

    poses, trans = poses[:target_frames], trans[:target_frames]
    from aitviewer.headless import HeadlessRenderer
    from h2b.export.aitviewer_vis import _table_mesh, _net_mesh, _smpl_sequence
    seq = _smpl_sequence(poses, trans, args.gender, args.model_type)
    r = HeadlessRenderer()
    r.scene.add(_table_mesh()); r.scene.add(_net_mesh()); r.scene.add(seq)
    # elevated, pulled-back camera (viewer frame: y up, z = -table width)
    try:
        c = r.scene.camera
        c.target = np.array([args.cam_target_x, 0.6, 0.0], np.float32)
        c.position = np.array([args.cam_target_x, args.cam_height, args.cam_dist], np.float32)
        c.up = np.array([0.0, 1.0, 0.0], np.float32)
    except Exception as e:
        print("camera set skipped:", e)
    # aitviewer subsamples by playback_fps vs output_fps; match them so every frame renders
    # (scene duration = n_frames / playback_fps; default playback is 60 -> halves a 30fps export).
    for _a in ("playback_fps", "_playback_fps"):
        if hasattr(r, _a):
            try:
                setattr(r, _a, args.fps)
            except Exception:
                pass
    r.save_video(video_dir=args.out, output_fps=args.fps)
    print(f"wrote {args.out} ({len(poses)} frames, {len(poses)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
