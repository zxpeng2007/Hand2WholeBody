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
    ap.add_argument("--ghost-wrist", action="store_true",
                    help="overlay a translucent sphere at the GT/input wrist (input 12D hand pos)")
    ap.add_argument("--ghost-radius", type=float, default=0.045)
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

    gt_pos = gt_R = gen_pos = gen_R = None
    if args.input:
        d = np.load(args.input, allow_pickle=True)
        poses, trans = np.asarray(d["poses"]), np.asarray(d["trans"])
        gt_pos = np.asarray(d["wrist"]) if "wrist" in d else None
        gt_R = np.asarray(d["wrist_R"]) if "wrist_R" in d else None
        gen_pos = np.asarray(d["gen_wrist"]) if "gen_wrist" in d else None
        gen_R = np.asarray(d["gen_wrist_R"]) if "gen_wrist_R" in d else None
    elif args.cache:
        import torch
        from h2b.data.cache import load_pairs_cache, clip_wrist_activity
        from h2b.eval import split_clips
        from h2b.representations import body as B
        from h2b.representations import rotations_torch as RT
        from h2b.models import fk_torch as FK
        from h2b import inference as INF
        from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
        from h2b.models.regressor import RegressorHand2Body
        clips, rest = load_pairs_cache(args.cache)
        _, val = split_clips(clips, val_frac=0.1, seed=0)
        lens = np.array([len(c[0]) for c in val])
        acts = np.array([clip_wrist_activity(c) for c in val])
        elig = np.where(lens >= target_frames)[0]
        idx = int(elig[np.argmax(acts[elig])]) if len(elig) else int(np.argmax(lens))
        hand = np.asarray(val[idx][0][:target_frames], np.float32)
        gt_pos = hand[:, 0:3]
        gt_R = RT.rotation_6d_to_matrix(torch.from_numpy(hand[:, 6:12])).numpy()
        print(f"val clip {idx}: {len(hand)} frames, activity {clip_wrist_activity((hand, hand)):.2f} m/s")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if args.arch == "diffusion":
            model = DiTDenoiser(hidden=256, n_layers=4).to(dev); diff = GaussianDiffusion(device=dev)
        else:
            model = RegressorHand2Body(hidden=256, n_layers=4).to(dev); diff = None
        model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
        motion = INF.generate_long(model, hand, arch=args.arch, diffusion=diff,
                                   sample_steps=8, device=dev)
        rest_t = None if rest is None else torch.as_tensor(rest, dtype=torch.float32)
        gwp, gwr = FK.left_wrist_pose(torch.tensor(motion)[None], rest_t)
        gen_pos = gwp[0].numpy()
        gen_R = RT.rotation_6d_to_matrix(gwr[0]).numpy()
        poses, trans = B.motion_to_smpl72(motion)
    else:
        raise SystemExit("pass --input <npz> or --cache <npz> + --checkpoint")

    poses, trans = poses[:target_frames], trans[:target_frames]
    from aitviewer.headless import HeadlessRenderer
    from h2b.export.aitviewer_vis import _table_mesh, _net_mesh, _smpl_sequence, wrist_overlays
    seq = _smpl_sequence(poses, trans, args.gender, args.model_type)
    r = HeadlessRenderer()
    r.scene.add(_table_mesh()); r.scene.add(_net_mesh()); r.scene.add(seq)
    if args.ghost_wrist:
        ov = wrist_overlays(gt_pos=gt_pos, gt_R=gt_R, gen_pos=gen_pos, gen_R=gen_R,
                            n=target_frames, sphere_r=args.ghost_radius)
        if not ov:
            print("--ghost-wrist requested but no wrist data available")
        for node in ov:
            r.scene.add(node)
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
    # aitviewer's save_video never overwrites — it appends _0/_1/... So a re-render leaves the
    # OLD _0 in place and writes _1, and anything globbing _0 silently picks up the stale clip.
    # Delete prior outputs for this base so the fresh render is always <base>_0.mp4.
    import glob
    import os as _os
    for _f in glob.glob(_os.path.splitext(args.out)[0] + "_*.mp4"):
        try:
            _os.remove(_f)
        except OSError:
            pass
    r.save_video(video_dir=args.out, output_fps=args.fps)
    print(f"wrote {args.out} ({len(poses)} frames, {len(poses)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
