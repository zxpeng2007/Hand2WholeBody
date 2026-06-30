"""aitviewer-based visualization of SMPL motion (the real renderer; matches the upstream
vis_smpl22_aitviewer.py conventions: SMPLLayer/SMPLSequence, z-up, table/net/ball at 0.76 m).

Visualizes:
  * our generated AMASS-style .npz   (visualize_npz)
  * our 135-D motion array            (visualize_motion)
  * the upstream train.pkl sequence (visualize_pkl)  -> handy to eyeball the raw data.

Requires aitviewer + SMPL model files configured in aitviewer's config (licensed; not in the
repo) and a display/GL context — runs on a workstation, not headless CI. Imports are lazy so
the rest of h2b never depends on aitviewer.

    python -m h2b.export.aitviewer_vis --input generated.npz
    python -m h2b.export.aitviewer_vis --input train.pkl --seq_idx 0 --model_type smpl
"""

from __future__ import annotations

import argparse

import numpy as np

from ..representations import body as B
from ..representations import frames as F


# --------------------------------------------------------------------------- #
# scene meshes (z-up), ported from the upstream viewer
# --------------------------------------------------------------------------- #
def _table_mesh(height=F.TABLE_TOP_Z, length=F.TABLE_LENGTH_X, width=F.TABLE_WIDTH_Y, thick=0.016):
    import trimesh
    from aitviewer.renderables.meshes import Meshes
    t = trimesh.creation.box(extents=(length, width, thick))
    t.vertices[:, 2] += float(height) - thick * 0.5
    m = Meshes(vertices=t.vertices, faces=t.faces, z_up=True)
    m.color = (0.10, 0.35, 0.22, 1.0)
    return m


def _net_mesh(height=F.TABLE_TOP_Z, width=F.TABLE_WIDTH_Y, net_h=0.1525, net_t=0.005):
    import trimesh
    from aitviewer.renderables.meshes import Meshes
    n = trimesh.creation.box(extents=(net_t, width, net_h))
    n.vertices[:, 2] += float(height) + net_h * 0.5
    m = Meshes(vertices=n.vertices, faces=n.faces, z_up=True)
    m.color = (0.85, 0.85, 0.85, 0.85)
    return m


def _ball_mesh(ball_trans, radius=0.02):
    import trimesh
    from aitviewer.renderables.meshes import Meshes
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    tf = np.tile(np.eye(4, dtype=np.float32), (ball_trans.shape[0], 1, 1))
    tf[:, :3, 3] = np.asarray(ball_trans, np.float32)
    m = Meshes(vertices=sphere.vertices, faces=sphere.faces,
               instance_transforms=tf[:, None], z_up=True)
    m.color = (0.95, 0.42, 0.28, 1.0)
    return m


def _wrist_ghost(wrist_trans, radius=0.045, color=(0.95, 0.2, 0.2, 0.6)):
    """Translucent sphere following a (T,3) world wrist trajectory — the ground-truth/target
    wrist (the input 12D hand position). Overlay on a generated body to see wrist deviation."""
    import trimesh
    from aitviewer.renderables.meshes import Meshes
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    wt = np.asarray(wrist_trans, np.float32)
    tf = np.tile(np.eye(4, dtype=np.float32), (wt.shape[0], 1, 1))
    tf[:, :3, 3] = wt
    m = Meshes(vertices=sphere.vertices, faces=sphere.faces,
               instance_transforms=tf[:, None], z_up=True)
    m.color = color
    return m


# data is z-up; aitviewer is y-up. This is the same remap meshes.py applies for z_up=True,
# but RigidBodies/Lines don't support z_up, so we apply it to positions/orientations directly.
_R_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)


def _rolling_trail(pos, window):
    """(T,3) -> (T, window, 3): each frame holds the last `window` positions (a rolling trail)."""
    T = pos.shape[0]
    out = np.empty((T, window, 3), np.float32)
    for t in range(T):
        seg = pos[max(0, t - window + 1):t + 1]
        if len(seg) < window:
            seg = np.concatenate([np.repeat(seg[:1], window - len(seg), 0), seg], 0)
        out[t] = seg
    return out


def wrist_overlays(gt_pos=None, gt_R=None, gen_pos=None, gen_R=None, n=None,
                   axis_len=0.20, sphere_r=0.030, trail_w=0.005, trail_window=22):
    """Renderables for the GT and generated wrist: a colored sphere + an xyz orientation gizmo
    (x=red,y=green,z=blue arrows) per frame, plus a rolling motion-trail (last `trail_window` frames).
      GT  wrist -> red sphere,  orange trail.   generated wrist -> cyan sphere, blue trail.
    pos: (T,3) world; R: (T,3,3) world rotation (axes = columns). Returns a list of renderables."""
    from aitviewer.renderables.rigid_bodies import RigidBodies
    from aitviewer.renderables.lines import Lines
    out = []

    def add(pos, R, sph_color, line_color):
        if pos is None:
            return
        pos = np.asarray(pos, np.float32)[:n]
        posv = pos @ _R_ZUP.T                                        # -> viewer frame
        out.append(Lines(_rolling_trail(posv, trail_window), r_base=trail_w, color=line_color,
                         mode="line_strip", cast_shadow=False))
        if R is not None:
            oriv = np.einsum("ij,tjk->tik", _R_ZUP, np.asarray(R, np.float32)[:n])
            out.append(RigidBodies(posv[:, None], oriv[:, None], radius=sphere_r,
                                   length=axis_len, radius_cylinder=axis_len / 24,
                                   color=sph_color))
        else:
            out.append(_wrist_ghost(pos, radius=sphere_r, color=sph_color))   # _wrist_ghost re-applies z_up

    add(gt_pos, gt_R, (0.90, 0.15, 0.15, 1.0), (1.0, 0.5, 0.0, 1.0))          # GT: red / orange
    add(gen_pos, gen_R, (0.10, 0.80, 0.92, 1.0), (0.10, 0.40, 1.0, 1.0))      # gen: cyan / blue
    return out


def _smpl_sequence(poses_aa, trans, gender="neutral", model_type="smpl"):
    """Build an SMPLSequence from (T, >=66) axis-angle + (T,3) trans, padding body to model dim."""
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    poses_aa = np.asarray(poses_aa, np.float32)
    trans = np.asarray(trans, np.float32)
    T = min(poses_aa.shape[0], trans.shape[0])
    poses_aa, trans = poses_aa[:T], trans[:T]
    root = poses_aa[:, :3]
    body_raw = poses_aa[:, 3:]
    layer = SMPLLayer(model_type=model_type, gender=gender)
    exp = int(layer.bm.NUM_BODY_JOINTS) * 3
    if body_raw.shape[1] >= exp:
        body = body_raw[:, :exp]
    else:
        body = np.concatenate([body_raw, np.zeros((T, exp - body_raw.shape[1]), np.float32)], axis=1)
    seq = SMPLSequence(poses_body=body, smpl_layer=layer, poses_root=root,
                       trans=trans, z_up=True)
    seq.color = (136 / 255.0, 156 / 255.0, 216 / 255.0, 1.0)
    return seq


def visualize(poses_aa, trans, gender="neutral", model_type="smpl",
              show_table=True, show_net=True, ball_pos=None, ball_radius=0.02):
    """Open the aitviewer with the SMPL sequence + table/net (+ optional ball)."""
    from aitviewer.viewer import Viewer
    v = Viewer()
    if show_table:
        v.scene.add(_table_mesh())
    if show_net:
        v.scene.add(_net_mesh())
    v.scene.add(_smpl_sequence(poses_aa, trans, gender, model_type))
    if ball_pos is not None:
        v.scene.add(_ball_mesh(np.asarray(ball_pos), radius=ball_radius))
    v.run()


def visualize_npz(path, **kw):
    """Visualize our exported AMASS-style .npz (keys poses (T,72), trans (T,3))."""
    d = np.load(path, allow_pickle=True)
    visualize(d["poses"], d["trans"], **kw)


def visualize_motion(motion, **kw):
    """Visualize our 135-D motion array."""
    poses72, trans = B.motion_to_smpl72(np.asarray(motion))
    visualize(poses72, trans, **kw)


def visualize_pkl(path, seq_idx=0, **kw):
    """Visualize one sequence from the upstream train.pkl (poses (T,66), trans)."""
    from ..data.pkl_loader import load_smpl_pkl, iter_sequences
    seqs = list(iter_sequences(load_smpl_pkl(path)))
    seq = seqs[seq_idx]
    visualize(seq["poses"], seq["trans"], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".npz (our export) or .pkl (upstream data)")
    ap.add_argument("--seq_idx", type=int, default=0, help="sequence index for .pkl input")
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--model_type", default="smpl", choices=["smpl", "smplx"])
    ap.add_argument("--no_table", action="store_true")
    ap.add_argument("--no_net", action="store_true")
    args = ap.parse_args()
    kw = dict(gender=args.gender, model_type=args.model_type,
              show_table=not args.no_table, show_net=not args.no_net)
    if args.input.endswith(".pkl"):
        visualize_pkl(args.input, seq_idx=args.seq_idx, **kw)
    else:
        visualize_npz(args.input, **kw)


if __name__ == "__main__":
    main()
