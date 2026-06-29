"""SCHEMATIC quicklook only — a matplotlib 3D stick figure (headless, no SMPL model needed).

For real visualization use h2b.export.aitviewer_vis (proper SMPL mesh, matches the team's
viewer). This module exists only as a zero-dependency, CI-testable sanity check: it uses the
APPROXIMATE rest skeleton (smpl_fk._approx_rest_joints), so limb lengths/topology are NOT
accurate — it shows gross motion/reach, not a faithful body. Do not judge pose quality from it.
"""

from __future__ import annotations

import numpy as np

from ..data import smpl_fk as FK
from ..representations import body as B
from ..representations import frames as F


def motion_to_joint_positions(motion: np.ndarray) -> np.ndarray:
    """(T,135) motion -> (T,24,3) world joint positions via SMPL FK (approx rest skeleton)."""
    poses72, trans = B.motion_to_smpl72(motion)
    return FK.synthetic_joints_fn(poses72, trans, np.zeros(10))


def _draw_table(ax):
    hx, hy = F.TABLE_LENGTH_X / 2, F.TABLE_WIDTH_Y / 2
    z = F.TABLE_TOP_Z
    corners = np.array([[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z], [-hx, -hy, z]])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="tab:green", lw=1.0, alpha=0.7)
    ax.plot([0, 0], [-hy, hy], [z, z], color="tab:gray", lw=1.0, alpha=0.6)  # net line at x=0


def animate_comparison(seqs, out_path: str, fps: int = 30, titles=None,
                       elev: int = 12, azim: int = -70, paddle_joint: int = F.LEFT_WRIST):
    """Render side-by-side 3D skeleton animations to a video (mp4 via ffmpeg, else .gif).

    seqs: list of (T, J, 3) position arrays (e.g. [generated, ground_truth]). All must share T.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seqs = [np.asarray(s) for s in seqs]
    T = min(s.shape[0] for s in seqs)
    n = len(seqs)
    titles = titles or [""] * n
    fig = plt.figure(figsize=(5 * n, 5))
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]

    def draw(frame):
        for ax, pos, title in zip(axes, seqs, titles):
            ax.clear()
            p = pos[frame]
            J = p.shape[0]
            for j in range(1, J):
                a, b = j, int(F.SMPL_PARENTS[j])
                ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]],
                        color="tab:blue", lw=2.5)
            ax.scatter(p[paddle_joint, 0], p[paddle_joint, 1], p[paddle_joint, 2],
                       color="tab:red", s=40)
            _draw_table(ax)
            ax.set_xlim(-2.6, 0.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, 2.0)
            ax.set_box_aspect((1, 1, 0.62)); ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title}  (frame {frame})")
        return axes

    anim = animation.FuncAnimation(fig, draw, frames=T, interval=1000.0 / fps)
    if out_path.endswith(".gif") or not animation.writers.is_available("ffmpeg"):
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    return out_path


def plot_positions_montage(positions: np.ndarray, out_path: str, n_frames: int = 6,
                           title: str = "", paddle_joint: int = F.LEFT_WRIST):
    """Montage from precomputed joint positions (T, J, 3) — J may be 22 or 24."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    positions = np.asarray(positions)
    T, J = positions.shape[0], positions.shape[1]
    idx = np.linspace(0, T - 1, min(n_frames, T)).round().astype(int)
    edges = [(j, int(F.SMPL_PARENTS[j])) for j in range(1, J)]
    cols = min(3, len(idx)); rows = int(np.ceil(len(idx) / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for k, t in enumerate(idx):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        p = positions[t]
        for a, b in edges:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]], color="tab:blue", lw=2)
        ax.scatter(p[paddle_joint, 0], p[paddle_joint, 1], p[paddle_joint, 2], color="tab:red", s=30)
        _draw_table(ax)
        ax.set_title(f"frame {t}")
        ax.set_xlim(-2.6, 0.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, 2.0)
        ax.set_box_aspect((1, 1, 0.62)); ax.view_init(elev=15, azim=-70)
    if title:
        fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=90); plt.close(fig)
    return out_path


def plot_skeleton_montage(motion: np.ndarray, out_path: str, n_frames: int = 6, title: str = ""):
    """Save a PNG montage of n_frames evenly sampled across the sequence. Returns out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = motion_to_joint_positions(np.asarray(motion))
    T = pos.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).round().astype(int)
    edges = [(j, int(F.SMPL_PARENTS[j])) for j in range(1, F.SMPL_NUM_JOINTS)]

    cols = min(3, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for k, t in enumerate(idx):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        p = pos[t]
        for a, b in edges:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]], color="tab:blue", lw=2)
        ax.scatter(p[F.LEFT_WRIST, 0], p[F.LEFT_WRIST, 1], p[F.LEFT_WRIST, 2],
                   color="tab:red", s=30, label="paddle hand")
        _draw_table(ax)
        ax.set_title(f"frame {t}")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, 2.0)
        ax.set_box_aspect((1, 1, 0.65))
        ax.view_init(elev=15, azim=-70)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
