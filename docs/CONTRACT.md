# Hand2WholeBody — inter-stage data contract

**Status:** v0.1 (2026-06-29). Items marked 🔒 are confirmed by the user/boss or by
`assets/urdf/`. Items marked ❓ await upstream. Change this file *first* when any
convention changes; code reads its constants from `configs/default.yaml`.

## 0. The three stages

```
 Stage 1                     Stage 2 (THIS REPO)            Stage 3
 table-tennis hand    ──►    Hand2WholeBody         ──►    GMR retarget ──► HoloMotion
 generator (upstream)        12D ➜ whole-body SMPL          (human→G1)       (Unitree G1, 50 Hz)
        │                          │                              │
   12D/frame                  SMPL .npz / stream            G1 .npz / live
```

HoloMotion is the open-source Horizon Robotics framework. It does **not** consume SMPL
at runtime — it consumes GMR-retargeted Unitree **G1 29-DoF** motion. So Stage 2's
job ends at AMASS-style SMPL; GMR does the robot retarget. (HITTER, arXiv 2508.21043,
validates this Stage-3 path end-to-end.)

## 1. World frame 🔒  (from `assets/urdf/table.urdf`)

- Origin: on the **floor** (z = 0), directly **below the table center**.
- **+x** = table length (2.740 m) · **+y** = table width (1.525 m) · **+z** = up · gravity = −z.
- Table top surface at **z = 0.76 m**; net is the plane **x = 0**.
- Single coordinate frame for **all three stages**. Right-handed, gravity-aligned.

## 2. Stage-1 → Stage-2 input: the 12D hand signal 🔒 / ❓

Per frame, one vector for the **LEFT wrist** (= SMPL joint index **20**; the paddle is on
the left, confirmed by `g1_29dof_rev_1_0_pingpong.urdf`):

| slice | name | dim | units | frame |
|------:|------|----:|-------|-------|
| `0:3`  | position      | 3 | meters    | world (§1) |
| `3:6`  | linear velocity | 3 | meters/sec | world |
| `6:12` | rotation 6D   | 6 | unitless  | **GLOBAL** orientation of the wrist in world frame |

- 🔒 **The wrist orientation is GLOBAL, not SMPL-local.** It is the world-frame
  orientation of the wrist link, *not* `body_pose[20]` (the parent-relative joint
  rotation). Forehand vs. backhand is decided **by this global orientation** — so it
  must be preserved, never yaw-canonicalized away.
- 🔒 Velocity is in **m/s** (frame-rate independent), central finite difference.
- 🔒 **6D basis convention = Zhou et al. 2019 COLUMNS** (`R6D_COLUMN`: first two columns
  of R), confirmed 2026-06-29. Single source of truth: `h2wb.representations.frames.PROJECT_R6D`
  (mirrored in `configs/default.yaml`). The whole project — input extraction, decode,
  internal body rotations, and export — uses this one convention.
- 🔒 The 6D encodes the **wrist JOINT** orientation (SMPL `left_wrist`, joint 20) —
  confirmed 2026-06-29. No wrist→paddle offset is applied on the human side. (On the
  robot the paddle is rigidly mounted to `left_wrist_yaw_link`; that offset is handled
  downstream by GMR/retarget, not here.)

## 3. Stage-2 → Stage-3 output: whole-body SMPL 🔒

Plain **SMPL** (rigid wrist is enough — no SMPL-X fingers). AMASS-style `.npz`:

| key | shape | meaning |
|-----|-------|---------|
| `poses` | (T, 72) | axis-angle; `[0:3]`=global_orient (pelvis, world), `[3:72]`=23 body joints (parent-relative) |
| `trans` | (T, 3)  | pelvis translation, world frame (§1), meters |
| `betas` | (10,)   | body shape (fixed per subject) |
| `gender`| str     | "neutral" unless specified |
| `mocap_frame_rate` | scalar | **30** |
| `contacts` | (T, 4) | *optional* foot-contact flags (ankles+feet) to aid GMR |

- Internally the model predicts rotations in **6D**; convert to axis-angle only at export.
- Root trajectory (`trans` + `poses[:, :3]`) must be **physically consistent** (plausible
  support, minimal foot skating) — HoloMotion tracks root velocity, root height,
  projected gravity, and root-relative key bodies.

**GMR ingest (verified 2026-06-29):** GMR does NOT read `poses (T,72)`. It reads SMPL-X keys
`{root_orient (T,3), pose_body (T,63), betas (16,), trans (T,3), gender, mocap_frame_rate}`.
We emit those directly via `h2wb.inference.generate_to_smplx_npz` (preferred zero-glue path) —
no `smpl_to_smplx.py` pass. `betas[0]` sets the auto-scale (height = 1.66 + 0.1·betas[0]).
Then: GMR `smplx_to_robot` → G1 `.pkl` → `gmr_to_holomotion` → HoloMotion NPZ (50 fps). Full
commands + key schemas + G1 29-DoF order in [stage3_runbook.md](stage3_runbook.md).

## 4. Temporal contract 🔒

- Generation / training canonical rate: **30 Hz** (matches GVHMR/HITTER, AMASS resampled).
- Interpolate **30 → 50 Hz only at the Stage-3 boundary** (HoloMotion control loop).
- Stage 2 must be **causal / streaming** (real-time). Condition on a window of the last
  `K` hand frames + last few generated body frames; emit the next `P` frames. Frame
  count `P` is tunable.

## 5. Canonicalization (network I/O) 🔒 design decision

- **Position:** express relative to a per-window anchor (first-frame pelvis or the fixed
  play-area origin) for translation robustness — `canonicalize_hand12(mode="root_relative_pos")`.
  Do **not** expect absolute world translation to be recoverable from a single hand; use a
  root anchor / dedicated root head.
- **Orientation:** keep in the **world frame** (do **not** remove yaw) — see §2.
- The removed anchor is stored and re-applied to map predictions back to world.

## 6. Open items (upstream)

1. ✅ **RESOLVED 2026-06-29** — 6D basis = Zhou-2019 **columns** (`R6D_COLUMN`).
2. ✅ **RESOLVED 2026-06-29** — 6D encodes the **wrist joint** (SMPL `left_wrist`), no offset.
3. ❓ Any extra anchor available besides the single hand (root/origin, or a strike-phase
   backswing/strike/recovery signal) — would sharply cut single-hand ambiguity.
4. The SMPL build being updated upstream — which package/version, num_betas, and
   whether it returns global joint orientations or only joint positions (we compute global
   orientations by walking `SMPL_PARENTS` either way).
5. Amount of paired (12D ↔ SMPL) table-tennis data available, if any.
