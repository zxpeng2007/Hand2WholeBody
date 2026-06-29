# Stage 3 runbook — Hand2Body SMPL → GMR → HoloMotion (Unitree G1)

How to take a generated SMPL clip all the way to a tracked motion on the Unitree G1, using
stock **GMR** (`github.com/YanjieZe/GMR`) and **HoloMotion** (`github.com/HorizonRobotics/HoloMotion`).
Verified against both repos' source (2026-06-29). **GMR/HoloMotion run on Linux/WSL2** (MuJoCo,
ROS2, IsaacLab); Hand2Body generation/export is platform-independent.

## TL;DR pipeline

```
Hand2Body  ──►  SMPL-X .npz  ──GMR──►  G1 .pkl  ──gmr_to_holomotion──►  HoloMotion .npz  ──►  track
 (our exporter)      (GMR ingest)          (qpos)       (FK to 30 bodies, 50fps)   (ref_* keys)    (clip/sim/robot)
```

Key facts that shaped this:
- **GMR does NOT read our `poses (T,72)`.** It reads SMPL-X keys `{root_orient (T,3), pose_body (T,63), betas (16,), trans (T,3), gender, mocap_frame_rate}`. → We emit those directly (`generate_to_smplx_npz`), so **no `smpl_to_smplx.py` pass is needed**.
- **HoloMotion never ingests raw SMPL** — only GMR-retargeted G1 motion.
- **Frame rates**: our 30 fps → GMR keeps 30 → `gmr_to_holomotion` upsamples to **50 fps** → policy 50 Hz, motor PD 500 Hz. Our 30 fps export is correct.
- **Quaternions**: GMR pkl `root_rot` and HoloMotion `ref_global_rotation_quat` are **XYZW**; the live-teleop ZMQ stream is **WXYZ**. (We only feed the offline file path.)

## Step 0 — generate + export (our side, any OS)

```bash
python scripts/generate.py --arch diffusion --checkpoint checkpoints/diffusion.pt \
    --hand my_hand.npy --gmr-out clip_smplx.npz --height 1.75
# clip_smplx.npz has keys: root_orient, pose_body, betas(16), trans, gender, mocap_frame_rate=30
```
`--height` sets `betas[0] = (height-1.66)/0.1`, which is how GMR auto-scales the human to the G1.

## Step 1 — GMR retarget SMPL-X → G1 (.pkl)

One-time GMR setup (Linux): `conda create -n gmr python=3.10 && pip install -e .`; put SMPL-X
models in `assets/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.pkl`.

```bash
# single clip (no re-grounding):
python scripts/smplx_to_robot.py --smplx_file clip_smplx.npz --robot unitree_g1 \
    --save_path clip_g1.pkl --rate_limit
# batch (NOTE: dataset mode applies HEIGHT_ADJUST + ROOT_ORIGIN_OFFSET — drops feet to z=0 and
#  zeroes frame-0 XY, discarding our absolute table placement):
python scripts/smplx_to_robot_dataset.py --src_folder smplx_dir --tgt_folder g1_pkls \
    --robot unitree_g1 --num_cpus 16
```
Output `.pkl` (joblib): `{fps, root_pos (T,3), root_rot (T,4 XYZW), dof_pos (T,29), local_body_pos, link_body_list}`.

> **Table-frame caveat**: if tracking must respect table/ball geometry, use the single-clip
> script (no re-grounding) or post-shift `root_pos`/`ref_global_translation` by the known table
> offset. The tracking policy uses root-relative key-body observations, so absolute XY matters
> for *where the robot stands*, not for tracking fidelity.

## Step 2 — GMR pkl → HoloMotion NPZ

(HoloMotion installs GMR as `thirdparties/GMR/`; run `apply_gmr_motion_retarget_patch.sh` once.)

```bash
python holomotion/src/motion_retargeting/gmr_to_holomotion.py \
    io.robot_config=holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml \
    io.src_dir=g1_pkls io.out_root=holomotion_npz \
    processing.target_fps=50 \
    "preprocess.pipeline=['filename_as_motionkey','legacy_to_ref_keys','slicing','add_padding','tagging']" \
    ray.num_workers=16
```
Output HoloMotion NPZ (float32): `ref_dof_pos [T,29]`, `ref_dof_vel [T,29]`,
`ref_global_translation [T,30,3]`, `ref_global_rotation_quat [T,30,4] (XYZW)`,
`ref_global_velocity [T,30,3]`, `ref_global_angular_velocity [T,30,3]`, `ft_ref_*` (filtered),
`metadata` (JSON: motion_fps=50, num_frames, num_dofs=29, num_bodies=30, …).

## Step 3 — MuJoCo QC (eyeball before the robot)

```bash
MUJOCO_GL=osmesa python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    +key_prefix='robot_' +draw_ref_body_spheres=true +ref_key_prefix='ref_' \
    +motion_npz_root=holomotion_npz skip_frames=6 max_workers=11 +motion_name='all'
# -> video_rendering/*.mp4 ; check joints within limits, feet on ground, no jitter
```

## Step 4 — track the clip

- **Sim2sim (before hardware):** `holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh` (set `ONNX_PATH` = a motion-tracking model, `motion_npz_path`, `robot_xml_path`).
- **On-robot offline clip:** copy the NPZ to `deployment/unitree_g1_ros2_29dof/src/motion_data/`
  (loader asserts `ref_dof_pos.shape[1]==29`, bodies==30), `enable_teleop_reference:false`,
  `launch_holomotion_29dof_docker.sh`, controller: Start → A (velocity) → D-pad select → B (track).

## (Only if training a tracker on our clips)

```bash
python holomotion/src/motion_retargeting/pack_hdf5_v2.py \
    robot=unitree/G1/29dof/29dof_training_isaaclab \
    holomotion_npz_root='["holomotion_npz"]' hdf5_root=h5v2/ours
# then holomotion train_motion_tracking.sh (backend hdf5_v2)
```
A pretrained tracking ONNX likely won't generalize to dynamic table-tennis swings → expect to
fine-tune/retrain on our packed HDF5.

## Reference: G1 29-DoF order (ref_dof_pos columns)

`left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,`
`right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,`
`waist_yaw, waist_roll, waist_pitch,`
`left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,`
`right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw`

## Open items to confirm against the real repos / first run

- `smpl_to_smplx.py` exact argparse (only needed if NOT using our direct SMPL-X export).
- Whether GMR's pelvis `rot_offset` + our world frame (x=length, y=width) need a constant root
  reorientation — **validate visually in the MuJoCo viewer on the first clip**.
- `_schema.json` `pose_aa [T,27,3]`: our GMR pkl has no `pose_aa`; `gmr_to_holomotion` zero-fills
  it and derives `ref_*` from `dof_pos`+root via FK — confirm this is acceptable.
- Contact masks: GMR produces none; derive from foot height/velocity if a tracker needs them.
