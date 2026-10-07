# sim2real — StackCube from Isaac Lab to the real SO-101

The rest of this repository learns from real teleoperation only. This directory
takes the other route: it trains in simulation and transfers to the real arm.
It follows the ResiP recipe:

1. A **state-based teacher** is trained in Isaac Lab, with privileged cube poses,
   as behaviour cloning plus residual RL.
2. The teacher's successful rollouts are **rendered with domain randomization**
   into image demonstrations.
3. An **image + joint student** (Diffusion Policy) is co-trained on those
   synthetic demos plus a few dozen real teleoperated demos, and deployed on the
   real arm.

The task is StackCube: pick the 2.5 cm cube and place it on the 4 cm cube.

Most of the work lies between steps 2 and 3. The simulated and real rigs have to
agree before synthetic data means anything: the joint readings, the fingertip
position, the camera intrinsics and mounts, cube colors, exposure and tracking
speed. The calibration tools here exist for that.

```
 leader arm ──► Isaac Lab StackCube (state, 36-D)          scripts/teleop_task.py
                        │  250 sim teleop episodes
                        ▼
              DBC base policy (JAX)                        main.py --agent=agents/dbc.py
                        │
              ResiP residual PPO, γ = 0.99                 online.py --agent=agents/resip.py
                        │  teacher: 91.6 % sim success
                        ▼
              PyTorch port of the teacher                  scripts/export_resip_teacher.py
                        │
   rendering DR ──► 600 synthetic image demos              scripts/generate_teacher_demos.py
                        │
   real rig ──► 40 real demos (calibrated mapping)         scripts/record_real_demos.py
                        │
                        ▼
              Diffusion Policy co-training                 scripts/train_dp.py
              (sim 0.93 / real 0.07 per batch)
                        │
                        ▼
              real deployment, 30 Hz                       scripts/deploy_policy_real.py
```

Results are in [`docs/results.md`](docs/results.md). In short, the co-trained DP
stacks on the real arm in about half of the attempts; failures are mostly missed
grasps. Environment, teleoperation and teacher-training notes are in
[`docs/state-policy.md`](docs/state-policy.md).

## Environments

Three Python environments. JAX's CUDA wheels clash with the torch CUDA libraries that
the student and deployment code use, so the teacher is exported once to a PyTorch port:

| purpose | contents | used by |
|---|---|---|
| simulation | Isaac Sim 5.1, Isaac Lab 0.54.2, torch 2.7 (cu128), `pip install -e .` | every `--headless` script, `generate_teacher_demos.py`, `eval_*_sim.py` |
| real robot + students | torch 2.7, `pip install -e ".[real]"` (LeRobot 0.4.4 with Feetech) | calibration capture, demo recording, `train_dp.py`, deployment |
| teacher training | Isaac Lab + JAX/Flax, `pip install -e ".[learning]"` (rollouts and evaluation run in Isaac) | `main.py`, `online.py`; `export_resip_teacher.py` needs only JAX |

We ran the real-robot and student steps in one env with Isaac Lab also installed,
so scripts that need both FK and hardware (`touch_calibration.py --fit`) run there.
Run commands from `sim2real/`. Scripts that launch Isaac take `--headless` and are
run as `python -u`.

The motors are set up and calibrated with LeRobot as in the top-level README §3.
The follower's LeRobot calibration (`~/.cache/huggingface/lerobot/...`) is what
`calibration/joint_mapping/follower.yaml` is fitted against, and the loader
refuses a mapping whose recorded ranges no longer match it.

## 1. Teacher (simulation only)

Commands for sim teleoperation, DBC pre-training and ResiP fine-tuning are in
`docs/state-policy.md`. The teacher used below is ResiP with γ = 0.99, fine-tuned
for 2000 iterations from a DBC base (sweep: `sweep.sh`). Export it once, from the
JAX env:

```bash
python scripts/export_resip_teacher.py --run exp/sweep/gamma99/so101-StackCube-v0/resip/<run> --epoch 2000
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_resip_teacher.py   # torch port == Flax
python -u scripts/eval_resip_teacher.py --headless --num-envs 256
```

The teacher was trained with the robot root at the origin, while the calibrated
scene places it where the real base is. `so101.tasks.teacher.teacher_observation`
converts positions to the robot frame; without that conversion the success rate
drops to 0 %.

## 2. Calibrate the real rig

Order matters: later steps assume earlier ones. All positions on the table are in
the "yellow-sphere" frame (`scripts/show_robot_base_frame.py`): origin at the table's
rear edge on the robot centre line, +X into the table, +Y to the robot's left.

| step | what | how | our result |
|---|---|---|---|
| 2.1 | camera intrinsics | `calibrate_intrinsics.py --camera {front,wrist}` with the A4 ChArUco board (8×5, 32 mm; `make_calibration_targets.py`). Gates: RMS < 0.3 px, full 3×3 coverage, ≥ 5 tilted shots | front 0.246 px / 41 shots, wrist 0.267 px / 48 shots |
| 2.2 | front camera pose | board taped at a measured centre, then known-centre PnP: `calibrate_handeye.py --camera front --solve --via-board --known-board-center-env X Y Z --known-board-yaw-deg -90` | 0.39 px |
| 2.3 | follower scale | `fit_follower_joint_mapping.py`: *physical* (4096 ticks/rev) scale instead of USD-range scaling | board scatter 40.8 → 17.0 mm |
| 2.4 | follower offsets | `touch_calibration.py --record`, then `--fit --write`: the fingertips touch 9 marked table points (X 18/25/32 cm × Y −14/0/14 cm) | grasp-point error 18.2 → 5.5 mm |
| 2.5 | `wrist_roll` offset | `roll_alignment.py --record`, then `--fit --write`: fingertip line parallel to the table edge, 3 poses | +8.0° |
| 2.6 | wrist camera mount | `calibrate_handeye.py --camera wrist --capture --leader-teleop`, then `known_board_calibration.py --fit --write` (board at a known pose) | 22 views, 32 px (see limitations) |
| 2.7 | cube colors | `match_cube_colors.py --headless --small R G B --large R G B` fits sim diffuse colors to real front-camera pixels | |
| 2.8 | tracking | `sysid_joint_tracking.py --real / --sim / --compare` | real lags sim by 1–3 ticks; servo P gain raised 16 → 32 |
| 2.9 | end-to-end check | `fixed_layout_check.py --plan / --replay / --render`: replay a teacher plan open loop on the real arm, then render the sim at the real joints | |

The scene reads `calibration/cameras/{front,wrist}.yaml` directly, so changing a
camera pose changes the rendered data. Real frames are rectified onto the ideal
pinhole `K_virtual` that Isaac renders (`calibration/cameras/README.md`). The
follower mapping (`calibration/joint_mapping/follower.yaml`) is used by every real
script. The files in `calibration/` are our rig's values; recalibrate on yours.

## 3. Synthetic demonstrations

```bash
python -u scripts/generate_teacher_demos.py --headless --num-envs 8 --num-episodes 600 \
    --motion-substeps 2 --post-success-steps 40 --stable-success-steps 15 \
    --wrist-pos-jitter-m 0.01 --wrist-rot-jitter-deg 3 --out outputs/synthetic/resip_v3_slow2
python -u scripts/preview_render_randomization.py --headless     # inspect the DR draws
```

- **Rendering DR** (`so101.tasks.render_randomization`) varies:
  - key and dome lights: direction, intensity ×0.4–1.6, tint
  - cube HSV and roughness
  - table brightness and tint
  - backdrop walls
  - front camera pose (±1 cm / ±1.5°) and wrist camera pose (±1 cm / ±3° here)

  Textures and physics are not randomized.
- **`--motion-substeps 2`** interpolates each teacher target over two ticks. Joint
  speed drops about 2.4×, which brings the teacher's per-tick steps (≤ 0.10 rad)
  well within the real arm's rate limit.
- **Saved episodes**: only successful episodes are saved, up to 40 ticks after the
  first success, and only if the last 15 frames are still successes. Images are
  240×320.
- `scripts/run_cotrain_v3.sh` chains generation, validation, DP/ACT training and
  sim evaluation.

## 4. Real demonstrations

```bash
python scripts/record_real_demos.py --out outputs/real_demos/v2
```

Keys:
- `r`: both arms move to the start pose; place the cubes.
- `s`: release the leader and record.
- `t`: save as a success.
- `b`: discard.

Only successes are stored, in the same `trajectory_*.pkl` format as the synthetic
demos, with raw readings so they can be remapped (`remap_real_episodes.py`) if the
joint mapping changes later. We recorded 40 demos.

## 5. Train and evaluate the student

```bash
python -u scripts/train_dp.py --data sim=outputs/synthetic/resip_v3_slow2 --data real=outputs/real_demos/v2 \
    --out outputs/dp_train/cotrain --steps 100000 --batch-size 128 --amp --num-workers 6 --save-freq 10000
python -u scripts/eval_dp_sim.py --headless --checkpoint outputs/dp_train/cotrain/dp_so101.pt \
    --num-envs 4 --num-rounds 10 [--render-randomization]
```

The student, `so101.learning.dp`, is the image Diffusion Policy from
robust-rearrangement:
- ResNet18 (GroupNorm) per camera, UNet, obs 1 / pred 32 / action 8
- DDPM training, DDIM inference with warm start

On an 11 GB GPU, batch 128 with AMP takes about 0.36 s/step. `--episode-selection`
and `--source-fractions` fix the episode subset and the sim/real sampling mix;
`scripts/run_dp_count_ablation.py` uses them for the data-count ablation.

An ACT student is also available (`train_act_cotrain.py`, `eval_act_sim.py`) but
performed worse on the real arm.

## 6. Deploy

```bash
python scripts/deploy_policy_real.py --policy dp --checkpoint <ckpt> --dry-run      # infer only
python scripts/deploy_policy_real.py --policy dp --checkpoint <ckpt> --episodes 10 \
    --record outputs/real_eval/<name>
```

- **Each tick**: rectified RGB and follower joints → policy → clamp to joint limits
  → rate limit (0.25 rad/tick arm, 0.5 gripper) → follower.
- **Episode start**: an eased move to the simulated start pose and an exposure
  re-meter.
- **Keys** (no Enter): `s` start, `y`/`n` label success or failure, `r` abandon
  the attempt (logged as interrupted, not rated), `q` quit.
- **Records**: every attempt is saved, along with a per-session JSONL and a
  summary (success rate, policy time, loop overruns).
- DDIM uses 4 steps on deployment (about 31 ms per plan).

## Known limitations

- **Wrist camera mount**: the fit residual is 32 px, far above the 3 px alignment
  target. The data was generated with it anyway, so it is the first thing to
  revisit.
- **Simulated actuators**: stiffness and damping were not tuned to the measured
  lag; only the real servo gain was changed.
- **Real evaluations are small**: 10–23 attempts per checkpoint, labelled by the
  operator, without randomized condition order. Treat differences between
  checkpoints as screening, not significance.
- **Leader mapping**: during teleoperation the leader uses a linear (USD-range)
  mapping while the follower uses the calibrated physical mapping.
- **Control rate**: the student re-plans every 8 ticks, and that planning tick
  overruns 33 ms, so the effective control rate is about 27 Hz.

## Layout

```
sim2real/
├── src/so101/
│   ├── assets/          SO-101 USD (right-mounted wrist camera) + ArticulationCfg
│   ├── configs/         env configs and Gym registry
│   ├── scenes/ tasks/   calibrated tabletop scene, StackCube env, rendering DR, teacher wrapper
│   ├── real/            LeRobot bridge, joint mapping, follower safety limits, cameras, recording
│   ├── learning/        dp/ (Diffusion Policy), act/, resip_teacher.py (PyTorch teacher)
│   └── camera_calibration.py
├── agents/ utils/ main.py online.py inference.py   JAX: FBC, DBC, DPPO, ResiP
├── scripts/             calibration, data generation, training, evaluation, deployment
├── calibration/         boards, camera YAMLs, follower mapping, hand-eye pose sets
├── tests/               PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
└── docs/                results.md, state-policy.md
```

## Credits

- **Seongsu Kim**: Isaac Lab StackCube environment, sim teleoperation, the
  FBC/DBC/DPPO/ResiP agents (`agents/`, `utils/`, `main.py`, `online.py`,
  `inference.py`), and the original package.
- **Jusung Kim**: camera and kinematic calibration, synthetic data, students,
  deployment and evaluation.
- The SO-101 USD and LeRobot helpers are adapted from the Sim-to-Real SO-101
  Workshop (Apache-2.0); those files keep their SPDX headers.
- The ResiP and image-DP ports follow robust-rearrangement (ResiP, Ankile et al.);
  DPPO follows the reference DPPO implementation.

This directory is MIT-licensed (`LICENSE`); the rest of the repository is Apache-2.0.
