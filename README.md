# so101-imitation

An end-to-end imitation learning stack for the low-cost **SO-101** leader–follower arm
pair: hand-guide the leader, record episodes over ROS 2, convert them into a LeRobot
dataset, train an **ACT** or **flow-matching** policy, and run that policy back on the
real arm as a closed loop.

* **Control** — ROS 2 Jazzy + `ros2_control` over Feetech STS3215 servos
* **Data** — rosbag2 (MCAP) → LeRobot dataset v3.0
* **Policies** — ACT (action-chunking transformer) and flow-matching BC, both pure
  PyTorch in one package with one checkpoint format
* **Deployment** — a single synchronous ROS 2 node runs either policy

The design rule throughout is that **the data contract is the interface**. Joint order,
topic names, message types, image sizes and sampling rates live in exactly one file —
`src/converter/config/so101.yaml` — and every stage validates against it. A contract
violation is an error, not a warning: a reordered joint or a resized image is silent
otherwise, and would train into a policy that deploys to nothing.

A full walkthrough, from wiring checks to a trained policy, is in
[`docs/team-guide.html`](docs/team-guide.html) (Korean).

## Pipeline

```
        human hand
            │
   leader arm (STS3215 × 6) ──USB──► ros2_control /leader ──► /leader/joint_states
                                                                    │
                                                        teleop_node (50 Hz, stale guard)
                                                                    │
                                     /follower/forward_controller/commands
                                                                    ▼
   follower arm ◄──USB── ros2_control /follower          2 × camera nodes
                                                                    │
                          episode_recorder_node  ◄── keyboard: s / e / d / q
                                                                    ▼
                                                         bags/ep_000/*.mcap
                                                                    │
                                  converter (offline: needs ROS + LeRobot at once)
                                                                    ▼
                                              datasets/  (LeRobot v3.0, 30 Hz)
                                                                    │
                                            act/tools/train_so101.py  (ACT)
                                            act/tools/train_fbc_so101.py  (flow matching)
                                                                    ▼
                                    sync_inference_node ──► the same command topic
```

The policy publishes to the topic teleoperation used, so it drops into the seat the
human just left — nothing downstream changes.

## Requirements

* Ubuntu 24.04 with **ROS 2 Jazzy** at `/opt/ros/jazzy`, plus `rosdep` and `colcon`
* [Pixi](https://pixi.sh) for the training/conversion layer (no sudo needed)
* SO-101 leader + follower over USB, and two cameras (wrist + third-person)

## 1. Build the ROS 2 layer

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --base-paths src --symlink-install
source scripts/ros_env.sh     # ROS + this workspace, with the LD_LIBRARY_PATH fixes
```

`scripts/ros_env.sh` also forces `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`. Fast-DDS's default
shared-memory transport uses a ~512 KB segment, a single 640×480 rgb8 frame is ~900 KB,
and the mismatch shows up as multi-second stalls on the image topics rather than as an
error.

## 2. Install the training layer

```bash
pixi install -e lerobot
pixi run -e lerobot lerobot-info      # check: lerobot 0.5.1, CUDA visible
```

The two layers are deliberately separate — the ROS side is the system install, the
training side is pinned in `pixi.toml`.

## 3. Motor setup and calibration (once per robot)

Do this **before** any ROS control, and at the robot's final mounting position: it writes
servo IDs and calibration into EEPROM. Motor IDs are permanent; re-assembling the arm
means re-homing.

```bash
pixi run -e lerobot lerobot-find-port
pixi run -e lerobot -- lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
pixi run -e lerobot -- lerobot-setup-motors --teleop.type=so101_leader   --teleop.port=<LEADER_PORT>
pixi run -e lerobot -- lerobot-calibrate    --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
pixi run -e lerobot -- lerobot-calibrate    --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader
```

Calibration lands in `~/.cache/huggingface/lerobot/calibration/` — back it up. An
unchanged robot can skip recalibration on another machine by copying that folder.

## 4. Teleoperation

```bash
ros2 launch bringup teleop.launch.py leader_usb:=<LEADER_PORT> follower_usb:=<FOLLOWER_PORT>
ros2 launch bringup teleop.launch.py hardware_type:=mock          # wiring check, no robot
```

Move the leader by hand and the follower mirrors it. Leader state is on
`/leader/joint_states`; follower commands are a 6-element `Float64MultiArray` on
`/follower/forward_controller/commands`. Individual bring-up lives in
`leader.launch.py`, `follower.launch.py` and `cameras.launch.py`.

## 5. Record episodes

```bash
ros2 run recorder episode_recorder_node
ros2 run recorder keyboard_teleop        # s start · e end+save · d discard · q quit
```

Bags are written raw — CDR bytes are stored untouched, so recording adds no
deserialization cost and cannot silently reinterpret a message. A discarded episode
leaves nothing behind. Each save prints per-topic rates, which is where a dropping
camera gets caught: at collection time, not after training.

## 6. Convert to a LeRobot dataset

The converter is an offline CLI, not a node, and it needs **both** environments at once:
`rosbag2_py` from system ROS and `lerobot` from Pixi.

```bash
source /opt/ros/jazzy/setup.bash                       # puts rosbag2_py on PYTHONPATH
export PYTHONPATH="$PWD/src/converter:$PYTHONPATH"     # append — never replace
pixi run -e lerobot python -m converter.cli --bags bags/ --verbose
```

Exit codes: `0` everything converted, `1` an episode was skipped or nothing converted,
`2` ROS is not on `PYTHONPATH` (it tells you what to source). Episode directory names
must be zero-padded — they are sorted lexically, so `ep_9` would land after `ep_10`.

Frames are built on a fixed 30 Hz grid by **as-of** sampling: each tick takes the newest
sample older than that instant, per feature, and drops the tick if that sample is older
than the feature's `max_age_s`. If too many *interior* ticks are dropped
(`--max-interior-drop-frac`, default 2%), the episode is skipped rather than written —
LeRobot synthesizes time as `frame_index / fps`, so an interior gap would be recorded as
a normal interval and would distort joint velocities.

## 7. Train

```bash
pixi run -e lerobot train-act --data-dir datasets/ --repo-id local/so101_pickplace --out outputs/act
pixi run -e lerobot train-fbc --data-dir datasets/ --repo-id local/so101_pickplace --out outputs/fbc
```

Both write `{"model", "config", "step"}` checkpoints (`step_<N>.pt`) plus a sibling
`stats.json`, so the inference node loads either one the same way. `--init-from`
continues from an existing checkpoint.

## 8. Run the policy on the robot

```bash
ros2 launch bringup teleop.launch.py enable_relay:=false ...   # bring up hardware, no relay

# the node reads the dataset contract from the converter package, so keep it importable
export PYTHONPATH="$PWD/src/converter:$PYTHONPATH"
pixi run -e lerobot python -m inference.sync_inference_node --ros-args \
    -p checkpoint_path:=outputs/act/step_0050000.pt \
    -p control_rate_hz:=30.0 -p n_action_steps:=8
```

Training writes `step_<N>.pt` checkpoints and one `stats.json` per run directory;
`checkpoint_path` points at the `.pt` file, and its `stats.json` sibling is picked up
automatically. The two must stay together — a checkpoint without the normalization
statistics it was trained under produces confident, wrong actions.

The node is synchronous by design: it waits for a complete, fresh observation
(`stale_timeout_s`) instead of extrapolating, executes the action chunk closed-loop, and
denormalizes with the stats the checkpoint was trained under. `flow_steps` applies to a
flow-matching checkpoint and is ignored by ACT.

## Joint contract

`observation.state`, `action` and the URDF all use this order:

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Dataset features (LeRobot v3.0, fps = 30):
`observation.state` (6) · `observation.images.wrist` (480, 640, 3) ·
`observation.images.top` (480, 640, 3) · `action` (6, absolute joint positions).

## Layout

```
so101-imitation/
├── src/
│   ├── bringup/           ros2_control launch + controller config + camera bring-up
│   ├── teleop/            leader → follower mirror relay node
│   ├── recorder/          episode recorder + keyboard control
│   ├── converter/         rosbag → LeRobot v3.0 CLI, and the dataset contract YAML
│   ├── inference/         synchronous ACT / flow-matching inference node
│   └── vendor/            third-party packages, see NOTICE
├── act/                   ACT + flow-matching policies, trainers, tests
├── scripts/               environment setup and camera / dataset inspection helpers
├── docs/team-guide.html   full walkthrough (Korean)
└── pixi.toml              LeRobot 0.5.1 environment
```

## Tests

The unit tests need neither hardware nor a GPU:

```bash
pixi run -e lerobot test-act
pixi run -e lerobot test-converter
pixi run -e lerobot test-inference
pixi run -e lerobot test-recorder
```

The converter's end-to-end test also needs ROS sourced, for `rosbag2_py`.

## Running on two machines

Bags hold raw frames — two uncompressed 480×640 rgb8 streams at 30 fps is roughly
55 MB/s, so a 20-second episode is over 1 GB. Keep the bags on the machine that recorded
them, convert there, and copy only the resulting dataset (video-encoded, far smaller) to
the training machine. `bags/`, `datasets/` and `outputs/` are gitignored.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Third-party code vendored under
`src/vendor/` keeps its own license; see [`NOTICE`](NOTICE).
