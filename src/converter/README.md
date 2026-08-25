# converter

Offline converter: recorded rosbag episodes -> one LeRobot v3.0 dataset that
`lerobot-train` can consume directly.

Not a ROS node. It is a CLI that needs **both** environments at once:
`rosbag2_py` from system ROS jazzy, and `lerobot==0.5.1` from the pixi env.
Both run on Python 3.12, so sourcing ROS puts `rosbag2_py` on `PYTHONPATH` and
the pixi environment inherits it.

## Run

```bash
source /opt/ros/jazzy/setup.bash          # required: puts rosbag2_py on PYTHONPATH
export PATH="$HOME/.pixi/bin:$PATH"

# Append to PYTHONPATH, never replace it: overwriting drops the ROS entry and
# rosbag2_py disappears.
export PYTHONPATH="$PWD/src/converter:$PYTHONPATH"

pixi run -e lerobot python -m converter.cli --bags bags/ --verbose
```

Without ROS sourced the CLI exits with status 2 and tells you what to source.

Exit codes: `0` all episodes converted, `1` some episode was skipped or nothing
was converted, `2` ROS is not on `PYTHONPATH`.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--bags` | (required) | directory holding one subdirectory per episode bag |
| `--config` | `src/converter/config/so101.yaml` | conversion config |
| `--repo-id` | config value (`local/so101_pickplace`) | dataset repo id |
| `--root` | `datasets/<repo_id>` | dataset output directory |
| `--overwrite` | off | replace an existing dataset |
| `--no-videos` | off | store images as files instead of encoded video |
| `--max-interior-drop-frac` | `0.02` | skip an episode above this interior drop fraction |
| `--limit` | none | convert only the first N episodes |
| `--verbose` | off | per-episode and per-feature sync statistics |

## Contract

`config/so101.yaml` is the single source of truth. The recorder must publish
exactly these topics and types:

| feature | topic | type |
|---|---|---|
| `observation.state` | `/follower/joint_states` | `sensor_msgs/msg/JointState` |
| `action` | `/follower/forward_controller/commands` | `std_msgs/msg/Float64MultiArray` |
| `observation.images.wrist` | `/follower/camera/wrist/image_raw` | `sensor_msgs/msg/Image` (`rgb8`) |
| `observation.images.top` | `/follower/camera/top/image_raw` | `sensor_msgs/msg/Image` (`rgb8`) |

Joint order is canonical everywhere: `shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper`. A topic missing from the bag, a `msg_type` that
disagrees with the bag, joint names out of canonical order, an image encoding
other than `rgb8`, or an image whose size differs from the configured shape are
all errors rather than warnings — that is what catches recorder drift.

Episode directories must be zero-padded (`ep_000`, `ep_001`, ...): they are
ordered lexicographically, so `ep_9` would sort after `ep_10`.

## How frames are built

Messages are resampled onto a fixed 30Hz grid. At each tick every feature is
sampled *as of* that instant: the newest sample at or before the tick, and only
if it is within that feature's `max_age_s`. Sampling at or before the tick is
what keeps a frame from carrying data the robot had not yet observed.

The grid is fixed rather than driven by one reference topic because lerobot
0.5.1 synthesises `timestamp = frame_index / fps` and discards real bag times.
Uneven frame spacing would therefore be recorded as uniform and distort the
joint velocities a policy learns. For the same reason an episode with too many
*interior* dropped ticks is skipped rather than silently compacted; drops before
the first or after the last emitted tick just trim the episode and are fine.

## Tests

```bash
source /opt/ros/jazzy/setup.bash
export PATH="$HOME/.pixi/bin:$PATH"

pixi run -e lerobot test-converter          # all, including one real AV1 encode
```

To skip the AV1 encode:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run -e lerobot \
  python -m pytest src/converter/tests -m "not slow"
```

Plugin autoload is off because the ROS `PYTHONPATH` advertises ROS's own pytest
plugin (`launch_testing`), whose `lark` dependency is not in this env.

`tests/` (plural) holds these pytest suites, and `pytest.ini` there puts the
package on `sys.path` without touching `PYTHONPATH`. `test/` (singular) is the
ament lint boilerplate that ships with an `ament_python` package and is run by
colcon, not by pixi.

## Layout

| file | responsibility |
|---|---|
| `config/so101.yaml` | the contract: joint order, topics, shapes, fps, `max_age_s` |
| `converter/decoders.py` | ROS message -> numpy. Duck-typed, imports no ROS |
| `converter/config.py` | YAML -> `Config` + the v3.0 feature schema. Imports no ROS |
| `converter/grid.py` | as-of buffers, tick arithmetic, drop accounting. Imports no ROS or lerobot |
| `converter/convert.py` | the only module importing ROS and lerobot |
| `converter/cli.py` | argparse entry point |
