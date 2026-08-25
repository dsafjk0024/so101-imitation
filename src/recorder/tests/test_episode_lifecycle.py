"""EpisodeWriter lifecycle: start/stop/discard, episode numbering, and the
integration check that recorder output round-trips through the existing
`converter.convert.convert_all` -- proving recorder bags are actually
consumable by the rest of the pipeline, not just superficially bag-shaped.

No rclpy Node and no live subscriptions here: EpisodeWriter only touches
rosbag2_py directly, the same seam converter/tests/test_convert.py exercises,
so this needs a sourced ROS jazzy for rosbag2_py but nothing else ROS-y.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rosbag2_py = pytest.importorskip(
    "rosbag2_py", reason="source /opt/ros/jazzy/setup.bash before running this test"
)

from rclpy.serialization import serialize_message           # noqa: E402
from sensor_msgs.msg import Image, JointState                # noqa: E402
from std_msgs.msg import Float64MultiArray                   # noqa: E402

from converter.config import Config, FeatureSpec              # noqa: E402
from converter.convert import convert_all                     # noqa: E402
from converter.grid import NS_PER_S, tick_ns                   # noqa: E402

from recorder.episode_writer import EpisodeWriter               # noqa: E402

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

STATE_TOPIC = "/follower/joint_states"
ACTION_TOPIC = "/follower/forward_controller/commands"
WRIST_TOPIC = "/follower/camera/wrist/image_raw"
TOP_TOPIC = "/follower/camera/top/image_raw"

IMG_H, IMG_W = 48, 64

TOPICS = [
    (STATE_TOPIC, "sensor_msgs/msg/JointState"),
    (ACTION_TOPIC, "std_msgs/msg/Float64MultiArray"),
    (WRIST_TOPIC, "sensor_msgs/msg/Image"),
    (TOP_TOPIC, "sensor_msgs/msg/Image"),
]

REPO_ID = "local/test_recorder_pickplace"


def make_test_cfg() -> Config:
    cfg = Config(
        robot_type="so101",
        fps=30,
        task="Pick up the cube and place it in the bin.",
        repo_id=REPO_ID,
        joint_names=JOINTS,
        features=(
            FeatureSpec("observation.state", STATE_TOPIC,
                        "sensor_msgs/msg/JointState", 0.05, names=JOINTS),
            FeatureSpec("action", ACTION_TOPIC,
                        "std_msgs/msg/Float64MultiArray", 0.05, names=JOINTS),
            FeatureSpec("observation.images.wrist", WRIST_TOPIC,
                        "sensor_msgs/msg/Image", 0.1, shape=(IMG_H, IMG_W, 3)),
            FeatureSpec("observation.images.top", TOP_TOPIC,
                        "sensor_msgs/msg/Image", 0.1, shape=(IMG_H, IMG_W, 3)),
        ),
    )
    cfg.validate()
    return cfg


def make_image_msg(k: int, tag: int) -> Image:
    base = np.arange(IMG_H * IMG_W * 3, dtype=np.uint32).reshape(IMG_H, IMG_W, 3)
    arr = ((base + 7 * k + 101 * tag) % 256).astype(np.uint8)
    msg = Image()
    msg.height, msg.width = IMG_H, IMG_W
    msg.encoding = "rgb8"
    msg.step = IMG_W * 3
    msg.data = arr.tobytes()
    return msg


def write_fixture_episode(writer: EpisodeWriter, *, duration_s: float = 1.0) -> Path:
    """Drive `writer` through one full start/write*/stop episode with
    synthetic messages on every configured topic -- same construction as
    converter/tests/test_convert.py's write_fixture_bag (every topic starts
    at ts=0), but through EpisodeWriter instead of a bare SequentialWriter."""
    episode_dir = writer.start()
    end_ns = int(duration_s * NS_PER_S)

    records: list[tuple[str, int, bytes]] = []

    k = 0
    while True:
        ts = tick_ns(0, k, 30)
        if ts > end_ns:
            break
        records.append((WRIST_TOPIC, ts, serialize_message(make_image_msg(k, 0))))
        records.append((TOP_TOPIC, ts, serialize_message(make_image_msg(k, 1))))
        k += 1

    for k in range(10_000):  # joint_states at 100Hz
        ts = k * 10_000_000
        if ts > end_ns:
            break
        msg = JointState()
        msg.name = list(JOINTS)
        msg.position = [0.001 * k + 0.1 * (i + 1) for i in range(6)]
        records.append((STATE_TOPIC, ts, serialize_message(msg)))

    for k in range(10_000):  # commands at 50Hz
        ts = k * 20_000_000
        if ts > end_ns:
            break
        data = [-0.001 * k - 0.1 * (i + 1) for i in range(6)]
        records.append((ACTION_TOPIC, ts, serialize_message(Float64MultiArray(data=data))))

    records.sort(key=lambda r: r[1])
    for topic, ts, raw in records:
        writer.write(topic, raw, ts)

    writer.stop()
    return episode_dir


# --- lifecycle --------------------------------------------------------------


def test_start_then_stop_writes_valid_metadata(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    assert writer.next_episode_index == 0

    episode_dir = write_fixture_episode(writer, duration_s=0.2)

    assert episode_dir.name == "ep_000"
    assert (episode_dir / "metadata.yaml").is_file()
    assert writer.next_episode_index == 1
    assert not writer.is_recording


def test_discard_removes_directory_and_keeps_index(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    episode_dir = writer.start()
    writer.write(STATE_TOPIC, serialize_message(JointState()), 0)
    writer.discard()

    assert not episode_dir.exists()
    assert writer.next_episode_index == 0
    assert not writer.is_recording


def test_episode_index_sequence_across_start_stop_discard(tmp_path):
    root = tmp_path / "bags"
    writer = EpisodeWriter(root, TOPICS)

    ep0 = writer.start()
    writer.stop()  # ep_000 saved, index -> 1
    assert ep0.name == "ep_000"
    assert writer.next_episode_index == 1

    ep1_first = writer.start()
    assert ep1_first.name == "ep_001"
    writer.discard()  # ep_001 removed, index stays 1
    assert writer.next_episode_index == 1
    assert not ep1_first.exists()

    ep1_second = writer.start()
    assert ep1_second.name == "ep_001"
    writer.stop()  # ep_001 saved this time, index -> 2
    assert writer.next_episode_index == 2

    saved = sorted(p.name for p in root.iterdir())
    assert saved == ["ep_000", "ep_001"]


def test_resumes_numbering_from_existing_episode_dirs(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    (root / "ep_000").mkdir()
    (root / "ep_003").mkdir()

    writer = EpisodeWriter(root, TOPICS)
    assert writer.next_episode_index == 4


def test_write_without_start_raises(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    with pytest.raises(RuntimeError, match="not recording"):
        writer.write(STATE_TOPIC, b"\x00", 0)


def test_start_while_recording_raises(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    writer.start()
    with pytest.raises(RuntimeError, match="already recording"):
        writer.start()


def test_stop_without_start_raises(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    with pytest.raises(RuntimeError, match="not recording"):
        writer.stop()


def test_discard_without_start_raises(tmp_path):
    writer = EpisodeWriter(tmp_path / "bags", TOPICS)
    with pytest.raises(RuntimeError, match="not recording"):
        writer.discard()


# --- integration: recorder output feeds the real converter -----------------


def test_recorded_episodes_convert_with_the_real_pipeline(tmp_path):
    root = tmp_path / "bags"
    writer = EpisodeWriter(root, TOPICS)
    write_fixture_episode(writer, duration_s=1.0)
    write_fixture_episode(writer, duration_s=1.0)
    assert writer.next_episode_index == 2

    out = tmp_path / "dataset"
    report = convert_all(root, make_test_cfg(), out, use_videos=False)

    assert report["episodes_converted"] == 2
    assert report["episodes_skipped"] == []
    # 1.0s at 30fps, t0 == 0 (every topic's first sample is at ts=0): 31 ticks.
    assert [e["frames"] for e in report["per_episode"]] == [31, 31]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=REPO_ID, root=out)
    assert ds.num_episodes == 2
    assert ds.num_frames == 62
