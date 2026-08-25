"""Seam 3 -- fixture rosbag through the converter into a LeRobot v3.0 dataset.

Needs both environments: rosbag2_py from a sourced ROS jazzy and lerobot from
the pixi env. Fixture images are 64x48 to keep the AV1 encode cheap; the real
480x640 schema is covered by Seam 2.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

rosbag2_py = pytest.importorskip(
    "rosbag2_py",
    reason="source /opt/ros/jazzy/setup.bash before running Seam 3",
)

from rclpy.serialization import serialize_message          # noqa: E402
from sensor_msgs.msg import Image, JointState              # noqa: E402
from std_msgs.msg import Float64MultiArray                 # noqa: E402

from converter.config import Config, FeatureSpec           # noqa: E402
from converter.convert import (                            # noqa: E402
    convert_all,
    find_episode_dirs,
    read_bag_messages,
)
from converter.grid import NS_PER_S, tick_ns               # noqa: E402

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

# JointState.name order in the fixture is deliberately NOT canonical, so the
# test proves the decoder reorders by name instead of trusting index order.
SCRAMBLED = ("gripper", "elbow_flex", "shoulder_pan", "wrist_roll",
             "shoulder_lift", "wrist_flex")

REPO_ID = "local/test_pickplace"


def make_test_cfg(image_shape: tuple[int, int, int] = (IMG_H, IMG_W, 3)) -> Config:
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
                        "sensor_msgs/msg/Image", 0.1, shape=image_shape),
            FeatureSpec("observation.images.top", TOP_TOPIC,
                        "sensor_msgs/msg/Image", 0.1, shape=image_shape),
        ),
    )
    cfg.validate()
    return cfg


# --- fixture bag builder --------------------------------------------------


def state_positions(k: int) -> list[float]:
    """Canonical-order joint values for joint sample k."""
    return [round(0.001 * k + 0.1 * (i + 1), 6) for i in range(6)]


def action_values(k: int) -> list[float]:
    return [round(-0.001 * k - 0.1 * (i + 1), 6) for i in range(6)]


def camera_frame(tag: int, k: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Deterministic per-(camera, frame) pattern the assertions can recompute."""
    base = np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3)
    return ((base + 7 * k + 101 * tag) % 256).astype(np.uint8)


def make_image_msg(arr: np.ndarray) -> Image:
    msg = Image()
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.step = int(arr.shape[1]) * 3
    msg.data = arr.tobytes()
    return msg


def write_fixture_bag(
    bag_dir: Path,
    *,
    duration_s: float = 2.0,
    wrist_gap_ns: tuple[int, int] | None = None,
    drop_top_after_ns: int | None = None,
    image_shape: tuple[int, int] = (IMG_H, IMG_W),
) -> None:
    """Write a bag where every topic starts at ts=0 and ends at ts=duration_s,
    so t0 == 0 and the tick count is exactly 61 for 2s at 30fps."""
    end_ns = int(duration_s * NS_PER_S)
    h, w = image_shape

    records: list[tuple[str, int, bytes]] = []

    k = 0
    while True:
        ts = tick_ns(0, k, 30)
        if ts > end_ns:
            break
        if not (wrist_gap_ns and wrist_gap_ns[0] < ts < wrist_gap_ns[1]):
            records.append(
                (WRIST_TOPIC, ts, serialize_message(make_image_msg(camera_frame(0, k, h, w))))
            )
        if drop_top_after_ns is None or ts <= drop_top_after_ns:
            records.append(
                (TOP_TOPIC, ts, serialize_message(make_image_msg(camera_frame(1, k, h, w))))
            )
        k += 1

    for k in range(10_000):  # joint_states at 100Hz
        ts = k * 10_000_000
        if ts > end_ns:
            break
        msg = JointState()
        msg.name = list(SCRAMBLED)
        by_name = dict(zip(JOINTS, state_positions(k)))
        msg.position = [by_name[n] for n in SCRAMBLED]
        records.append((STATE_TOPIC, ts, serialize_message(msg)))

    for k in range(10_000):  # commands at 50Hz
        ts = k * 20_000_000
        if ts > end_ns:
            break
        records.append(
            (ACTION_TOPIC, ts,
             serialize_message(Float64MultiArray(data=action_values(k))))
        )

    records.sort(key=lambda r: r[1])

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    for i, (topic, msg_type) in enumerate(
        [
            (STATE_TOPIC, "sensor_msgs/msg/JointState"),
            (ACTION_TOPIC, "std_msgs/msg/Float64MultiArray"),
            (WRIST_TOPIC, "sensor_msgs/msg/Image"),
            (TOP_TOPIC, "sensor_msgs/msg/Image"),
        ]
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=i, name=topic, type=msg_type, serialization_format="cdr"
            )
        )
    for topic, ts, raw in records:
        writer.write(topic, raw, ts)
    del writer  # closes the bag and writes metadata.yaml


@pytest.fixture
def bags_root(tmp_path) -> Path:
    root = tmp_path / "bags"
    root.mkdir()
    write_fixture_bag(root / "ep_000")
    return root


def load_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=REPO_ID, root=root)


# --- episode discovery ----------------------------------------------------


def test_find_episode_dirs_is_sorted_and_skips_non_bags(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    for name in ("ep_002", "ep_000", "ep_001"):
        write_fixture_bag(root / name, duration_s=0.2)
    (root / "not_a_bag").mkdir()
    (root / "loose_file.txt").write_text("x", encoding="utf-8")

    dirs = find_episode_dirs(root)
    assert [d.name for d in dirs] == ["ep_000", "ep_001", "ep_002"]


# --- bag reading + contract enforcement -----------------------------------


def test_read_bag_messages_yields_ascending_configured_topics(bags_root):
    cfg = make_test_cfg()
    messages = list(read_bag_messages(bags_root / "ep_000", cfg))
    topics = {m[0] for m in messages}
    assert topics == {STATE_TOPIC, ACTION_TOPIC, WRIST_TOPIC, TOP_TOPIC}
    stamps = [m[1] for m in messages]
    assert stamps == sorted(stamps)


def test_missing_topic_in_bag_raises(bags_root):
    cfg = make_test_cfg()
    bad = dataclasses.replace(
        cfg,
        features=(dataclasses.replace(cfg.features[0], topic="/nope"),) + cfg.features[1:],
    )
    with pytest.raises(ValueError, match="not found in bag"):
        list(read_bag_messages(bags_root / "ep_000", bad))


def test_msg_type_mismatch_raises(bags_root):
    cfg = make_test_cfg()
    bad = dataclasses.replace(
        cfg,
        features=(
            dataclasses.replace(
                cfg.features[0], msg_type="std_msgs/msg/Float64MultiArray"
            ),
        )
        + cfg.features[1:],
    )
    with pytest.raises(ValueError, match="type mismatch"):
        list(read_bag_messages(bags_root / "ep_000", bad))


# --- full conversion ------------------------------------------------------


def test_converted_dataset_matches_the_contract(bags_root, tmp_path):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    out = tmp_path / "dataset"
    report = convert_all(bags_root, make_test_cfg(), out, use_videos=False)

    assert report["episodes_converted"] == 1
    assert report["episodes_skipped"] == []

    ds = load_dataset(out)
    assert ds.meta.fps == 30
    assert ds.meta.robot_type == "so101"
    assert CODEBASE_VERSION == "v3.0"
    assert ds.num_episodes == 1
    assert ds.num_frames == 61

    feats = ds.meta.features
    assert tuple(feats["observation.state"]["shape"]) == (6,)
    assert tuple(feats["action"]["shape"]) == (6,)
    assert feats["observation.state"]["names"] == [j + ".pos" for j in JOINTS]
    for key in ("observation.images.wrist", "observation.images.top"):
        assert tuple(feats[key]["shape"]) == (IMG_H, IMG_W, 3)


def test_state_and_action_values_are_reordered_and_aligned(bags_root, tmp_path):
    out = tmp_path / "dataset"
    convert_all(bags_root, make_test_cfg(), out, use_videos=False)
    ds = load_dataset(out)

    # tick 0 sits exactly on joint sample 0 and command sample 0
    frame0 = ds[0]
    np.testing.assert_allclose(
        np.asarray(frame0["observation.state"], dtype=np.float32),
        np.asarray(state_positions(0), dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(frame0["action"], dtype=np.float32),
        np.asarray(action_values(0), dtype=np.float32),
        atol=1e-5,
    )

    # tick 3 = 100_000_000ns -> joint sample 10, command sample 5
    frame3 = ds[3]
    np.testing.assert_allclose(
        np.asarray(frame3["observation.state"], dtype=np.float32),
        np.asarray(state_positions(10), dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(frame3["action"], dtype=np.float32),
        np.asarray(action_values(5), dtype=np.float32),
        atol=1e-5,
    )


def test_image_pixels_survive_the_lossless_path(bags_root, tmp_path):
    out = tmp_path / "dataset"
    convert_all(bags_root, make_test_cfg(), out, use_videos=False)
    ds = load_dataset(out)

    wrist = np.asarray(ds[0]["observation.images.wrist"])
    if wrist.ndim == 3 and wrist.shape[0] == 3:  # CHW tensor
        wrist = np.transpose(wrist, (1, 2, 0))
    if wrist.dtype != np.uint8:
        wrist = np.round(wrist * 255).astype(np.uint8)
    np.testing.assert_array_equal(wrist, camera_frame(0, 0))


def test_interior_camera_gap_skips_the_episode(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    write_fixture_bag(root / "ep_000", wrist_gap_ns=(700_000_000, 1_000_000_000))
    out = tmp_path / "dataset"

    report = convert_all(root, make_test_cfg(), out, use_videos=False)
    assert report["episodes_converted"] == 0
    assert [name for name, _ in report["episodes_skipped"]] == ["ep_000"]
    assert "interior" in report["episodes_skipped"][0][1]

    # A dataset with no episodes cannot be opened, so check the metadata lerobot
    # wrote rather than round-tripping through LeRobotDataset.
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 0
    assert info["total_frames"] == 0


def test_good_episode_still_converts_when_a_sibling_is_skipped(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    write_fixture_bag(root / "ep_000", wrist_gap_ns=(700_000_000, 1_000_000_000))
    write_fixture_bag(root / "ep_001")
    out = tmp_path / "dataset"

    report = convert_all(root, make_test_cfg(), out, use_videos=False)
    assert report["episodes_converted"] == 1
    assert [name for name, _ in report["episodes_skipped"]] == ["ep_000"]

    ds = load_dataset(out)
    assert ds.num_episodes == 1
    assert ds.num_frames == 61


def test_trailing_camera_loss_trims_instead_of_skipping(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    write_fixture_bag(root / "ep_000", drop_top_after_ns=1_500_000_000)
    out = tmp_path / "dataset"

    report = convert_all(root, make_test_cfg(), out, use_videos=False)
    assert report["episodes_converted"] == 1
    ds = load_dataset(out)
    assert 0 < ds.num_frames < 61


def test_existing_output_refuses_without_overwrite(bags_root, tmp_path):
    out = tmp_path / "dataset"
    convert_all(bags_root, make_test_cfg(), out, use_videos=False)
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_all(bags_root, make_test_cfg(), out, use_videos=False)


def test_overwrite_replaces_the_dataset(bags_root, tmp_path):
    out = tmp_path / "dataset"
    convert_all(bags_root, make_test_cfg(), out, use_videos=False)
    report = convert_all(
        bags_root, make_test_cfg(), out, use_videos=False, overwrite=True
    )
    assert report["episodes_converted"] == 1
    assert load_dataset(out).num_episodes == 1


def test_limit_converts_only_the_first_n_episodes(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    for name in ("ep_000", "ep_001", "ep_002"):
        write_fixture_bag(root / name, duration_s=0.5)
    out = tmp_path / "dataset"

    report = convert_all(root, make_test_cfg(), out, use_videos=False, limit=2)
    assert report["episodes_converted"] == 2
    assert load_dataset(out).num_episodes == 2


def test_empty_bags_root_raises(tmp_path):
    root = tmp_path / "bags"
    root.mkdir()
    with pytest.raises(ValueError, match="no episode"):
        convert_all(root, make_test_cfg(), tmp_path / "dataset", use_videos=False)


def test_per_episode_report_carries_sync_statistics(bags_root, tmp_path):
    out = tmp_path / "dataset"
    report = convert_all(bags_root, make_test_cfg(), out, use_videos=False)
    entry = report["per_episode"][0]
    assert entry["name"] == "ep_000"
    assert entry["frames"] == 61
    assert set(entry["features"]) == {
        "observation.state",
        "action",
        "observation.images.wrist",
        "observation.images.top",
    }
    assert entry["features"]["observation.images.wrist"]["match_rate"] == 1.0


@pytest.mark.slow
def test_video_path_produces_encoded_files(bags_root, tmp_path):
    """use_videos=True runs a real AV1 encode, so this only checks that the
    pipeline completes and video files land. AV1 is lossy -- no pixel asserts."""
    out = tmp_path / "dataset"
    report = convert_all(bags_root, make_test_cfg(), out, use_videos=True)
    assert report["episodes_converted"] == 1
    videos = list(out.rglob("*.mp4"))
    assert videos, f"no encoded video under {out}"
    assert all(v.stat().st_size > 0 for v in videos)
