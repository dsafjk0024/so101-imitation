"""Seam 2 — the config must produce exactly the agreed LeRobot v3.0 schema.

Runs without ROS and without lerobot: config.py and decoders.py import neither.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from converter.config import (
    DEFAULT_CONFIG_PATH,
    POS_SUFFIX,
    Config,
    load_config,
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
POS_NAMES = [j + POS_SUFFIX for j in JOINTS]


@pytest.fixture
def cfg() -> Config:
    return load_config(DEFAULT_CONFIG_PATH)


# --- the contract ---------------------------------------------------------


def test_scalar_contract_values(cfg):
    assert cfg.robot_type == "so101"
    assert cfg.fps == 30
    assert cfg.repo_id == "local/so101_pickplace"
    assert cfg.task == "Pick up the cube and place it in the bin."


def test_canonical_joint_order(cfg):
    assert cfg.joint_names == JOINTS


def test_bag_input_contract(cfg):
    by_key = {f.key: f for f in cfg.features}
    assert by_key["observation.state"].topic == "/follower/joint_states"
    assert by_key["observation.state"].msg_type == "sensor_msgs/msg/JointState"
    assert by_key["action"].topic == "/follower/forward_controller/commands"
    assert by_key["action"].msg_type == "std_msgs/msg/Float64MultiArray"
    assert by_key["observation.images.wrist"].topic == "/follower/camera/wrist/image_raw"
    assert by_key["observation.images.top"].topic == "/follower/camera/top/image_raw"
    for key in ("observation.images.wrist", "observation.images.top"):
        assert by_key[key].msg_type == "sensor_msgs/msg/Image"


def test_max_age_windows(cfg):
    by_key = {f.key: f for f in cfg.features}
    assert by_key["observation.state"].max_age_s == pytest.approx(0.05)
    assert by_key["observation.state"].max_age_ns == 50_000_000
    assert by_key["action"].max_age_ns == 50_000_000
    assert by_key["observation.images.wrist"].max_age_ns == 100_000_000


def test_lerobot_features_matches_contract_exactly(cfg):
    assert cfg.lerobot_features() == {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": POS_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": POS_NAMES,
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }


def test_no_videos_switches_image_dtype(cfg):
    feats = cfg.lerobot_features(use_videos=False)
    assert feats["observation.images.wrist"]["dtype"] == "image"
    assert feats["observation.images.top"]["dtype"] == "image"
    # non-image features unaffected
    assert feats["observation.state"]["dtype"] == "float32"


def test_by_topic_maps_every_feature(cfg):
    by_topic = cfg.by_topic()
    assert len(by_topic) == 4
    assert by_topic["/follower/joint_states"].key == "observation.state"


def test_is_image_flag(cfg):
    by_key = {f.key: f for f in cfg.features}
    assert by_key["observation.images.top"].is_image is True
    assert by_key["action"].is_image is False


# --- validation rejections ------------------------------------------------


def _mutate(cfg: Config, **changes) -> Config:
    return dataclasses.replace(cfg, **changes)


def test_rejects_non_positive_fps(cfg):
    with pytest.raises(ValueError, match="fps"):
        _mutate(cfg, fps=0).validate()


def test_rejects_empty_features(cfg):
    with pytest.raises(ValueError, match="features"):
        _mutate(cfg, features=()).validate()


def test_rejects_empty_task(cfg):
    with pytest.raises(ValueError, match="task"):
        _mutate(cfg, task="").validate()


def test_rejects_duplicate_feature_keys(cfg):
    dupe = cfg.features + (dataclasses.replace(cfg.features[0], topic="/other"),)
    with pytest.raises(ValueError, match="duplicate feature keys"):
        _mutate(cfg, features=dupe).validate()


def test_rejects_duplicate_topics(cfg):
    dupe = cfg.features + (dataclasses.replace(cfg.features[0], key="other.key"),)
    with pytest.raises(ValueError, match="duplicate feature topics"):
        _mutate(cfg, features=dupe).validate()


def test_rejects_missing_required_keys(cfg):
    only_images = tuple(f for f in cfg.features if f.is_image)
    with pytest.raises(ValueError, match="observation.state"):
        _mutate(cfg, features=only_images).validate()


def test_rejects_non_positive_max_age(cfg):
    bad = (dataclasses.replace(cfg.features[0], max_age_s=0.0),) + cfg.features[1:]
    with pytest.raises(ValueError, match="max_age_s"):
        _mutate(cfg, features=bad).validate()


def test_rejects_msg_type_without_decoder(cfg):
    bad = (
        dataclasses.replace(cfg.features[0], msg_type="sensor_msgs/msg/CompressedImage"),
    ) + cfg.features[1:]
    with pytest.raises(ValueError, match="no decoder"):
        _mutate(cfg, features=bad).validate()


def test_rejects_vector_names_not_in_canonical_order(cfg):
    scrambled = tuple(reversed(JOINTS))
    bad = (dataclasses.replace(cfg.features[0], names=scrambled),) + cfg.features[1:]
    with pytest.raises(ValueError, match="canonical"):
        _mutate(cfg, features=bad).validate()


def test_rejects_image_without_three_dim_shape(cfg):
    images = [f for f in cfg.features if f.is_image]
    bad = tuple(
        dataclasses.replace(f, shape=(480, 640)) if f.is_image else f
        for f in cfg.features
    )
    assert images  # guard: the fixture really has image features
    with pytest.raises(ValueError, match=r"\[H, W, C\]"):
        _mutate(cfg, features=bad).validate()


def test_rejects_image_with_wrong_channel_count(cfg):
    bad = tuple(
        dataclasses.replace(f, shape=(480, 640, 1)) if f.is_image else f
        for f in cfg.features
    )
    with pytest.raises(ValueError, match="channels"):
        _mutate(cfg, features=bad).validate()


# --- loader ---------------------------------------------------------------


def test_load_rejects_missing_top_level_field(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["fps"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_config(path)


def test_load_rejects_unknown_feature_field(tmp_path):
    """A stale key such as the removed `stamp_src` must not be silently ignored."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["features"][0]["stamp_src"] = "header"
    path = tmp_path / "stale.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_config(path)
