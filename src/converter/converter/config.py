"""Conversion contract: joint order, bag topic mapping, and the LeRobot v3.0
feature schema.

Pure Python — imports yaml and `converter.decoders` only, never ROS or lerobot,
so the contract can be unit-tested without a ROS environment (Seam 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from converter.decoders import (
    IMAGE_MSG_TYPE,
    MSG_TYPE_DTYPES,
    SUPPORTED_MSG_TYPES,
)

POS_SUFFIX = ".pos"

# Feature keys whose `names` get the LeRobot per-joint `.pos` suffix.
POS_KEYS = frozenset({"observation.state", "action"})

# Keys every dataset must carry for training and inference to line up.
REQUIRED_KEYS = frozenset({"observation.state", "action"})

_FEATURE_FIELDS = frozenset({"key", "topic", "msg_type", "max_age_s", "names", "shape"})
_TOP_LEVEL_FIELDS = ("robot_type", "fps", "task", "repo_id", "joint_names", "features")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "so101.yaml"

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    topic: str
    msg_type: str
    max_age_s: float
    names: tuple[str, ...] | None = None
    shape: tuple[int, ...] | None = None

    @property
    def is_image(self) -> bool:
        return self.msg_type == IMAGE_MSG_TYPE

    @property
    def max_age_ns(self) -> int:
        return int(round(self.max_age_s * NS_PER_S))


@dataclass(frozen=True)
class Config:
    robot_type: str
    fps: int
    task: str
    repo_id: str
    joint_names: tuple[str, ...]
    features: tuple[FeatureSpec, ...]

    def by_topic(self) -> dict[str, FeatureSpec]:
        return {f.topic: f for f in self.features}

    def lerobot_features(self, use_videos: bool = True) -> dict:
        """Schema for `LeRobotDataset.create(features=...)`.

        Images are channel-last (H, W, C) per LeRobot convention.
        """
        features: dict = {}
        for spec in self.features:
            if spec.is_image:
                features[spec.key] = {
                    "dtype": "video" if use_videos else "image",
                    "shape": tuple(spec.shape),
                    "names": ["height", "width", "channels"],
                }
            else:
                names = list(spec.names)
                if spec.key in POS_KEYS:
                    names = [n + POS_SUFFIX for n in names]
                features[spec.key] = {
                    "dtype": MSG_TYPE_DTYPES[spec.msg_type],
                    "shape": (len(spec.names),),
                    "names": names,
                }
        return features

    def validate(self) -> None:
        if not self.robot_type:
            raise ValueError("robot_type must be non-empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if not self.task:
            raise ValueError("task must be non-empty")
        if not self.repo_id:
            raise ValueError("repo_id must be non-empty")
        if not self.joint_names:
            raise ValueError("joint_names must be non-empty")
        if not self.features:
            raise ValueError("features must be non-empty")

        keys = [f.key for f in self.features]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature keys: {duplicates}")

        topics = [f.topic for f in self.features]
        duplicate_topics = sorted({t for t in topics if topics.count(t) > 1})
        if duplicate_topics:
            raise ValueError(f"duplicate feature topics: {duplicate_topics}")

        missing = REQUIRED_KEYS - set(keys)
        if missing:
            raise ValueError(f"missing required feature keys: {sorted(missing)}")

        for spec in self.features:
            if spec.msg_type not in SUPPORTED_MSG_TYPES:
                raise ValueError(
                    f"feature {spec.key!r}: no decoder for msg_type "
                    f"{spec.msg_type!r}; supported: {sorted(SUPPORTED_MSG_TYPES)}"
                )
            if spec.max_age_s <= 0:
                raise ValueError(
                    f"feature {spec.key!r}: max_age_s must be > 0, got {spec.max_age_s}"
                )
            if spec.is_image:
                if spec.shape is None or len(spec.shape) != 3:
                    raise ValueError(
                        f"feature {spec.key!r}: image feature needs shape [H, W, C]"
                    )
                if spec.shape[2] != 3:
                    raise ValueError(
                        f"feature {spec.key!r}: image must have 3 channels, "
                        f"got {spec.shape[2]}"
                    )
            else:
                if spec.names is None:
                    raise ValueError(
                        f"feature {spec.key!r}: vector feature needs names"
                    )
                if tuple(spec.names) != self.joint_names:
                    raise ValueError(
                        f"feature {spec.key!r}: names {list(spec.names)} do not match "
                        f"canonical joint_names {list(self.joint_names)}"
                    )


def load_config(path: str | Path) -> Config:
    """Load and validate a conversion config YAML."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    missing = [k for k in _TOP_LEVEL_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"{path}: config missing required fields: {missing}")

    features = []
    for entry in raw["features"]:
        unknown = sorted(set(entry) - _FEATURE_FIELDS)
        if unknown:
            raise ValueError(
                f"{path}: feature {entry.get('key')!r} has unknown field(s): {unknown}; "
                f"allowed: {sorted(_FEATURE_FIELDS)}"
            )
        features.append(
            FeatureSpec(
                key=entry["key"],
                topic=entry["topic"],
                msg_type=entry["msg_type"],
                max_age_s=float(entry["max_age_s"]),
                names=tuple(entry["names"]) if entry.get("names") else None,
                shape=tuple(int(x) for x in entry["shape"]) if entry.get("shape") else None,
            )
        )

    cfg = Config(
        robot_type=raw["robot_type"],
        fps=int(raw["fps"]),
        task=raw["task"],
        repo_id=raw["repo_id"],
        joint_names=tuple(raw["joint_names"]),
        features=tuple(features),
    )
    cfg.validate()
    return cfg
