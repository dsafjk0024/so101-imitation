"""Bag reading and LeRobot v3.0 dataset writing.

This is the only module that imports ROS or lerobot. It needs both at once:
rosbag2_py from a sourced ROS jazzy and lerobot from the pixi env. Both run on
Python 3.12, so a shell with ROS sourced makes rosbag2_py importable inside the
pixi environment through PYTHONPATH.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterator

import rosbag2_py
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from converter.config import Config
from converter.decoders import decode
from converter.grid import GridStats, iter_frames

logger = logging.getLogger(__name__)


class EpisodeSkipped(Exception):
    """Raised when an episode must not be written to the dataset."""


def find_episode_dirs(root: str | Path) -> list[Path]:
    """Episode bag directories under `root`, in ascending name order.

    Names must be zero-padded (`ep_000`, `ep_001`, ...): lexicographic order is
    what decides episode order, and `ep_9` would sort after `ep_10`.
    """
    root = Path(root)
    return sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "metadata.yaml").is_file()
    )


def _storage_id(bag_dir: Path, default: str = "mcap") -> str:
    meta_path = bag_dir / "metadata.yaml"
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return default
    info = meta.get("rosbag2_bagfile_information") or {}
    storage = info.get("storage_identifier")
    return storage if isinstance(storage, str) and storage else default


def open_reader(bag_dir: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=_storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    return reader


def read_bag_messages(bag_dir: Path, cfg: Config) -> Iterator[tuple[str, int, object]]:
    """Yield `(topic, bag_ts_ns, decoded_value)` for configured topics.

    Timestamps are the bag receive times, not `header.stamp`: not every message
    type carries a header, and one clock source keeps alignment consistent.
    """
    reader = open_reader(bag_dir)
    bag_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    by_topic = cfg.by_topic()

    for topic, spec in by_topic.items():
        actual = bag_types.get(topic)
        if actual is None:
            raise ValueError(
                f"{bag_dir.name}: configured topic {topic!r} not found in bag "
                f"(feature {spec.key!r}); bag has {sorted(bag_types)}"
            )
        if actual != spec.msg_type:
            raise ValueError(
                f"{bag_dir.name}: type mismatch on {topic!r}: config expects "
                f"{spec.msg_type!r} but bag recorded {actual!r}"
            )

    msg_classes = {topic: get_message(bag_types[topic]) for topic in by_topic}

    while reader.has_next():
        topic, raw, ts_ns = reader.read_next()
        spec = by_topic.get(topic)
        if spec is None:
            continue
        msg = deserialize_message(raw, msg_classes[topic])
        yield topic, ts_ns, decode(
            msg, spec.msg_type, names=spec.names, shape=spec.shape
        )


def convert_episode(
    bag_dir: Path,
    cfg: Config,
    dataset: LeRobotDataset,
    *,
    max_interior_drop_frac: float,
) -> GridStats:
    """Convert one bag into one dataset episode.

    Frames are buffered by `add_frame` and only committed by `save_episode`, so
    the interior-drop verdict -- which needs the last emitted tick, and thus the
    whole episode -- can still reject the episode.
    """
    stats = GridStats()
    for frame in iter_frames(read_bag_messages(bag_dir, cfg), cfg, stats):
        frame["task"] = cfg.task
        dataset.add_frame(frame)
    stats.finalize()

    if stats.emitted == 0:
        dataset.clear_episode_buffer()
        raise EpisodeSkipped(
            f"{bag_dir.name}: no frames produced "
            f"(leading drops={stats.dropped_leading}); check topic rates and max_age_s"
        )

    if stats.interior_drop_frac > max_interior_drop_frac:
        first_gaps = stats.interior_drops[:5]
        dataset.clear_episode_buffer()
        raise EpisodeSkipped(
            f"{bag_dir.name}: {stats.dropped_interior}/{stats.span_ticks} interior "
            f"ticks dropped ({stats.interior_drop_frac:.1%} > "
            f"{max_interior_drop_frac:.1%}). lerobot derives timestamps from "
            f"frame_index, so interior gaps would be recorded as normal spacing "
            f"and distort the timeline. First gaps (tick, missing feature): "
            f"{first_gaps}"
        )

    for tick, key in stats.interior_drops:
        logger.warning(
            "%s: tick %d dropped, feature %r had no fresh sample",
            bag_dir.name,
            tick,
            key,
        )

    dataset.save_episode()
    return stats


def convert_all(
    bags_root: str | Path,
    cfg: Config,
    root: str | Path,
    *,
    use_videos: bool = True,
    max_interior_drop_frac: float = 0.02,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    """Convert every episode bag under `bags_root` into one dataset at `root`."""
    bags_root = Path(bags_root)
    root = Path(root)

    episode_dirs = find_episode_dirs(bags_root)
    if limit is not None:
        episode_dirs = episode_dirs[:limit]
    if not episode_dirs:
        raise ValueError(
            f"no episode bag directories under {bags_root} "
            "(expected subdirectories containing metadata.yaml)"
        )

    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"dataset root {root} already exists; pass overwrite to replace it"
            )
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        features=cfg.lerobot_features(use_videos=use_videos),
        root=root,
        robot_type=cfg.robot_type,
        use_videos=use_videos,
    )

    by_topic = cfg.by_topic()
    converted: list[tuple[str, GridStats]] = []
    skipped: list[tuple[str, str]] = []

    try:
        for bag_dir in episode_dirs:
            try:
                stats = convert_episode(
                    bag_dir, cfg, dataset, max_interior_drop_frac=max_interior_drop_frac
                )
            except EpisodeSkipped as exc:
                logger.error("skipping episode: %s", exc)
                skipped.append((bag_dir.name, str(exc)))
                continue
            converted.append((bag_dir.name, stats))
            logger.info(
                "%s: %d frames (leading=%d interior=%d trailing=%d)",
                bag_dir.name,
                stats.emitted,
                stats.dropped_leading,
                stats.dropped_interior,
                stats.dropped_trailing,
            )
    finally:
        # Required, not optional: without it the parquet footer metadata is never
        # written and the dataset is unreadable. LeRobotDataset only flushes on
        # __del__ otherwise, which makes durability depend on GC timing.
        dataset.finalize()

    return {
        "root": root,
        "episodes_converted": len(converted),
        "episodes_skipped": skipped,
        "frames": sum(s.emitted for _, s in converted),
        "per_episode": [
            {
                "name": name,
                "frames": s.emitted,
                "dropped_leading": s.dropped_leading,
                "dropped_interior": s.dropped_interior,
                "dropped_trailing": s.dropped_trailing,
                "features": {
                    by_topic[topic].key: buf.summary()
                    for topic, buf in s.buffers.items()
                },
            }
            for name, s in converted
        ],
    }
