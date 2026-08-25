"""Episode bag-writing lifecycle: start/write/stop/discard.

No rclpy Node here on purpose: this class only touches `rosbag2_py`, so it can
be exercised directly in tests (Seam: episode lifecycle) without a live ROS
graph. `episode_recorder_node.py` just wires rclpy subscription callbacks into
`write()` and command-topic messages into `start()`/`stop()`/`discard()`.

Follows the exact rosbag2_py API already proven by
`converter.convert`/`test_convert.write_fixture_bag`: `SequentialWriter`,
`StorageOptions(uri=..., storage_id=...)`, `ConverterOptions(cdr, cdr)`,
`create_topic(TopicMetadata(...))`, `.write(topic, raw, ts_ns)`. Dropping the
writer reference (rather than an explicit `.close()` -- rosbag2_py's
SequentialWriter has none) is what flushes and writes `metadata.yaml`; the
same pattern `test_convert.py` relies on (`del writer`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import rosbag2_py

EPISODE_DIR_PREFIX = "ep_"


class EpisodeWriter:
    def __init__(
        self,
        root_dir: str | Path,
        topics: list[tuple[str, str]],
        *,
        storage_id: str = "mcap",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.topics = list(topics)
        self.storage_id = storage_id
        self.next_episode_index = self._scan_next_index()

        self._writer: rosbag2_py.SequentialWriter | None = None
        self._current_dir: Path | None = None

    def _scan_next_index(self) -> int:
        """Resume numbering after existing `ep_NNN` dirs so re-runs don't clobber them."""
        if not self.root_dir.exists():
            return 0
        indices = []
        for d in self.root_dir.iterdir():
            if d.is_dir() and d.name.startswith(EPISODE_DIR_PREFIX):
                try:
                    indices.append(int(d.name[len(EPISODE_DIR_PREFIX):]))
                except ValueError:
                    continue
        return max(indices) + 1 if indices else 0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def current_dir(self) -> Path | None:
        return self._current_dir

    def start(self) -> Path:
        if self.is_recording:
            raise RuntimeError("already recording; stop or discard first")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = self.root_dir / f"{EPISODE_DIR_PREFIX}{self.next_episode_index:03d}"

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(episode_dir), storage_id=self.storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        for i, (topic, msg_type) in enumerate(self.topics):
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=i, name=topic, type=msg_type, serialization_format="cdr"
                )
            )

        self._writer = writer
        self._current_dir = episode_dir
        return episode_dir

    def write(self, topic: str, raw: bytes, timestamp_ns: int) -> None:
        if not self.is_recording:
            raise RuntimeError("not recording; call start() first")
        self._writer.write(topic, raw, timestamp_ns)

    def stop(self) -> Path:
        """Finalize the current episode (flushing metadata.yaml) and advance the counter."""
        if not self.is_recording:
            raise RuntimeError("not recording; nothing to stop")
        finished_dir = self._current_dir
        self._writer = None  # drops the only reference -> rosbag2_py closes and flushes
        self._current_dir = None
        self.next_episode_index += 1
        return finished_dir

    def discard(self) -> None:
        """Finalize then delete the current episode; the index is NOT advanced."""
        if not self.is_recording:
            raise RuntimeError("not recording; nothing to discard")
        discarded_dir = self._current_dir
        self._writer = None
        self._current_dir = None
        if discarded_dir.exists():
            shutil.rmtree(discarded_dir)
