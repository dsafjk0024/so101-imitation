"""Episode recorder node: synchronized MCAP recording with keyboard-driven
start/stop/discard control.

Topics come from `converter.config` (the so101.yaml dataset contract), not a
hardcoded list, so recorder and converter can never record/expect different
topics (same "single source of truth" pattern as `inference.obs_builder`).
Each configured topic is subscribed to with `raw=True`: the callback receives
the still-serialized `bytes` rather than a deserialized message object, since
the recorder only needs to re-write the exact same bytes into a rosbag2 file
(a real message class must still be passed to `create_subscription` for
QoS/type introspection -- see `msg_types.py`). A separate `command_topic`
(`std_msgs/String`) carries "start"/"stop"/"discard"/"quit" from
`keyboard_teleop.py`, keeping terminal I/O out of this node entirely.

The bag-writing lifecycle itself lives in `EpisodeWriter`
(`episode_writer.py`); this node only translates ROS callbacks into calls on
it and tracks per-topic staleness for the start gate.

Run (after colcon build, so `converter`/`recorder` are on PYTHONPATH):

    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    ros2 run recorder episode_recorder_node --ros-args -p root_dir:=outputs/recordings

Not verified against a live ROS graph (no real camera/robot topics exist in
this sandbox) -- treat as documented but unverified beyond the
`EpisodeWriter`-level tests in `tests/test_episode_lifecycle.py`.
"""

from __future__ import annotations

import enum

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from converter.config import DEFAULT_CONFIG_PATH, load_config

from recorder.episode_writer import EpisodeWriter
from recorder.msg_types import MSG_CLASSES


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"


class EpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("episode_recorder_node")

        self.declare_parameter("config_path", str(DEFAULT_CONFIG_PATH))
        self.declare_parameter("root_dir", "outputs/recordings")
        self.declare_parameter("storage_id", "mcap")
        self.declare_parameter("start_gate_max_age_s", 0.5)
        self.declare_parameter("command_topic", "/recorder/command")

        cfg = load_config(self.get_parameter("config_path").value)
        root_dir = self.get_parameter("root_dir").value
        storage_id = self.get_parameter("storage_id").value
        self.start_gate_max_age_s = float(self.get_parameter("start_gate_max_age_s").value)
        command_topic = self.get_parameter("command_topic").value

        self.cfg = cfg
        self.topics = [(spec.topic, spec.msg_type) for spec in cfg.features]
        self.writer = EpisodeWriter(root_dir, self.topics, storage_id=storage_id)
        self.state = State.IDLE
        self.shutdown_requested = False

        # Last wall-clock receive time per topic -- the start gate refuses to
        # arm recording if a configured topic (dead/missing sensor) hasn't
        # published recently, same staleness pattern as sync_inference_node.
        self._last_seen_s: dict[str, float] = {}

        for topic, msg_type in self.topics:
            msg_class = MSG_CLASSES[msg_type]
            self.create_subscription(
                msg_class, topic, self._make_raw_callback(topic), 10, raw=True
            )

        self.create_subscription(String, command_topic, self._on_command, 10)

        self.get_logger().info(
            f"episode_recorder_node: root_dir={self.writer.root_dir}, "
            f"topics={[t for t, _ in self.topics]}, "
            f"next_episode={self.writer.next_episode_index}, "
            f"command_topic={command_topic}"
        )

    def _make_raw_callback(self, topic: str):
        def _callback(raw: bytes) -> None:
            self._last_seen_s[topic] = self._now_s()
            if self.state is State.RECORDING:
                # Float64MultiArray carries no header, so there is no
                # ROS-side stamp available without deserializing (which
                # would defeat raw=True); the node's own receive-time clock
                # is used for every topic for simplicity and consistency.
                self.writer.write(topic, raw, self.get_clock().now().nanoseconds)

        return _callback

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().lower()
        handler = {
            "start": self._handle_start,
            "stop": self._handle_stop,
            "discard": self._handle_discard,
            "quit": self._handle_quit,
        }.get(command)
        if handler is None:
            self.get_logger().warn(f"unknown command {command!r}")
            return
        handler()

    def _handle_start(self) -> None:
        if self.state is State.RECORDING:
            self.get_logger().warn("start ignored: already RECORDING")
            return
        stale = self._start_gate_failures()
        if stale:
            self.get_logger().warn(
                f"start refused: topic(s) missing or stale "
                f"(>{self.start_gate_max_age_s}s): {stale}"
            )
            return
        episode_dir = self.writer.start()
        self.state = State.RECORDING
        self.get_logger().info(f"IDLE -> RECORDING: {episode_dir}")

    def _start_gate_failures(self) -> list[str]:
        now = self._now_s()
        failures = []
        for topic, _ in self.topics:
            last = self._last_seen_s.get(topic)
            if last is None or now - last > self.start_gate_max_age_s:
                failures.append(topic)
        return failures

    def _handle_stop(self) -> None:
        if self.state is not State.RECORDING:
            self.get_logger().warn("stop ignored: not RECORDING")
            return
        episode_dir = self.writer.stop()
        self.state = State.IDLE
        self.get_logger().info(f"RECORDING -> IDLE: saved {episode_dir}")
        self._log_topic_hz(episode_dir)

    def _log_topic_hz(self, episode_dir) -> None:
        """Read back the metadata.yaml rosbag2 just flushed and log each
        topic's average hz (message_count / whole-bag duration), so a stale
        or dead publisher shows up immediately in the terminal instead of
        only being noticed later at conversion time."""
        info = yaml.safe_load((episode_dir / "metadata.yaml").read_text())[
            "rosbag2_bagfile_information"
        ]
        duration_s = info["duration"]["nanoseconds"] * 1e-9
        parts = [
            f"{entry['topic_metadata']['name']}="
            f"{(entry['message_count'] / duration_s if duration_s > 0 else 0.0):.1f}Hz"
            for entry in info["topics_with_message_count"]
        ]
        self.get_logger().info("episode hz (avg): " + ", ".join(parts))

    def _handle_discard(self) -> None:
        if self.state is not State.RECORDING:
            self.get_logger().warn("discard ignored: not RECORDING")
            return
        self.writer.discard()
        self.state = State.IDLE
        self.get_logger().info("RECORDING -> IDLE: episode discarded, index reused")

    def _handle_quit(self) -> None:
        if self.state is State.RECORDING:
            self.get_logger().warn("quit: discarding in-progress episode")
            self.writer.discard()
            self.state = State.IDLE
        self.get_logger().info("quit command received, shutting down")
        self.shutdown_requested = True


def main() -> None:
    rclpy.init()
    node = EpisodeRecorderNode()
    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
