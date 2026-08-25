"""Leader -> follower joint mirror relay.

Subscribes to the leader's joint states and republishes the six joint positions,
in the canonical order, onto the follower's forward_command_controller command
topic at a fixed rate. Stops commanding if the leader data goes stale.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Canonical joint order — shared by observation.state / action / URDF.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class TeleopRelay(Node):
    def __init__(self) -> None:
        super().__init__("leader_follower_relay")
        self.declare_parameter("leader_topic", "/leader/joint_states")
        self.declare_parameter("follower_topic", "/follower/forward_controller/commands")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("stale_timeout_s", 0.25)

        leader_topic = self.get_parameter("leader_topic").value
        follower_topic = self.get_parameter("follower_topic").value
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)

        self._latest: dict[str, float] | None = None
        self._last_stamp: float | None = None

        self.sub = self.create_subscription(JointState, leader_topic, self._on_leader, 10)
        self.pub = self.create_publisher(Float64MultiArray, follower_topic, 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"relay {leader_topic} -> {follower_topic} @ {rate:g}Hz, joints={JOINTS}"
        )
        self._warned_missing = False

    def _on_leader(self, msg: JointState) -> None:
        self._latest = dict(zip(msg.name, msg.position))
        self._last_stamp = self._now_s()

    def _tick(self) -> None:
        if self._latest is None or self._last_stamp is None:
            return
        if self._now_s() - self._last_stamp > self.stale_timeout_s:
            return  # leader stale — stop commanding
        try:
            positions = [float(self._latest[j]) for j in JOINTS]
        except KeyError as exc:
            if not self._warned_missing:
                self.get_logger().warn(f"leader joint_states missing joint {exc}; not commanding")
                self._warned_missing = True
            return
        self.pub.publish(Float64MultiArray(data=positions))

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> None:
    rclpy.init()
    node = TeleopRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
