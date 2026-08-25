"""Keyboard control for episode_recorder_node.

Reads single keypresses from the terminal (no Enter required, via
`termios`/`tty` cbreak mode + `select.select` on stdin -- the well-known
`teleop_twist_keyboard`-style pattern) and publishes a `std_msgs/String`
command to `command_topic`. Deliberately its own node/script, separate from
episode_recorder_node: terminal raw-mode I/O has nothing to do with bag
writing, and decoupling them means the bag-writing node never has to touch
stdin at all.

Keys: s = start, e = stop+save (end), d = discard, q = quit.

Run:

    ros2 run recorder keyboard_teleop
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEY_COMMANDS = {
    "s": "start",
    "e": "stop",
    "d": "discard",
    "q": "quit",
}

HELP = """keyboard_teleop -- recorder controls:
  s = start recording
  e = stop + save the current episode (end)
  d = discard the current episode
  q = quit
"""


class KeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__("recorder_keyboard_teleop")
        self.declare_parameter("command_topic", "/recorder/command")
        command_topic = self.get_parameter("command_topic").value
        self.pub = self.create_publisher(String, command_topic, 10)
        self.get_logger().info(f"publishing recorder commands on {command_topic}")

    def send(self, command: str) -> None:
        self.pub.publish(String(data=command))
        self.get_logger().info(f"sent command: {command}")


def _read_key(timeout_s: float = 0.1) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    return sys.stdin.read(1) if ready else None


def main() -> None:
    print(HELP, flush=True)
    rclpy.init()
    node = KeyboardTeleop()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = _read_key()
            if key is not None:
                command = KEY_COMMANDS.get(key.lower())
                if command is not None:
                    node.send(command)
                    if command == "quit":
                        break
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
