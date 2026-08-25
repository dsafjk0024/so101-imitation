"""Registry of the so101.yaml `msg_type` strings -> importable rclpy message
classes.

ROS-dependent by necessity (unlike `converter.decoders`, which deliberately
avoids ROS imports): `create_subscription` needs the real message class to
resolve QoS/type info, even for `raw=True` subscriptions where the callback
never sees a deserialized message. Keys match
`converter.decoders.SUPPORTED_MSG_TYPES` exactly, checked at import time so
this registry and the decoder registry cannot silently drift apart.
"""

from __future__ import annotations

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from converter.decoders import SUPPORTED_MSG_TYPES

MSG_CLASSES = {
    "sensor_msgs/msg/JointState": JointState,
    "sensor_msgs/msg/Image": Image,
    "std_msgs/msg/Float64MultiArray": Float64MultiArray,
}

assert set(MSG_CLASSES) == set(SUPPORTED_MSG_TYPES), (
    f"msg_types.MSG_CLASSES {sorted(MSG_CLASSES)} does not match "
    f"converter.decoders.SUPPORTED_MSG_TYPES {sorted(SUPPORTED_MSG_TYPES)}"
)
