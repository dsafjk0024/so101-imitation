"""ROS message -> numpy decoders.

No ROS imports here on purpose: messages are duck-typed via the attributes we
read, so decoders can be exercised with plain stand-in objects and `config.py`
can consult the supported-type registry without a ROS environment.
"""

from __future__ import annotations

import numpy as np

IMAGE_MSG_TYPE = "sensor_msgs/msg/Image"

# msg_type -> LeRobot feature dtype
MSG_TYPE_DTYPES: dict[str, str] = {
    IMAGE_MSG_TYPE: "video",
    "sensor_msgs/msg/JointState": "float32",
    "std_msgs/msg/Float64MultiArray": "float32",
}

SUPPORTED_MSG_TYPES = frozenset(MSG_TYPE_DTYPES)

IMAGE_ENCODING = "rgb8"
IMAGE_CHANNELS = 3


def decode(
    msg,
    msg_type: str,
    *,
    names: tuple[str, ...] | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Decode a ROS message into the array stored in the dataset frame."""
    if msg_type == IMAGE_MSG_TYPE:
        return _decode_image(msg, shape)
    if msg_type == "sensor_msgs/msg/JointState":
        return _decode_joint_state(msg, names)
    if msg_type == "std_msgs/msg/Float64MultiArray":
        return _decode_float64_multiarray(msg, names)
    raise KeyError(
        f"no decoder for msg_type {msg_type!r}; "
        f"supported: {sorted(SUPPORTED_MSG_TYPES)}"
    )


def _decode_image(msg, shape: tuple[int, ...] | None) -> np.ndarray:
    if msg.encoding != IMAGE_ENCODING:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; "
            f"the dataset contract requires {IMAGE_ENCODING!r}"
        )

    height, width = int(msg.height), int(msg.width)
    expected_step = width * IMAGE_CHANNELS
    if int(msg.step) != expected_step:
        raise ValueError(
            f"padded image rows are not supported: step={msg.step}, "
            f"expected {expected_step} for width={width}"
        )

    flat = np.frombuffer(memoryview(msg.data), dtype=np.uint8)
    expected_size = height * expected_step
    if flat.size != expected_size:
        raise ValueError(
            f"image data size {flat.size} != height*step {expected_size} "
            f"({height}x{width})"
        )

    # Copy rather than return a view into the message buffer: the array outlives
    # the message inside the as-of buffer, and downstream writers may need it
    # writable.
    img = flat.reshape(height, width, IMAGE_CHANNELS).copy()

    if shape is not None and img.shape != tuple(shape):
        raise ValueError(
            f"decoded image shape {img.shape} != configured shape {tuple(shape)}"
        )
    return img


def _decode_joint_state(msg, names: tuple[str, ...] | None) -> np.ndarray:
    if not names:
        raise ValueError("JointState decoding requires configured joint names")

    position = np.asarray(msg.position, dtype=np.float32)
    index = {name: i for i, name in enumerate(msg.name)}

    out = np.empty(len(names), dtype=np.float32)
    for slot, joint in enumerate(names):
        i = index.get(joint)
        if i is None:
            # Filling a default here would silently corrupt the dataset.
            raise KeyError(
                f"joint {joint!r} missing from JointState.name={list(msg.name)}"
            )
        if i >= position.size:
            raise ValueError(
                f"JointState.position has {position.size} values, too short for "
                f"joint {joint!r} at index {i}"
            )
        out[slot] = position[i]
    return out


def _decode_float64_multiarray(msg, names: tuple[str, ...] | None) -> np.ndarray:
    arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if names is not None and arr.size != len(names):
        raise ValueError(
            f"Float64MultiArray has {arr.size} values, expected {len(names)}"
        )
    return arr
