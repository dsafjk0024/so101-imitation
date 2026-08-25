"""Assemble a LeRobot `observation` dict from live ROS messages.

Pure function, no rclpy import: it only needs `converter.config`/`decoders`, so
it can be unit-tested (Seam 4) without a ROS environment. Reuses the same
`decode()` path the converter uses when building training datasets, so
training and inference can't silently drift apart on joint order, image
shape, or feature keys.
"""

from __future__ import annotations

import numpy as np

from converter.config import Config, DEFAULT_CONFIG_PATH, load_config
from converter.decoders import decode

STATE_KEY = "observation.state"
WRIST_KEY = "observation.images.wrist"
TOP_KEY = "observation.images.top"


def build_observation(
    joint_state_msg,
    wrist_msg,
    top_msg,
    cfg: Config | None = None,
) -> dict[str, np.ndarray]:
    """Decode the three live inputs into the dict `predict_action()` expects."""
    if cfg is None:
        cfg = load_config(DEFAULT_CONFIG_PATH)

    by_key = {spec.key: spec for spec in cfg.features}

    state_spec = by_key[STATE_KEY]
    wrist_spec = by_key[WRIST_KEY]
    top_spec = by_key[TOP_KEY]

    return {
        STATE_KEY: decode(joint_state_msg, state_spec.msg_type, names=state_spec.names),
        WRIST_KEY: decode(wrist_msg, wrist_spec.msg_type, shape=wrist_spec.shape),
        TOP_KEY: decode(top_msg, top_spec.msg_type, shape=top_spec.shape),
    }
