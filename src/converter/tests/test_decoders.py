"""Decoder unit tests. Messages are stand-in objects, not real ROS messages —
decoders duck-type their input, so these run without a ROS environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from converter.decoders import (
    IMAGE_MSG_TYPE,
    MSG_TYPE_DTYPES,
    SUPPORTED_MSG_TYPES,
    decode,
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class FakeImage:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes


@dataclass
class FakeJointState:
    name: list[str] = field(default_factory=list)
    position: list[float] = field(default_factory=list)


@dataclass
class FakeFloat64MultiArray:
    data: list[float] = field(default_factory=list)


def make_image(h=4, w=3, encoding="rgb8", step=None, fill=None):
    arr = (
        fill
        if fill is not None
        else np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    )
    return FakeImage(
        height=h,
        width=w,
        encoding=encoding,
        step=w * 3 if step is None else step,
        data=arr.tobytes(),
    )


# --- registry -------------------------------------------------------------


def test_supported_msg_types_and_dtypes():
    assert SUPPORTED_MSG_TYPES == {
        "sensor_msgs/msg/Image",
        "sensor_msgs/msg/JointState",
        "std_msgs/msg/Float64MultiArray",
    }
    assert MSG_TYPE_DTYPES["sensor_msgs/msg/Image"] == "video"
    assert MSG_TYPE_DTYPES["sensor_msgs/msg/JointState"] == "float32"
    assert MSG_TYPE_DTYPES["std_msgs/msg/Float64MultiArray"] == "float32"
    assert IMAGE_MSG_TYPE == "sensor_msgs/msg/Image"


def test_unknown_msg_type_raises():
    with pytest.raises(KeyError, match="no decoder"):
        decode(object(), "sensor_msgs/msg/CompressedImage")


# --- Image ----------------------------------------------------------------


def test_image_decodes_to_hwc_uint8():
    expected = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    out = decode(make_image(), IMAGE_MSG_TYPE, shape=(4, 3, 3))
    assert out.dtype == np.uint8
    assert out.shape == (4, 3, 3)
    np.testing.assert_array_equal(out, expected)


def test_image_result_is_writable_and_independent_of_message():
    msg = make_image()
    out = decode(msg, IMAGE_MSG_TYPE)
    out[0, 0, 0] = 200  # must not raise: downstream writers may need writability
    assert out[0, 0, 0] == 200


def test_image_rejects_non_rgb8_encoding():
    with pytest.raises(ValueError, match="rgb8"):
        decode(make_image(encoding="bgr8"), IMAGE_MSG_TYPE)


def test_image_rejects_padded_rows():
    with pytest.raises(ValueError, match="padded"):
        decode(make_image(step=99), IMAGE_MSG_TYPE)


def test_image_rejects_shape_mismatch_with_config():
    with pytest.raises(ValueError, match="!="):
        decode(make_image(h=4, w=3), IMAGE_MSG_TYPE, shape=(480, 640, 3))


def test_image_rejects_truncated_data():
    msg = make_image(h=4, w=3)
    msg.data = msg.data[:-3]
    with pytest.raises(ValueError, match="data size"):
        decode(msg, IMAGE_MSG_TYPE)


# --- JointState -----------------------------------------------------------


def test_joint_state_reorders_by_name():
    # deliberately scrambled relative to canonical order
    msg = FakeJointState(
        name=[
            "gripper",
            "elbow_flex",
            "shoulder_pan",
            "wrist_roll",
            "shoulder_lift",
            "wrist_flex",
        ],
        position=[0.6, 0.3, 0.1, 0.5, 0.2, 0.4],
    )
    out = decode(msg, "sensor_msgs/msg/JointState", names=JOINTS)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)


def test_joint_state_ignores_extra_joints():
    msg = FakeJointState(
        name=[*JOINTS, "some_other_joint"],
        position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.9],
    )
    out = decode(msg, "sensor_msgs/msg/JointState", names=JOINTS)
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)


def test_joint_state_missing_joint_raises_rather_than_filling_zero():
    msg = FakeJointState(
        name=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        position=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    with pytest.raises(KeyError, match="gripper"):
        decode(msg, "sensor_msgs/msg/JointState", names=JOINTS)


def test_joint_state_position_shorter_than_name_raises():
    msg = FakeJointState(name=list(JOINTS), position=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="position"):
        decode(msg, "sensor_msgs/msg/JointState", names=JOINTS)


def test_joint_state_requires_names():
    msg = FakeJointState(name=list(JOINTS), position=[0.1] * 6)
    with pytest.raises(ValueError, match="names"):
        decode(msg, "sensor_msgs/msg/JointState")


# --- Float64MultiArray ----------------------------------------------------


def test_float64_multiarray_decodes_to_float32_vector():
    msg = FakeFloat64MultiArray(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = decode(msg, "std_msgs/msg/Float64MultiArray", names=JOINTS)
    assert out.dtype == np.float32
    assert out.shape == (6,)
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)


def test_float64_multiarray_wrong_length_raises():
    msg = FakeFloat64MultiArray(data=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="expected 6"):
        decode(msg, "std_msgs/msg/Float64MultiArray", names=JOINTS)
