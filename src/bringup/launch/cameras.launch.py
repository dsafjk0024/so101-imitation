"""Bring up the observation cameras declared by the dataset contract.

The topics, encodings and image sizes are NOT written here. They are read from
`converter.config` (`so101.yaml`) — the same single source of truth the recorder
and the converter read — and each camera node is placed in the namespace that
makes it publish exactly the configured topic. That is the whole point of this
file: a hand-typed `ros2 run usb_cam ...` line is the easiest way to end up with
a topic name that the recorder's start gate then refuses, or that the converter
rejects at conversion time.

Derivation, per image feature in the contract:

    topic  /follower/camera/wrist/image_raw
           └────────────┬──────────────┘└──┬──┘
                    namespace           node's own topic
    shape  [480, 640, 3]  ->  height 480, width 640, rgb8

Two drivers are supported; pick whichever is installed:

    driver:=usb_cam       (default)  ros-jazzy-usb-cam
    driver:=v4l2_camera              ros-jazzy-v4l2-camera

Run standalone:

    ros2 launch bringup cameras.launch.py

Or together with the arms:

    ros2 launch bringup teleop.launch.py enable_cameras:=true

Device paths default to `/dev/cam_<short>` (e.g. `/dev/cam_wrist`), matching
the udev SYMLINK rule keyed on USB port path (see `99-so101-cameras.rules`) --
raw `/dev/videoN` indices shift across replugs/reboots since both cameras are
the same model. Override with `wrist_device:=`/`top_device:=` only if those
symlinks don't exist yet or you're on a machine without the udev rule
installed; find the raw path with `v4l2-ctl --list-devices`. Verify after
launching, because a driver that silently falls back to another encoding or
size is not an error until the converter runs:

    ros2 topic echo <topic> --field encoding --once     # must be rgb8
    ros2 topic echo <topic> --field width    --once
    ros2 topic hz   <topic> --window 300               # with BOTH cameras running

Camera intrinsics/extrinsics are deliberately not set up here: ACT is
pixel-to-action end-to-end and uses neither. `frame_id` is filled in only so TF
and RViz have something sensible to show; getting those frames metrically right
is the separate hand-eye slice.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from converter.config import DEFAULT_CONFIG_PATH, load_config

IMAGE_TOPIC_SUFFIX = "/image_raw"

# Short feature name -> the URDF frame that camera is mounted on. Only affects
# TF/RViz, not the dataset (see module docstring). Unknown names fall back to
# `<short>_camera_optical_frame`.
FRAME_BY_SHORT = {
    "wrist": "wrist_camera_optical_frame",
    "top": "static_camera_optical_frame",
}


def _short_name(feature_key: str) -> str:
    """`observation.images.wrist` -> `wrist`."""
    return feature_key.rsplit(".", 1)[-1]


def _namespace(topic: str, feature_key: str) -> str:
    """Namespace that makes a camera node publish `topic` as its `image_raw`."""
    if not topic.endswith(IMAGE_TOPIC_SUFFIX):
        raise ValueError(
            f"{feature_key}: contract topic {topic!r} does not end in "
            f"{IMAGE_TOPIC_SUFFIX!r}, so no camera node namespace can produce it. "
            f"Either rename the topic in so101.yaml or launch that driver by hand."
        )
    return topic[: -len(IMAGE_TOPIC_SUFFIX)]


def _build_camera_actions(context, image_features):
    """Resolve every LaunchConfiguration up front (via `.perform(context)`) and build
    plain Node/TimerAction objects from the results.

    This runs synchronously as an OpaqueFunction, deliberately not leaving any
    LaunchConfiguration substitution to be resolved later (e.g. inside a
    TimerAction's deferred callback): when this file is included from
    teleop.launch.py, the include sits inside a GroupAction, which pops the
    LaunchConfigurations it scopes as soon as the synchronous visit finishes --
    before a TimerAction's delayed callback would get around to resolving them.
    Resolving everything here, while the scope is still live, sidesteps that
    entirely.
    """
    driver = LaunchConfiguration("driver").perform(context)
    if driver not in ("usb_cam", "v4l2_camera"):
        raise ValueError(f"driver must be usb_cam or v4l2_camera, got {driver!r}")
    framerate = float(LaunchConfiguration("framerate").perform(context))
    frame_prefix = LaunchConfiguration("frame_prefix").perform(context)
    stagger_s = float(LaunchConfiguration("camera_start_stagger_s").perform(context))

    actions = []
    for idx, spec in enumerate(image_features):
        short = _short_name(spec.key)
        namespace = _namespace(spec.topic, spec.key)
        height, width = spec.shape[0], spec.shape[1]
        device = LaunchConfiguration(f"{short}_device").perform(context)
        frame_id = frame_prefix + FRAME_BY_SHORT.get(short, f"{short}_camera_optical_frame")

        if driver == "usb_cam":
            node = Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name=f"{short}_camera",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "video_device": device,
                        "image_width": width,
                        "image_height": height,
                        # mjpeg2rgb, not yuyv2rgb: real-hardware smoke recording
                        # (outputs/recordings/so101_smoke5_ros) showed yuyv2rgb's
                        # software YUYV->RGB conversion producing washed-out,
                        # low-contrast frames plus intermittent tearing on the
                        # wrist camera (visible even in the raw bag, before AV1
                        # encoding). mjpeg2rgb decodes the camera's own hardware
                        # JPEG instead and was verified clean at the same
                        # 640x480@30fps. Both still deliver rgb8 on the wire.
                        "pixel_format": "mjpeg2rgb",
                        "framerate": framerate,
                        "camera_name": f"{short}_camera",
                        "frame_id": frame_id,
                    }
                ],
            )
        else:
            node = Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                name=f"{short}_camera",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "video_device": device,
                        # v4l2_camera takes [width, height], the opposite order
                        # from the contract's [height, width, channels] shape.
                        "image_size": [width, height],
                        "output_encoding": "rgb8",
                        "camera_frame_id": frame_id,
                    }
                ],
            )

        # Two identical UVC cameras (same Innomaker vendor/product ID) calling
        # VIDIOC_STREAMON at the same instant reliably crash both -- confirmed
        # on-hardware, reproducible even with cable/bandwidth ruled out as
        # causes. Starting them a beat apart avoids the race entirely.
        actions.append(TimerAction(period=idx * stagger_s, actions=[node]))

    return actions


def generate_launch_description():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    image_features = [spec for spec in cfg.features if spec.is_image]
    if not image_features:
        raise ValueError(
            f"{DEFAULT_CONFIG_PATH} declares no image features; nothing to launch"
        )

    actions = [
        DeclareLaunchArgument("driver", default_value="usb_cam"),  # usb_cam | v4l2_camera
        DeclareLaunchArgument(
            "framerate",
            default_value=str(float(cfg.fps)),
            description="requested camera fps; defaults to the contract fps",
        ),
        DeclareLaunchArgument("frame_prefix", default_value="follower/"),
        DeclareLaunchArgument(
            "camera_start_stagger_s",
            default_value="1.0",
            description="delay between starting each camera node (see _build_camera_actions)",
        ),
    ]

    for spec in image_features:
        short = _short_name(spec.key)
        actions.append(
            DeclareLaunchArgument(
                f"{short}_device",
                # Matches the udev SYMLINK rule (e.g. /dev/cam_wrist) keyed on
                # USB port path, not a raw /dev/videoN index -- those shift
                # across replugs/reboots since both cameras are the same model.
                default_value=f"/dev/cam_{short}",
                description=f"V4L2 device publishing {spec.topic}",
            )
        )

    actions.append(
        OpaqueFunction(function=_build_camera_actions, args=[image_features])
    )

    return LaunchDescription(actions)
