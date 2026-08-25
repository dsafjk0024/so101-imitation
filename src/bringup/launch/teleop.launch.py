"""Leader-follower teleoperation bring-up.

Starts the leader (state-only) + follower (forward position controller), then
after a short delay starts the teleop relay node that mirrors leader joint
positions onto the follower command topic.

Cameras are off by default so this launch stays usable for wiring checks with
`hardware_type:=mock` and no cameras plugged in. Recording needs them, since the
recorder's start gate refuses to arm while a contract topic is missing:

    ros2 launch bringup teleop.launch.py enable_cameras:=true

`wrist_device`/`top_device` default to the udev SYMLINK paths (`/dev/cam_wrist`,
`/dev/cam_top`, see `99-so101-cameras.rules`) -- override only on a machine
without that rule installed.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hardware_type = LaunchConfiguration("hardware_type")
    leader_usb = LaunchConfiguration("leader_usb")
    follower_usb = LaunchConfiguration("follower_usb")
    joint_config_file = LaunchConfiguration("joint_config_file")
    teleop_delay_s = LaunchConfiguration("teleop_delay_s")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")

    bringup = FindPackageShare("bringup")

    # GroupAction scopes each include's LaunchConfigurations so that the leader
    # and follower `controller_config_file` (and other same-named args) don't
    # leak into each other's scope.
    leader = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([bringup, "launch", "leader.launch.py"])
            ),
            launch_arguments={
                "namespace": "leader",
                "hardware_type": hardware_type,
                "usb_port": leader_usb,
                "joint_config_file": joint_config_file,
            }.items(),
        )
    ])

    follower = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([bringup, "launch", "follower.launch.py"])
            ),
            launch_arguments={
                "namespace": "follower",
                "hardware_type": hardware_type,
                "usb_port": follower_usb,
                "joint_config_file": joint_config_file,
            }.items(),
        )
    ])

    # Observation cameras. cameras.launch.py owns the topic/encoding/size
    # derivation from so101.yaml; the arguments forwarded here are only the
    # ones a user has to set per machine. To bring up more cameras than the
    # two the contract currently declares, launch cameras.launch.py directly.
    cameras = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([bringup, "launch", "cameras.launch.py"])
            ),
            launch_arguments={
                "driver": LaunchConfiguration("driver"),
                "wrist_device": LaunchConfiguration("wrist_device"),
                "top_device": LaunchConfiguration("top_device"),
            }.items(),
        )
    ], condition=IfCondition(LaunchConfiguration("enable_cameras")))

    # Delay teleop relay until controllers are up and joint_states flow.
    # Disable with enable_relay:=false when something else (e.g.
    # inference.sync_inference_node) needs to publish
    # /follower/forward_controller/commands instead -- both writers on the
    # same topic would otherwise race.
    teleop_relay = TimerAction(
        period=teleop_delay_s,
        actions=[
            Node(
                package="teleop",
                executable="teleop_node",
                name="leader_follower_relay",
                output="screen",
                parameters=[{
                    "leader_topic": "/leader/joint_states",
                    "follower_topic": "/follower/forward_controller/commands",
                    "publish_rate_hz": publish_rate_hz,
                }],
            )
        ],
        condition=IfCondition(LaunchConfiguration("enable_relay")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("hardware_type", default_value="real"),  # real | mock
        DeclareLaunchArgument("leader_usb", default_value="/dev/so101_leader"),
        DeclareLaunchArgument("follower_usb", default_value="/dev/so101_follower"),
        DeclareLaunchArgument("joint_config_file", default_value=""),
        DeclareLaunchArgument("teleop_delay_s", default_value="3.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("enable_relay", default_value="true"),
        # Cameras: off by default so mock wiring checks need no hardware.
        # Defaults mirror cameras.launch.py, which is where they are documented.
        DeclareLaunchArgument("enable_cameras", default_value="false"),
        DeclareLaunchArgument("driver", default_value="usb_cam"),  # usb_cam | v4l2_camera
        DeclareLaunchArgument("wrist_device", default_value="/dev/cam_wrist"),
        DeclareLaunchArgument("top_device", default_value="/dev/cam_top"),
        leader,
        follower,
        cameras,
        teleop_relay,
    ])
