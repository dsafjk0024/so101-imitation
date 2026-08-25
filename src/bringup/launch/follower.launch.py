"""Bring up the FOLLOWER arm under the /follower namespace.

Loads the SO-101 URDF (variant=follower → position command interface), starts
robot_state_publisher + ros2_control_node + joint_state_broadcaster +
forward_controller. Absolute position commands are accepted on
/follower/forward_controller/commands (Float64MultiArray, 6 joints).
Cameras are disabled here (added in the camera-calibration slice).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    frame_prefix = LaunchConfiguration("frame_prefix")
    hardware_type = LaunchConfiguration("hardware_type")
    usb_port = LaunchConfiguration("usb_port")
    joint_config_file = LaunchConfiguration("joint_config_file")
    controller_config_file = LaunchConfiguration("controller_config_file")

    xacro_file = PathJoinSubstitution(
        [FindPackageShare("so101_description"), "urdf", "so101_arm.urdf.xacro"]
    )
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                xacro_file,
                " variant:=follower",
                " use_ros2_control:=true",
                " hardware_type:=",
                hardware_type,
                " usb_port:=",
                usb_port,
                " joint_config_file:=",
                joint_config_file,
                " enable_static_cam:=false",
                " enable_wrist_cam:=false",
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="follower"),
            DeclareLaunchArgument("hardware_type", default_value="real"),  # real | mock
            DeclareLaunchArgument("usb_port", default_value="/dev/so101_follower"),
            DeclareLaunchArgument("frame_prefix", default_value="follower/"),
            DeclareLaunchArgument("joint_config_file", default_value=""),
            DeclareLaunchArgument(
                "controller_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bringup"),
                        "config",
                        "ros2_control",
                        "follower_controllers.yaml",
                    ]
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=namespace,
                parameters=[
                    {
                        "robot_description": robot_description,
                        "frame_prefix": frame_prefix,
                    }
                ],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                namespace=namespace,
                parameters=[controller_config_file],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=[
                    "joint_state_broadcaster",
                    "--param-file",
                    controller_config_file,
                ],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=[
                    "forward_controller",
                    "--param-file",
                    controller_config_file,
                ],
                output="screen",
            ),
        ]
    )
