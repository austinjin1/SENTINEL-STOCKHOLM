"""Drone-side launch (Raspberry Pi): RGB camera + optical TF.

    ros2 launch sentinel_camera drone.launch.py
    ros2 launch sentinel_camera drone.launch.py backend:=synthetic   # no camera

NIR camera is added here later (its own node + TF) once calibrated.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend = DeclareLaunchArgument(
        "backend", default_value="auto",
        description="auto|libcamera|opencv|synthetic",
    )
    config_file = PathJoinSubstitution(
        [FindPackageShare("sentinel_camera"), "config", "camera_params.yaml"]
    )

    rgb_node = Node(
        package="sentinel_camera",
        executable="rgb_camera",
        name="sentinel_rgb_camera",
        output="screen",
        parameters=[config_file, {"backend": LaunchConfiguration("backend")}],
    )

    # base_link -> camera_optical_frame (translation only; orientation TODO from mount)
    optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_tf",
        arguments=["0", "0", "0.1", "0", "0", "0", "1",
                   "base_link", "camera_optical_frame"],
    )

    return LaunchDescription([backend, rgb_node, optical_tf])
