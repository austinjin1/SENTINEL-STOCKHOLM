"""Ground-computer launch (Omen 15): WaterDroneNet inference node.

    ros2 launch sentinel_inference computer.launch.py
    ros2 launch sentinel_inference computer.launch.py device:=cuda checkpoint:=/path/to.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    device = DeclareLaunchArgument("device", default_value="cpu")
    checkpoint = DeclareLaunchArgument("checkpoint", default_value="")
    nir_topic = DeclareLaunchArgument("nir_topic", default_value="")

    config_file = PathJoinSubstitution(
        [FindPackageShare("sentinel_inference"), "config", "inference_params.yaml"]
    )

    node = Node(
        package="sentinel_inference",
        executable="inference",
        name="sentinel_inference",
        output="screen",
        parameters=[config_file, {
            "device": LaunchConfiguration("device"),
            "checkpoint": LaunchConfiguration("checkpoint"),
            "nir_topic": LaunchConfiguration("nir_topic"),
        }],
    )

    return LaunchDescription([device, checkpoint, nir_topic, node])
