from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_config = str(
        Path(get_package_share_directory("nick_safe_ctrl"))
        / "config"
        / "safe_ctrl.yaml"
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to the safe_ctrl_node parameter YAML.",
    )

    node = Node(
        package="nick_safe_ctrl",
        executable="safe_ctrl_node",
        name="safe_ctrl_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, node])
