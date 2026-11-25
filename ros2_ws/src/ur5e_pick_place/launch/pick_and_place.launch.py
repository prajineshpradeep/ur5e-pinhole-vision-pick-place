"""Unified ROS 2 launch file starting MuJoCo, MoveIt 2, eye-in-hand vision, and C++ pick-and-place."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")

    moveit_pkg = FindPackageShare("ur5e_moveit_config")

    mujoco_node = Node(
        package="ur5e_mujoco_sim",
        executable="mujoco_node",
        output="screen",
        parameters=[{"headless": headless}],
    )

    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([moveit_pkg, "launch", "rsp.launch.py"])
        )
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([moveit_pkg, "launch", "move_group.launch.py"])
        )
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([moveit_pkg, "launch", "moveit_rviz.launch.py"])
        ),
        condition=IfCondition(use_rviz),
    )

    vision_node = Node(
        package="ur5e_mujoco_sim",
        executable="vision_node",
        output="screen",
    )

    pick_place_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="ur5e_pick_place",
                executable="pick_place_node",
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            mujoco_node,
            rsp_launch,
            move_group_launch,
            rviz_launch,
            vision_node,
            pick_place_node,
        ]
    )
