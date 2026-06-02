import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


LOCALIZATION_MANAGER_DELAY_SEC = 2.0
NAVIGATION_STACK_DELAY_SEC = 5.0
ISAAC_NAVIGATION_STACK_DELAY_SEC = 40.0


def _mode_equals_expression(mode: LaunchConfiguration, expected_mode: str) -> PythonExpression:
    return PythonExpression(["'", mode, "' == '", expected_mode, "'"])


def _read_robot_description(robot_navigation_dir: str) -> str:
    urdf_file_path = os.path.join(robot_navigation_dir, "urdf", "simple_robot.urdf")
    with open(urdf_file_path, "r", encoding="utf-8") as file_obj:
        return file_obj.read()


def generate_launch_description():
    robot_navigation_dir = get_package_share_directory("robot_navigation")

    mode = LaunchConfiguration("mode", default="isaac")
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    map_yaml_path = LaunchConfiguration(
        "map",
        default=os.path.join(robot_navigation_dir, "maps", "expff_map.yaml"),
    )
    nav2_param_path = LaunchConfiguration(
        "params_file",
        default=os.path.join(robot_navigation_dir, "config", "nav2_params.yaml"),
    )

    robot_description = _read_robot_description(robot_navigation_dir)

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": use_sim_time,
        }],
    )

    joint_state_pub = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "publish_default_positions": True,
        }],
    )

    fake_mode_condition = IfCondition(_mode_equals_expression(mode, "fake"))
    real_mode_condition = IfCondition(_mode_equals_expression(mode, "real"))
    isaac_mode_condition = IfCondition(_mode_equals_expression(mode, "isaac"))
    non_isaac_mode_condition = IfCondition(PythonExpression(["'", mode, "' != 'isaac'"]))

    fake_odom_node = Node(
        package="fake_nav",
        executable="fake_odom",
        name="fake_odom",
        output="screen",
        condition=fake_mode_condition,
    )
    fake_scan_node = Node(
        package="fake_nav",
        executable="fake_scan",
        name="fake_scan",
        output="screen",
        condition=fake_mode_condition,
    )

    real_odom_node = Node(
        package="real_robot_driver",
        executable="odom_driver",
        name="odom_driver",
        output="screen",
        condition=real_mode_condition,
    )
    real_lidar_node = Node(
        package="real_robot_driver",
        executable="lidar_driver",
        name="lidar_driver",
        output="screen",
        condition=real_mode_condition,
    )

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "yaml_filename": map_yaml_path,
            "use_sim_time": use_sim_time,
        }],
    )
    map_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_server"],
        }],
    )
    map_to_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    nav2_remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    lifecycle_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
    ]

    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[nav2_param_path],
        remappings=nav2_remappings + [("cmd_vel", "cmd_vel_nav")],
    )
    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[nav2_param_path],
        remappings=nav2_remappings,
    )
    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[nav2_param_path],
        remappings=nav2_remappings,
    )
    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[nav2_param_path],
        remappings=nav2_remappings,
    )
    navigation_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": lifecycle_nodes,
        }],
    )

    navigation_stack = GroupAction(
        actions=[
            controller_server,
            planner_server,
            behavior_server,
            bt_navigator,
            navigation_lifecycle_manager,
        ]
    )
    localization_manager_delayed = TimerAction(
        period=LOCALIZATION_MANAGER_DELAY_SEC,
        actions=[map_lifecycle_manager],
    )
    navigation_stack_now = TimerAction(
        period=NAVIGATION_STACK_DELAY_SEC,
        actions=[navigation_stack],
        condition=non_isaac_mode_condition,
    )
    navigation_stack_delayed = TimerAction(
        period=ISAAC_NAVIGATION_STACK_DELAY_SEC,
        actions=[navigation_stack],
        condition=isaac_mode_condition,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="isaac",
            description="Navigation I/O mode: fake, real, or isaac.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time.",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(robot_navigation_dir, "maps", "expff_map.yaml"),
            description="Path to the occupancy map YAML.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(robot_navigation_dir, "config", "nav2_params.yaml"),
            description="Path to the Nav2 parameter file.",
        ),
        robot_state_pub,
        joint_state_pub,
        fake_odom_node,
        fake_scan_node,
        real_odom_node,
        real_lidar_node,
        map_server_node,
        map_to_odom_tf,
        localization_manager_delayed,
        navigation_stack_now,
        navigation_stack_delayed,
    ])
