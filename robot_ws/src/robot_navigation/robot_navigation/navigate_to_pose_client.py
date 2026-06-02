from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


@dataclass
class NavigationGoal:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"


class NavigateToPoseClient(Node):
    """Minimal Nav2 goal client for integration with TAL-VLA."""

    def __init__(self) -> None:
        super().__init__("navigate_to_pose_client")
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_sec)

    def _build_goal_msg(self, goal: NavigationGoal) -> NavigateToPose.Goal:
        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = goal.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(goal.x)
        pose.pose.position.y = float(goal.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(goal.yaw / 2.0)
        pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        goal_msg.pose = pose
        return goal_msg

    def navigate_to_goal(self, goal: NavigationGoal, timeout_sec: float = 120.0) -> bool:
        if not self.wait_for_server():
            self.get_logger().error("NavigateToPose action server is not available.")
            return False

        goal_msg = self._build_goal_msg(goal)
        self.get_logger().info(
            "Sending Nav2 goal: x=%.3f y=%.3f yaw=%.3f frame=%s"
            % (goal.x, goal.y, goal.yaw, goal.frame_id)
        )
        send_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout_sec)
        if not send_future.done():
            self.get_logger().error("Timed out while sending Nav2 goal.")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Nav2 goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        if not result_future.done():
            self.get_logger().error("Timed out while waiting for Nav2 result.")
            return False

        result = result_future.result()
        if result is None:
            self.get_logger().error("Nav2 returned no result.")
            return False

        status = int(result.status)
        success = status == 4
        if success:
            self.get_logger().info("Nav2 goal reached successfully.")
        else:
            self.get_logger().warning(f"Nav2 goal finished with status={status}.")
        return success


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a NavigateToPose goal to Nav2.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--frame-id", type=str, default="map")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init(args=None)
    node = NavigateToPoseClient()
    try:
        success = node.navigate_to_goal(
            NavigationGoal(
                x=args.x,
                y=args.y,
                yaw=args.yaw,
                frame_id=args.frame_id,
            ),
            timeout_sec=args.timeout_sec,
        )
        raise SystemExit(0 if success else 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
