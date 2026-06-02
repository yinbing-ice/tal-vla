#!/usr/bin/env python3
from __future__ import annotations

# 2026-05-30 新增：独立子进程发送单次 NavigateToPose 目标，并把 Nav2 结果序列化回主控制脚本。

import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a single Nav2 NavigateToPose goal and wait for the result.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--frame-id", type=str, default="map")
    parser.add_argument("--server-timeout", type=float, default=10.0)
    parser.add_argument("--result-timeout", type=float, default=120.0)
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _spin_until_future(node: Node, future: object, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if future.done():
            return True
    return future.done()


def _wait_for_nav2_active(node: Node, timeout_sec: float) -> tuple[bool, str]:
    client = node.create_client(GetState, "/bt_navigator/get_state")
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if not client.wait_for_service(timeout_sec=0.5):
                continue
            future = client.call_async(GetState.Request())
            if not _spin_until_future(node, future, timeout_sec=2.0):
                continue
            response = future.result()
            if response is None:
                continue
            if response.current_state.id == State.PRIMARY_STATE_ACTIVE:
                return True, response.current_state.label
            time.sleep(0.2)
        return False, "bt_navigator did not reach ACTIVE state before timeout."
    finally:
        node.destroy_client(client)


def _send_goal_once(
    node: Node,
    client: ActionClient,
    goal: NavigateToPose.Goal,
    timeout_sec: float,
) -> tuple[bool, str | None, object | None]:
    send_future = client.send_goal_async(goal)
    if not _spin_until_future(node, send_future, timeout_sec=timeout_sec):
        return False, "Timed out while sending Nav2 goal.", None
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        return False, "Nav2 goal was rejected.", None
    return True, None, goal_handle


def main() -> int:
    args = _build_argparser().parse_args()
    rclpy.init(args=None)
    node = Node("nav2_goal_runner")
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")

    try:
        active_ready, active_detail = _wait_for_nav2_active(node, timeout_sec=args.server_timeout)
        if not active_ready:
            _emit({"success": False, "error": active_detail})
            return 2

        if not client.wait_for_server(timeout_sec=args.server_timeout):
            _emit({"success": False, "error": "NavigateToPose action server is not available."})
            return 3

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = args.frame_id
        pose.pose.position.x = float(args.x)
        pose.pose.position.y = float(args.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(float(args.yaw) / 2.0)
        pose.pose.orientation.w = math.cos(float(args.yaw) / 2.0)
        goal.pose = pose

        sent, error, goal_handle = _send_goal_once(node, client, goal, timeout_sec=args.server_timeout)
        if not sent:
            time.sleep(1.0)
            active_ready, active_detail = _wait_for_nav2_active(node, timeout_sec=5.0)
            if not active_ready:
                _emit({"success": False, "error": active_detail})
                return 4
            sent, error, goal_handle = _send_goal_once(node, client, goal, timeout_sec=args.server_timeout)
            if not sent:
                _emit({"success": False, "error": error, "detail": "goal_send_retry_exhausted"})
                return 5

        result_future = goal_handle.get_result_async()
        if not _spin_until_future(node, result_future, timeout_sec=args.result_timeout):
            goal_handle.cancel_goal_async()
            _emit({"success": False, "error": "Timed out waiting for Nav2 result."})
            return 6

        result = result_future.result()
        if result is None:
            _emit({"success": False, "error": "Nav2 returned no result."})
            return 7

        status = int(result.status)
        success = status == 4
        payload = {"success": success, "status": status}
        if not success:
            payload["error"] = f"Nav2 goal finished with status={status}."
        _emit(payload)
        return 0 if success else 8
    except Exception as exc:  # noqa: BLE001
        _emit({"success": False, "error": str(exc)})
        return 10
    finally:
        try:
            client.destroy()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
