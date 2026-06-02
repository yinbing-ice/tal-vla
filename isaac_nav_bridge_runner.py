#!/usr/bin/env python3
from __future__ import annotations

# 2026-05-30 新增：ROS 侧 Isaac 导航桥接进程，负责把 Isaac 中的底盘状态转成 /clock /odom /scan /tf，并接收 Nav2 输出的 /cmd_vel。

import argparse
import json
import math
import socket
from typing import Any

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS-side Isaac navigation bridge runner.")
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--cmd-port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser


class IsaacNavBridgeRunner:
    def __init__(self, host: str, state_port: int, cmd_port: int) -> None:
        self._node = Node("isaac_nav_bridge")
        self._node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self._host = host
        self._cmd_port = cmd_port
        self._cmd_vx = 0.0
        self._cmd_vw = 0.0
        self._latest_sender: tuple[str, int] | None = None

        self._clock_pub = self._node.create_publisher(Clock, "/clock", 10)
        self._odom_pub = self._node.create_publisher(Odometry, "/odom", 10)
        self._scan_pub = self._node.create_publisher(LaserScan, "/scan", 10)
        self._tf_pub = self._node.create_publisher(TFMessage, "/tf", 10)
        self._tf_broadcaster = TransformBroadcaster(self._node)
        self._node.create_subscription(Twist, "/cmd_vel_nav", self._cmd_callback, 10)
        self._node.create_subscription(Twist, "/cmd_vel", self._cmd_callback, 10)

        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_socket.bind((host, state_port))
        self._state_socket.setblocking(False)

    def _cmd_callback(self, msg: Twist) -> None:
        self._cmd_vx = float(msg.linear.x)
        self._cmd_vw = float(msg.angular.z)
        payload = json.dumps({"vx": self._cmd_vx, "vw": self._cmd_vw}, separators=(",", ":")).encode("utf-8")
        try:
            self._state_socket.sendto(payload, (self._host, self._cmd_port))
        except OSError:
            pass

    def _stamp_msg(self, sim_time_s: float) -> TimeMsg:
        stamp = TimeMsg()
        stamp.sec = int(sim_time_s)
        stamp.nanosec = int((sim_time_s - stamp.sec) * 1e9)
        return stamp

    def _yaw_quaternion_msg(self, yaw: float) -> Quaternion:
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion

    def _publish_state(self, payload: dict[str, Any]) -> None:
        sim_time_s = float(payload["sim_time_s"])
        x = float(payload["x"])
        y = float(payload["y"])
        z = float(payload["z"])
        yaw = float(payload["yaw"])
        vx = float(payload.get("vx", 0.0))
        vw = float(payload.get("vw", 0.0))
        stamp = self._stamp_msg(sim_time_s)

        clock_msg = Clock()
        clock_msg.clock = stamp
        self._clock_pub.publish(clock_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.pose.pose.orientation = self._yaw_quaternion_msg(yaw)
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.angular.z = vw
        self._odom_pub.publish(odom_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation = self._yaw_quaternion_msg(yaw)
        self._tf_pub.publish(TFMessage(transforms=[tf_msg]))
        self._tf_broadcaster.sendTransform(tf_msg)

        scan_ranges = payload.get("scan_ranges")
        scan_msg = LaserScan()
        scan_msg.header.stamp = stamp
        scan_msg.header.frame_id = "base_scan"
        scan_msg.angle_min = float(payload.get("scan_angle_min", -math.pi / 2.0))
        scan_msg.angle_max = float(payload.get("scan_angle_max", math.pi / 2.0))
        scan_msg.angle_increment = float(payload.get("scan_angle_increment", math.pi / 180.0))
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = float(payload.get("scan_range_min", 0.12))
        scan_msg.range_max = float(payload.get("scan_range_max", 3.5))
        if isinstance(scan_ranges, list) and len(scan_ranges) > 0:
            scan_msg.ranges = [float(v) for v in scan_ranges]
        else:
            beam_count = max(1, int(round((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)) + 1)
            scan_msg.ranges = [3.0] * beam_count
        scan_msg.intensities = [0.0] * len(scan_msg.ranges)
        self._scan_pub.publish(scan_msg)

    def spin(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)
            while True:
                try:
                    packet, sender = self._state_socket.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    return
                self._latest_sender = sender
                payload = json.loads(packet.decode("utf-8"))
                self._publish_state(payload)

    def close(self) -> None:
        try:
            self._state_socket.close()
        except OSError:
            pass
        self._node.destroy_node()


def main() -> int:
    args = _build_argparser().parse_args()
    rclpy.init(args=None)
    runner = IsaacNavBridgeRunner(args.host, args.state_port, args.cmd_port)
    try:
        runner.spin()
    finally:
        runner.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
