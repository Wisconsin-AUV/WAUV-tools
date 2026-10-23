#!/usr/bin/env python3
"""ROS 2 bridge: nav_msgs/Odometry -> JSON datagram over UDP.

Sends one datagram per message, in the format odom_viz_udp.py expects:
    {"x": 1.0, "y": 2.0, "z": 0.5, "t": 12.34}

Usage:
    python3 odom_to_udp.py --topic /odom --host 127.0.0.1 --port 9870
"""
import argparse
import json
import socket

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args


class OdomToUdp(Node):
    def __init__(self, topic: str, host: str, port: int):
        super().__init__("odom_to_udp")
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.create_subscription(Odometry, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"{topic} -> udp://{host}:{port}")

    def cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:  # unstamped (some sims); fall back to node clock
            t = self.get_clock().now().nanoseconds * 1e-9
        p = msg.pose.pose.position
        payload = json.dumps({"x": p.x, "y": p.y, "z": p.z, "t": t})
        try:
            self.sock.sendto(payload.encode(), self.addr)
        except OSError as e:
            self.get_logger().warn(f"UDP send failed: {e}", throttle_duration_sec=2.0)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/visual_slam/tracking/odometry")
    ap.add_argument("--host", default="192.168.137.1", help="destination IP")
    ap.add_argument("--port", type=int, default=9870)
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = OdomToUdp(args.topic, args.host, args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()