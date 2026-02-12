#!/usr/bin/env python3
# Publish MCA linear predictions in ROS2 at 20 Hz using the saved model.
# Uses the same loading/prediction routines as tester_mca_mp4.py / mca_brute.py.

import os
import json
import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


def load_mca_model(model_dir: str):
    """Load W, mu, config (and infer n, m, center, col_index)."""
    W_path = os.path.join(model_dir, "W.npy")
    mu_path = os.path.join(model_dir, "mu.npy")
    cfg_path = os.path.join(model_dir, "config.json")
    if not (os.path.exists(W_path) and os.path.exists(mu_path) and os.path.exists(cfg_path)):
        raise FileNotFoundError("Model folder must contain W.npy, mu.npy, and config.json")

    W = np.load(W_path)           # (m x n)
    mu = np.load(mu_path)         # (n+m,)
    with open(cfg_path, "r") as f:
        config = json.load(f)

    n = int(config["n"])
    m = int(config["m"])
    center = bool(config.get("center", True))
    col_index = int(config.get("col_index", 0))
    return W, mu, config, n, m, center, col_index


def predict_once(W: np.ndarray, x1: np.ndarray, mu: np.ndarray, n: int, m: int, centered: bool) -> np.ndarray:
    """
    x1: shape (n,)
    return: yhat shape (m,)
    """
    if centered:
        mu1 = mu[:n]
        mu2 = mu[n:]
        x1c = x1 - mu1
        y2c = W @ x1c
        return y2c + mu2
    else:
        return W @ x1


class MCAPredictionPublisher(Node):
    def __init__(self):
        super().__init__('mca_prediction_publisher')

        default_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'models',
                                         'MCA_n800_m120_stride1_P906_cut0.01_ridge1e-06')

        # Parameters
        self.declare_parameter('model_dir', default_model_dir)
        self.declare_parameter('uav_pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('platform_pose_topic', '/stewart/top_platform_pose')
        self.declare_parameter('prediction_topic', '/stewart/prediction')
        self.declare_parameter('hz', 20.0)
        self.declare_parameter('z_bias', 2.8479)
        self.declare_parameter('scale_factor', 0.4)

        self.model_dir = self.get_parameter('model_dir').value
        self.uav_pose_topic = self.get_parameter('uav_pose_topic').value
        self.platform_pose_topic = self.get_parameter('platform_pose_topic').value
        self.pred_topic = self.get_parameter('prediction_topic').value
        self.hz = float(self.get_parameter('hz').value)
        self.z_bias = float(self.get_parameter('z_bias').value)
        self.scale_factor = float(self.get_parameter('scale_factor').value)

        # Load model
        W, mu, cfg, n, m, centered, _ = load_mca_model(self.model_dir)
        self.W = W
        self.mu = mu
        self.n = n
        self.m = m
        self.centered = centered

        # Rolling buffer (pre-filled with zeros to length n)
        self.history = deque([0.0] * self.n, maxlen=self.n)

        # State
        self.last_uav_z: Optional[float] = None
        self.last_plat_z: Optional[float] = None

        io_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Pub/Sub
        self.pred_pub = self.create_publisher(Float32MultiArray, self.pred_topic, 10)
        self.create_subscription(PoseStamped, self.uav_pose_topic, self._cb_uav_pose, io_qos)
        self.create_subscription(PoseStamped, self.platform_pose_topic, self._cb_platform_pose, io_qos)

        dt = 1.0 / self.hz if self.hz > 0 else 0.05
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(
            f"MCA prediction publisher started. Model={self.model_dir}, n={self.n}, m={self.m}, hz={self.hz}")

    def _cb_uav_pose(self, msg: PoseStamped):
        self.last_uav_z = msg.pose.position.z

    def _cb_platform_pose(self, msg: PoseStamped):
        self.last_plat_z = msg.pose.position.z

    def _tick(self):
        if self.last_uav_z is None or self.last_plat_z is None:
            return

        del_z = self.last_uav_z - self.last_plat_z
        x_val = self.last_uav_z - self.z_bias - del_z
        x_val = x_val / self.scale_factor

        # Update buffer
        self.history.append(x_val)
        x_arr = np.array(self.history, dtype=float)

        try:
            yhat = predict_once(self.W, x_arr, mu=self.mu, n=self.n, m=self.m, centered=self.centered)
        except Exception as e:
            self.get_logger().warn(f'MCA prediction failed: {e}')
            return

        msg = Float32MultiArray()
        msg.data = yhat.astype(float).tolist()
        self.pred_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MCAPredictionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
