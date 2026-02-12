#!/usr/bin/env python3
# Live plot /stewart/prediction vs /heave_predicted_true (scaled prediction).

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.rcParams['figure.raise_window'] = False  # do not steal focus
import matplotlib.pyplot as plt
plt.ion()  # interactive mode for live updates


class LivePredictionPlotter(Node):
    def __init__(self):
        super().__init__('live_prediction_plotter')

        self.declare_parameter('prediction_topic', '/stewart/prediction')
        self.declare_parameter('truth_topic', '/heave_predicted_true')
        self.declare_parameter('scale_factor', 1)
        self.declare_parameter('z_bias', 2.8479)
        self.declare_parameter('hz', 5.0)  # plot refresh

        self.pred_topic = self.get_parameter('prediction_topic').value
        self.truth_topic = self.get_parameter('truth_topic').value
        self.scale = float(self.get_parameter('scale_factor').value)
        self.z_bias = float(self.get_parameter('z_bias').value)
        self.hz = float(self.get_parameter('hz').value)

        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.last_pred = None
        self.last_truth = None

        self.create_subscription(Float32MultiArray, self.pred_topic, self._cb_pred, qos_best_effort)
        self.create_subscription(Float32MultiArray, self.truth_topic, self._cb_truth, qos_best_effort)

        dt = 1.0 / self.hz if self.hz > 0 else 0.2
        self.timer = self.create_timer(dt, self._tick)

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))
        self.pred_line, = self.ax.plot([], [], 'r-', label='Prediction (scaled)')
        self.truth_line, = self.ax.plot([], [], 'b--', label='Heave true')
        self.ax.set_xlabel('Horizon index')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw_idle()

        self.get_logger().info(
            f"Live plotter listening on pred={self.pred_topic}, truth={self.truth_topic}, scale={self.scale}")

    def _cb_pred(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data, dtype=float)
            self.last_pred = arr * self.scale
        except Exception as e:
            self.get_logger().warn(f'Pred parse failed: {e}')

    def _cb_truth(self, msg: Float32MultiArray):
        try:
            self.last_truth = np.array(msg.data, dtype=float) * self.scale + self.z_bias
        except Exception as e:
            self.get_logger().warn(f'Truth parse failed: {e}')

    def _tick(self):
        if self.last_pred is None and self.last_truth is None:
            return

        x_pred = np.arange(len(self.last_pred)) if self.last_pred is not None else None
        x_truth = np.arange(len(self.last_truth)) if self.last_truth is not None else None

        if self.last_pred is not None:
            self.pred_line.set_data(x_pred, self.last_pred)
        if self.last_truth is not None:
            self.truth_line.set_data(x_truth, self.last_truth)

        # update axes limits
        all_y = []
        if self.last_pred is not None:
            all_y.extend(self.last_pred.tolist())
        if self.last_truth is not None:
            all_y.extend(self.last_truth.tolist())
        if all_y:
            ymin = min(all_y) - 0.1
            ymax = max(all_y) + 0.1
            self.ax.set_ylim(2, 3.5)

        xmax = 0
        if x_pred is not None:
            xmax = max(xmax, x_pred[-1])
        if x_truth is not None:
            xmax = max(xmax, x_truth[-1])
        self.ax.set_xlim(0, max(xmax, 1))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = LivePredictionPlotter()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    plt.show(block=False)
    try:
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            executor.spin_once(timeout_sec=0.05)
            plt.pause(0.01)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        plt.ioff()
        plt.close('all')


if __name__ == '__main__':
    main()
