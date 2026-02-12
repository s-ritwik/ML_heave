#!/usr/bin/env python3
# ROS2 publisher that streams an LSTM model on live z measurements and publishes its forecast.
# Input x built exactly like the MCA/GRU publishers: x = (z_uav - 2.8479 - (z_uav - z_plat)) / scale.

import os
import re
import numpy as np
import torch
import torch.nn as nn
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

NAME_RATE_HZ = 20  # fields in filename are seconds at this rate


def parse_model_path(model_path: str):
    """Parse seq/output/hidden sizes from names like ..._LSTM_<seq>_<out>_<hidden>[...]"""
    fname = os.path.basename(model_path)
    m = re.search(r"_LSTM_(\d+)_(\d+)_([0-9_]+)(?:_in\d+)?\.(pt|pth)$", fname)
    if not m:
        raise ValueError(f"Model filename {fname} does not match '*_LSTM_<seq>_<out>_<hidden>[_inX].pt(pth)'.")
    seq_len = int(m.group(1)) * NAME_RATE_HZ
    out_len = int(m.group(2)) * NAME_RATE_HZ
    hidden_sizes = list(map(int, m.group(3).split('_')))
    return seq_len, out_len, hidden_sizes


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, state):
        next_state = []
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            h_i, c_i = state[i]
            out, (h_o, c_o) = lstm(out, (h_i, c_i))
            next_state.append((h_o, c_o))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.tanh(out)
        return out, next_state

    def init_state(self, batch_size, device):
        return [(
            torch.zeros(1, batch_size, hs, device=device),
            torch.zeros(1, batch_size, hs, device=device)
        ) for hs in self.hidden_sizes]


class LSTMPredictionPublisher(Node):
    def __init__(self):
        super().__init__('lstm_prediction_publisher')

        default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'noisyLSTM_models_seq',
                                          'noisy_D1_LSTM_40_6_1024_1024_in1',
                                          'epoch_200.pth')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('uav_pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('platform_pose_topic', '/stewart/top_platform_pose')
        self.declare_parameter('prediction_topic', '/stewart/prediction_lstm')
        self.declare_parameter('hz', 20.0)
        self.declare_parameter('z_bias', 2.8479)
        self.declare_parameter('scale_factor', 0.5)
        self.declare_parameter('epoch', None)

        self.model_path = self.get_parameter('model_path').value
        self.model_epoch = self.get_parameter('epoch').value
        self.uav_pose_topic = self.get_parameter('uav_pose_topic').value
        self.platform_pose_topic = self.get_parameter('platform_pose_topic').value
        self.pred_topic = self.get_parameter('prediction_topic').value
        self.hz = float(self.get_parameter('hz').value)
        self.z_bias = float(self.get_parameter('z_bias').value)
        self.scale_factor = float(self.get_parameter('scale_factor').value)

        resolved_path = self._resolve_model_path(self.model_path, self.model_epoch)

        seq_len, out_len, hidden_sizes = parse_model_path(resolved_path)
        self.seq_len = seq_len
        self.out_len = out_len
        self.device = torch.device('cpu')

        self.model = LSTMModel(input_size=1, hidden_sizes=hidden_sizes, output_size=self.out_len).to(self.device)
        state_dict = torch.load(resolved_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.state = self.model.init_state(batch_size=1, device=self.device)

        self.last_uav_z = None
        self.last_plat_z = None

        io_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pred_pub = self.create_publisher(Float32MultiArray, self.pred_topic, 10)
        self.create_subscription(PoseStamped, self.uav_pose_topic, self._cb_uav, io_qos)
        self.create_subscription(PoseStamped, self.platform_pose_topic, self._cb_plat, io_qos)

        dt = 1.0 / self.hz if self.hz > 0 else 0.05
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(
            f"LSTM prediction publisher started. Model={resolved_path}, seq_len={self.seq_len}, out_len={self.out_len}, hz={self.hz}")

    def _resolve_model_path(self, path_in: str, epoch: Optional[int]) -> str:
        if os.path.isdir(path_in):
            candidates = []
            for fname in os.listdir(path_in):
                if fname.endswith('.pth') or fname.endswith('.pt'):
                    full = os.path.join(path_in, fname)
                    m = re.search(r'epoch_(\d+)\.(pt|pth)$', fname)
                    ep = int(m.group(1)) if m else 0
                    ext_rank = 0 if fname.endswith('.pth') else 1
                    mtime = os.path.getmtime(full)
                    candidates.append((ep, mtime, ext_rank, full))
            if not candidates:
                raise FileNotFoundError(f"No .pth/.pt files found in directory {path_in}")
            if epoch is not None:
                filtered = [(ep, mt, er, fp) for (ep, mt, er, fp) in candidates if ep == int(epoch)]
                if filtered:
                    filtered.sort()
                    return filtered[-1][3]
            candidates.sort()
            return candidates[-1][3]
        if os.path.isfile(path_in):
            return path_in
        raise FileNotFoundError(f"Model path {path_in} not found")

    def _cb_uav(self, msg: PoseStamped):
        self.last_uav_z = msg.pose.position.z

    def _cb_plat(self, msg: PoseStamped):
        self.last_plat_z = msg.pose.position.z

    def _tick(self):
        if self.last_uav_z is None or self.last_plat_z is None:
            return

        del_z = self.last_uav_z - self.last_plat_z
        x_val = (self.last_uav_z - self.z_bias - del_z) / self.scale_factor

        x_tensor = torch.tensor([[[x_val]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            yhat, self.state = self.model(x_tensor, self.state)
            self.state = [(h.detach(), c.detach()) for (h, c) in self.state]
            y_np = yhat.cpu().numpy().flatten()
            y_np = y_np * self.scale_factor + self.z_bias

        msg = Float32MultiArray()
        msg.data = y_np.astype(float).tolist()
        self.pred_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LSTMPredictionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
