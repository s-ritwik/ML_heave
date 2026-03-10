#!/usr/bin/env python3
# ROS2 publisher that streams a trained GRU model on live z measurements and publishes its forecast.
# Input x is built exactly like mca_prediction_publisher: x = (z_uav - 2.8479 - (z_uav - z_plat)) / scale.

import os
import re
import pickle
from typing import Optional
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


def parse_model_path(model_path: str, name_rate_hz: int = 20):
    """Parse seq/output/hidden sizes from any path containing 'GRU_<seq>_<out>_<hidden>'. """
    matches = re.findall(r"GRU_(\d+)_(\d+)_([0-9_]+)", model_path)
    if not matches:
        raise ValueError(f"Could not parse seq/out/hidden from path {model_path}")
    seq_str, out_str, hid_str = matches[-1]  # take the last occurrence
    seq_sec = int(seq_str)
    out_sec = int(out_str)
    hidden_sizes = [int(h) for h in hid_str.split('_') if h.strip() != ""]
    seq_len = seq_sec * name_rate_hz
    out_len = out_sec * name_rate_hz
    return seq_len, out_len, hidden_sizes


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        h_out = []
        out = x
        for i, gru in enumerate(self.gru_layers):
            out, h_i = gru(out, h[i])
            h_out.append(h_i)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.tanh(out)
        return out, h_out

    def init_hidden(self, batch_size, device):
        return [torch.zeros(1, batch_size, hs, device=device) for hs in self.hidden_sizes]


class GRUPredictionPublisher(Node):
    def __init__(self):
        super().__init__('gru_prediction_publisher')

        default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'noisyGRU_models_seq',
                                          'noisy_D1_GRU_40_6_1024_1024_ritwik_trained_backup')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('uav_pose_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('platform_pose_topic', '/qualisys/ship_deck_platform/pose')
        self.declare_parameter('prediction_topic', '/stewart/prediction')
        self.declare_parameter('hz', 20.0)
        self.declare_parameter('z_bias', 2.03) # 2.8479 for gazebo, 2.03 for exp
        self.declare_parameter('scale_factor', 0.35)
        self.declare_parameter('epoch', 480)

        self.model_path = self.get_parameter('model_path').value
        epoch_val = int(self.get_parameter('epoch').value) if self.get_parameter('epoch').value is not None else -1
        self.model_epoch = epoch_val if epoch_val >= 0 else None
        self.uav_pose_topic = self.get_parameter('uav_pose_topic').value
        self.platform_pose_topic = self.get_parameter('platform_pose_topic').value
        self.pred_topic = self.get_parameter('prediction_topic').value
        self.hz = float(self.get_parameter('hz').value)
        self.z_bias = float(self.get_parameter('z_bias').value)
        self.scale_factor = float(self.get_parameter('scale_factor').value)

        # Resolve model: allow folder path → pick latest .pth/.pt
        resolved_path = self._resolve_model_path(self.model_path, self.model_epoch)

        # Parse model info and load
        seq_len, out_len, hidden_sizes = parse_model_path(resolved_path)
        self.seq_len = seq_len
        self.out_len = out_len
        self.device = torch.device('cpu')  # lightweight inference

        self.model = GRUModel(input_size=1, hidden_sizes=hidden_sizes, output_size=self.out_len).to(self.device)
        self._load_weights(resolved_path)
        self.model.eval()
        self.h = self.model.init_hidden(batch_size=1, device=self.device)

        # Rolling history (not directly fed; we stream via hidden state but keep for optional debugging)
        self.sample_count = 0

        # State
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
            f"GRU prediction publisher started. Model={resolved_path}, seq_len={self.seq_len}, out_len={self.out_len}, hz={self.hz}")

    def _load_weights(self, path: str):
        last_err = None
        for kwargs in (
            dict(weights_only=False, pickle_module=pickle),
            dict(weights_only=False),
        ):
            try:
                sd = torch.load(path, map_location=self.device, **kwargs)
                if isinstance(sd, dict) and any(k in sd for k in ('state_dict', 'model_state_dict')):
                    if 'model_state_dict' in sd:
                        sd = sd['model_state_dict']
                    elif 'state_dict' in sd:
                        sd = sd['state_dict']
                self.model.load_state_dict(sd)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to load checkpoint {path}: {last_err}")

    def _resolve_model_path(self, path_in: str, epoch: Optional[int]) -> str:
        """If given a directory, pick .pth/.pt file (epoch-specific if provided, else latest); else return the file."""
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
                # if requested epoch missing, fall back to latest
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

        # Single-step streaming with hidden state
        x_tensor = torch.tensor([[[x_val]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            yhat, self.h = self.model(x_tensor, self.h)
            self.h = [h_i.detach() for h_i in self.h]
            y_np = yhat.cpu().numpy().flatten()
            y_np = y_np * self.scale_factor + self.z_bias

        msg = Float32MultiArray()
        msg.data = y_np.astype(float).tolist()
        self.pred_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GRUPredictionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
