# -*- coding: utf-8 -*-
"""
Noisy GRU training & evaluation with periodic checkpointing and file logging.

Changes vs. original:
- Creates a per-model folder under `noisyGRU_models_seq/<model_name>/`
- Saves model checkpoints every `SAVE_EVERY` epochs (default: 20)
- Writes all logs (no prints) to:
    - Global run hyperparams:        log/run_hyperparams.txt
    - Per-model training/testing:    log/<model_name>.log
    - Summary of metrics:            model_summary.txt   (appends)
- Logs epoch, loss, elapsed time, LR every SAVE_EVERY epochs
- Records GPU selection, noise std, video settings, and constants in logs
- Implements prediction timing during test and includes it in summary
"""

import os
import re
import json
import time
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler  # (import preserved)
# -------------------------------------------------------------------------
# -------------------------- GLOBAL CONSTANTS ------------------------------
# -------------------------------------------------------------------------

# Root directories
MODEL_ROOT_DIR = 'noisyGRU_models_seq'
PLOT_DIR       = 'noisyprediction_videos'
LOG_DIR        = 'log'

# Ensure directories exist
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# GPU selection (0 or 1)
GPU_INDEX = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global data/config files
CONFIG_FILE_PATH = 'model_configs_seq2.txt'
MODEL_SUMMARY_PATH = 'model_summary.txt'

# Training data locations (kept consistent; original code mixed two dirs)
TRAIN_DIR = 'train_data_normalised'
TRAIN_MOCAP_DIR = 'train_data_normalised_mocap'  # used for test file below

# Testing
TEST_FILE_PATH = os.path.join(TRAIN_MOCAP_DIR, 'D1H3_normalised.csv')

# Noise and scaling constants
NOISE_STD_DEFAULT = 0.05        # Gaussian noise std on inputs
METERS_TO_CM      = 25          # conversion factor as used in the original code

# Saving / logging cadence
SAVE_EVERY = 20                 # epochs
LOG_EVERY  = 20                 # epochs (same as SAVE_EVERY per request)

# Video writer settings
VIDEO_FPS = 20
VIDEO_DPI = 100

# -------------------------------------------------------------------------
# -------------------------- UTILITY FUNCTIONS ----------------------------
# -------------------------------------------------------------------------

def now_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def write_global_hyperparams_log():
    """Write global, run-level hyperparameters once at start."""
    run_info = {
        "timestamp": now_str(),
        "gpu_index": GPU_INDEX,
        "torch_device": str(device),
        "noise_std_default": NOISE_STD_DEFAULT,
        "meters_to_cm": METERS_TO_CM,
        "save_every_epochs": SAVE_EVERY,
        "log_every_epochs": LOG_EVERY,
        "video_fps": VIDEO_FPS,
        "video_dpi": VIDEO_DPI,
        "model_root_dir": MODEL_ROOT_DIR,
        "plot_dir": PLOT_DIR,
        "config_file_path": CONFIG_FILE_PATH,
        "train_dir": TRAIN_DIR,
        "train_mocap_dir": TRAIN_MOCAP_DIR,
        "test_file_path": TEST_FILE_PATH,
    }
    with open(os.path.join(LOG_DIR, 'run_hyperparams.txt'), 'a') as f:
        f.write(json.dumps(run_info, indent=2) + '\n')

def parse_config_line(config_line):
    """Parse a single model config line of the form:
       'sequence_length: 400; output_size: 160; hidden_sizes: [512,256]; x_seconds: 3; w: 2; batch_size: 64; epochs: 120; learning_rate: 0.001'
    """
    config = {}
    params = config_line.split(';')
    for param in params:
        key, value = param.split(':')
        key = key.strip()
        value = value.strip()
        if key == 'hidden_sizes':
            config[key] = list(map(int, re.findall(r'\d+', value)))
        elif key in ['sequence_length', 'output_size', 'x_seconds', 'w', 'batch_size', 'epochs']:
            config[key] = int(value)
        elif key == 'learning_rate':
            config[key] = float(value)
    return config

def load_data(file_list, data_dir=TRAIN_MOCAP_DIR):
    """Load 1D sequences from CSV files (first column) and return a single concatenated numpy array."""
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data)

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# -------------------------------------------------------------------------
# ---------------------------- MODEL DEFINITION ---------------------------
# -------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super(GRUModel, self).__init__()
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
        out = out[:, -1, :]           # take last timestep features
        out = self.fc(out)
        out = self.tanh(out)
        return out, h_out
    
    def init_hidden(self, batch_size):
        h = [torch.zeros(1, batch_size, hidden_size).to(device) for hidden_size in self.hidden_sizes]
        return h

# -------------------------------------------------------------------------
# ------------------------------- MAIN FLOW -------------------------------
# -------------------------------------------------------------------------

def main():
    # Log global hyperparams
    write_global_hyperparams_log()

    # Read model configurations
    with open(CONFIG_FILE_PATH, 'r') as f:
        model_configs = [line.strip() for line in f if line.strip()]

    # Load test data (kept as original)
    test_data = pd.read_csv(TEST_FILE_PATH, header=None).iloc[:, 0].values
    meters_to_cm = METERS_TO_CM

    # Prepare model summary file (append mode)
    summary_file = open(MODEL_SUMMARY_PATH, 'a')

    # Iterate through model configurations
    for config_line in model_configs:
        # Parse and prepare config
        config = parse_config_line(config_line)

        sequence_length = config['sequence_length']
        output_size     = config['output_size']
        hidden_sizes    = config['hidden_sizes']
        x_seconds       = config['x_seconds']
        w               = config['w']
        batch_size      = config['batch_size']
        epochs          = config['epochs']
        learning_rate   = config['learning_rate']
        noise_std       = NOISE_STD_DEFAULT  # single global noise hyperparam for all models (as in original code)

        # Build model name and per-model paths
        model_name   = f"noisy_D1_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}"
        model_folder = os.path.join(MODEL_ROOT_DIR, model_name)
        os.makedirs(model_folder, exist_ok=True)

        model_log_path = os.path.join(LOG_DIR, f"{model_name}.log")
        with open(model_log_path, 'a') as mlog:
            # Header for this model
            mlog.write('='*80 + '\n')
            mlog.write(f"[{now_str()}] Starting model: {model_name}\n")
            mlog.write(f"Config: {json.dumps(config)}\n")
            mlog.write(f"Device: {device} | GPU_INDEX={GPU_INDEX}\n")
            mlog.write(f"noise_std={noise_std}, meters_to_cm={meters_to_cm}\n")
            mlog.write(f"Checkpoints every {SAVE_EVERY} epochs | Logs every {LOG_EVERY} epochs\n")
            mlog.write('='*80 + '\n')
            mlog.flush()

        # Initialize model/optimizer/etc.
        model = GRUModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)

        # --------------------- Prepare training data ----------------------
        # Original code listed from 'train_data_normalised', so we read the files there
        csv_files   = os.listdir(TRAIN_DIR)
        train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
        # Fix to load from TRAIN_DIR (original mixed dirs)
        train_data  = load_data(train_files, data_dir=TRAIN_DIR)

        X_train, y_train = [], []
        for i in range(len(train_data) - sequence_length - output_size):
            X_train.append(train_data[i:i + sequence_length])
            y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])

        # X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, seq_len, 1]
        # y_train = torch.tensor(y_train, dtype=torch.float32)                 # [N, output_size]
        X_train = np.asarray(X_train, dtype=np.float32)           # shape: [N, sequence_length]
        X_train = X_train[..., None]                              # add feature dim -> [N, sequence_length, 1]
        y_train = np.asarray(y_train, dtype=np.float32)           # shape: [N, output_size]

        X_train = torch.from_numpy(np.ascontiguousarray(X_train)) # ensure contiguous
        y_train = torch.from_numpy(np.ascontiguousarray(y_train))
        # Add Gaussian noise to X_train
        noise = torch.randn_like(X_train) * noise_std
        X_train_noisy = X_train + noise

        # Dataset/loader
        train_dataset = TensorDataset(X_train_noisy, y_train)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # -------------------------- Training loop -------------------------
        start_time = time.time()
        all_losses = []
        x_time_steps = x_seconds * 20  # Steps for weighted loss partition

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                h = model.init_hidden(inputs.size(0))
                optimizer.zero_grad()

                outputs, h = model(inputs, h)
                h = [h_i.detach() for h_i in h]

                # Weighted loss for first x_seconds and remaining
                first_x_steps    = targets[:, :x_time_steps]
                remaining_steps  = targets[:, x_time_steps:]

                loss_first_x     = criterion(outputs[:, :x_time_steps], first_x_steps) * w
                loss_remaining   = criterion(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else 0.0

                # Smoothness/continuity loss on output sequence
                continuity_loss  = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)

                loss = loss_first_x + loss_remaining + 0.1 * continuity_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_epoch_loss = total_loss / max(1, len(train_loader))
            all_losses.append(avg_epoch_loss)

            # Step LR scheduler AFTER epoch
            scheduler.step()

            # Save & log every LOG_EVERY / SAVE_EVERY epochs (no prints)
            if (epoch % LOG_EVERY == 0) or (epoch == epochs):
                elapsed = time.time() - start_time
                cur_lr  = get_current_lr(optimizer)

                # Save checkpoint
                ckpt_path = os.path.join(model_folder, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)

                # Log
                with open(model_log_path, 'a') as mlog:
                    mlog.write(f"[{now_str()}] epoch={epoch} | avg_loss={avg_epoch_loss:.6f} "
                               f"| elapsed_s={elapsed:.2f} | lr={cur_lr:.6e} | ckpt='{ckpt_path}'\n")
                    mlog.flush()

        training_time = time.time() - start_time

        # ---------------------------- Testing -----------------------------
        # Load the last checkpoint (final epoch)
        final_ckpt = os.path.join(model_folder, f"epoch_{epochs}.pth")
        model.load_state_dict(torch.load(final_ckpt, map_location=device))
        model.eval()

        prediction_times = []
        absolute_errors  = []
        errors_3s, errors_4s, errors_5s = [], [], []

        steps_3s = 3 * 20
        steps_4s = 4 * 20
        steps_5s = 5 * 20

        total_steps = len(test_data) - sequence_length - output_size - 1
        start_index = sequence_length
        end_index   = start_index + total_steps
        if end_index + output_size > len(test_data):
            end_index   = len(test_data) - output_size
            total_steps = end_index - start_index

        h = model.init_hidden(1)  # batch size 1 for inference

        # Video writer
        fig = plt.figure(figsize=(12, 8))
        writer = animation.FFMpegWriter(fps=VIDEO_FPS)

        video_out_path = os.path.join(PLOT_DIR, f"{model_name}.mp4")
        with torch.no_grad(), writer.saving(fig, video_out_path, dpi=VIDEO_DPI):
            for i in range(start_index, end_index):
                input_tensor = torch.tensor([[[test_data[i]]]], dtype=torch.float32).to(device)

                # Noisy input at test time as in original code
                test_noise = torch.randn_like(input_tensor) * noise_std
                input_tensor_noisy = input_tensor + test_noise

                t0 = time.perf_counter()
                output, h = model(input_tensor_noisy, h)
                h = [h_i.detach() for h_i in h]
                t1 = time.perf_counter()
                prediction_times.append(t1 - t0)

                predicted = output.detach().cpu().numpy().flatten()
                true_future      = test_data[i + 1:i + 1 + output_size] * meters_to_cm
                predicted_future = predicted * meters_to_cm
                abs_error        = np.abs(true_future - predicted_future)

                if i >= start_index + sequence_length:
                    absolute_errors.append(abs_error.mean())
                    errors_3s.append(np.mean(abs_error[:steps_3s]))
                    errors_4s.append(np.mean(abs_error[:steps_4s]))
                    errors_5s.append(np.mean(abs_error[:steps_5s]))

                # Plot
                fig.clear()
                ax1 = fig.add_subplot(2, 1, 1)
                xs = np.arange(i + 1, i + 1 + output_size) / 20.0
                ax1.plot(xs, true_future, 'g--', label='True Future Data (cm)')
                ax1.plot(xs, predicted_future, 'r',   label='Predicted Data (cm)')
                ax1.set_ylim(-30, 30)
                ax1.set_title(f"Time Elapsed: {(i - start_index) / 20:.2f} s")
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Prediction (cm)')
                ax1.legend()

                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(xs, abs_error, 'b', label='Absolute Error (cm)')
                ax2.set_ylim(0, 15)
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Error (cm)')
                ax2.legend()

                writer.grab_frame()

        # -------------------------- Metrics & Logs ------------------------
        avg_prediction_time = float(np.mean(prediction_times)) if prediction_times else 0.0
        avg_error           = float(np.mean(absolute_errors))  if absolute_errors  else 0.0
        avg_error_3s        = float(np.mean(errors_3s))        if errors_3s        else 0.0
        avg_error_4s        = float(np.mean(errors_4s))        if errors_4s        else 0.0
        avg_error_5s        = float(np.mean(errors_5s))        if errors_5s        else 0.0

        # Append to summary file
        summary_file.write(
            f"Model: {model_name}\n"
            f"Average Prediction Time: {avg_prediction_time:.4f} s\n"
            f"Average Absolute Error (Total): {avg_error:.4f} cm\n"
            f"Average Absolute Error (3s): {avg_error_3s:.4f} cm\n"
            f"Average Absolute Error (4s): {avg_error_4s:.4f} cm\n"
            f"Average Absolute Error (5s): {avg_error_5s:.4f} cm\n\n"
            f"------------------------------------------------------------\n"
        )
        summary_file.flush()

        # Also write per-model test summary to model log
        with open(model_log_path, 'a') as mlog:
            mlog.write('-'*80 + '\n')
            mlog.write(f"[{now_str()}] Training finished in {training_time:.2f} s\n")
            mlog.write(f"Final checkpoint: {final_ckpt}\n")
            mlog.write(f"Test video saved: {video_out_path}\n")
            mlog.write(f"Average Prediction Time: {avg_prediction_time:.6f} s\n")
            mlog.write(f"Average Absolute Error (Total): {avg_error:.6f} cm\n")
            mlog.write(f"Average Absolute Error (3s): {avg_error_3s:.6f} cm\n")
            mlog.write(f"Average Absolute Error (4s): {avg_error_4s:.6f} cm\n")
            mlog.write(f"Average Absolute Error (5s): {avg_error_5s:.6f} cm\n")
            mlog.write('-'*80 + '\n\n')
            mlog.flush()

    summary_file.close()

# Entry point
if __name__ == '__main__':
    main()
