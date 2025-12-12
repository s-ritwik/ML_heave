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
- Adds robust error logging and configurable LR schedule (exponential/step/cosine)

NEW:
- CLI resume: pass --resume to auto-resume per config from newest checkpoint in its folder
  (prefers .pt; falls back to .pth). Optionally use --resume-path to specify an exact file.
"""

import os
import re
import sys
import glob
import json
import time
import math
import traceback
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------ CONSTANTS / PATHS --------------------------------

# Root dirs for artifacts
MODEL_ROOT_DIR = 'noisyGRU_models_seq'
PLOT_DIR        = 'noisyGRU_videos'
LOG_DIR         = 'log'
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,        exist_ok=True)
os.makedirs(LOG_DIR,         exist_ok=True)

# Video font fix for headless
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

# Simple stdout/stderr logger
class Logger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.logfile  = open(filename, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

# GPU selection (0 or 1)
GPU_INDEX = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global data/config files
CONFIG_FILE_PATH = 'model_configs_seq.txt'
MODEL_SUMMARY_PATH = 'model_summary.txt'

# Training data locations (kept consistent; original code mixed two dirs)
TRAIN_DIR = 'train_data_normalised'
TRAIN_MOCAP_DIR = 'train_data_normalised_mocap'  # used for test file below

# Testing
TEST_FILE_PATH = os.path.join(TRAIN_MOCAP_DIR, 'D1H3_normalised.csv')

# Noise and scaling constants
NOISE_STD_DEFAULT = 0.05        # Gaussian noise std on inputs
METERS_TO_CM      = 25          # conversion factor as used in the original code
# MCA residual-learning controls (added)
USE_MCA = False                 # if True, train GRU on residuals and add MCA baseline at test
MCA_N = None                    # defaults to sequence_length if None
MCA_M = None                    # defaults to output_size if None
MCA_CENTER = True               # subtract per-time-offset mean on training windows
MCA_ENERGY_CUTOFF = 0.01        # fraction of energy in minor subspace if P not given
MCA_P = None                    # number of minor components to keep; overrides cutoff when set
MCA_RIDGE = 1e-6                # Tikhonov regularization for (B2^T B2 + λI)^{-1}
MCA_CACHE = None

# Saving / logging cadence
SAVE_EVERY = 20                 # epochs
LOG_EVERY  = 20                 # epochs (same as SAVE_EVERY per request)
CONTINUITY_WEIGHT = 10.0         # smoothness weight between adjacent outputs (all horizons)
DERIV_WEIGHT_FIRST = 40.0        # weight for matching first-derivative over first x_seconds
CURV_WEIGHT_FIRST  = 1.0        # weight for second-derivative smoothness over first x_seconds

# Video writer settings
VIDEO_FPS = 20
VIDEO_DPI = 100

# --------------------- RESUME CONTROLS (overridden by CLI) ---------------------
RESUME = False
RESUME_PATH = ""   # e.g. "noisyGRU_models_seq/noisy_D1_GRU_20_8_512_256/epoch_100.pt"

# -------------------------------------------------------------------------
# -------------------------- UTILITY FUNCTIONS ----------------------------
# -------------------------- MCA helper functions (added) --------------------------

def _row_center_windows(windows):
    """Row-center across samples: subtract mean per time offset (axis=0)."""
    mu = windows.mean(axis=0, keepdims=True)
    return windows - mu, mu.squeeze(0)

def _select_minor_components(eigvals, U, energy_cutoff=0.01, P=None):
    """
    Match mca_brute: pick smallest eigenvalues first, either fixed P or until cumulative
    fraction exceeds energy_cutoff. Always keep at least 1 and at most N-1.
    """
    N = eigvals.shape[0]
    order = np.argsort(eigvals)  # ascending (minor first)
    if P is not None:
        P = int(P)
        if P < 1:
            P = 1
        if P >= N:
            P = N - 1
        idx_minor = order[:P]
        return idx_minor, U[:, idx_minor]
    total = eigvals.sum()
    if total <= 0:
        idx_minor = order[:1]
        return idx_minor, U[:, idx_minor]
    csum = 0.0
    chosen = []
    for j in order:
        csum += eigvals[j]
        chosen.append(j)
        if (csum / total) >= energy_cutoff:
            break
    if len(chosen) == 0:
        chosen = [order[0]]
    idx_minor = np.array(chosen, dtype=int)
    return idx_minor, U[:, idx_minor]

def _compute_W_from_B(B, n, m, ridge):
    """
    Match mca_brute: B shape (P, n+m); W = -(B2^T B2 + λI)^{-1} B2^T B1  (m x n).
    """
    P, N = B.shape
    assert N == n + m, "B must have shape (P, n+m)"
    B1 = B[:, :n]
    B2 = B[:, n:]
    BtB = B2.T @ B2               # (m x m)
    BtB_reg = BtB + ridge * np.eye(BtB.shape[0])
    BtB_B1 = B2.T @ B1            # (m x n)
    W = -np.linalg.solve(BtB_reg, BtB_B1)
    return W

def _mca_predict_block(W, X1, mu_full, n, m, centered):
    """
    Vectorized predict: X1 shape (K, n) -> returns (K, m).
    """
    if centered:
        mu_full = np.asarray(mu_full)
        mu1 = mu_full[:n]
        mu2 = mu_full[n:]
        X1c = X1 - mu1
        return (W @ X1c.T).T + mu2
    return (W @ X1.T).T

def _enforce_first_point(outputs, last_inputs):
    """
    Hard constraint: set first predicted point to the last input reading.
    outputs: tensor [B, m]
    last_inputs: tensor [B] (last value of input sequence)
    Returns a new tensor with outputs[:,0] overwritten.
    """
    out = outputs.clone()
    out[:, 0] = last_inputs
    return out

def compute_mca_predictor(windows, n, m, center=True, energy_cutoff=0.01, P=None, ridge=1e-6):
    """
    Compute MCA linear predictor with the same mean handling/orientation as mca_brute.
    windows: [Nsamples, n+m] clean training windows (no noise).
    """
    X = windows.astype(np.float64)
    if center:
        Xc, mu = _row_center_windows(X)
    else:
        Xc, mu = X, np.zeros(X.shape[1], dtype=np.float64)

    # Autocorrelation eigendecomposition (same as SVD-based in mca_brute)
    R = (Xc.T @ Xc) / max(1, Xc.shape[0])
    eigvals, evecs = np.linalg.eigh(R)
    idx_minor, U_minor = _select_minor_components(eigvals, evecs, energy_cutoff=energy_cutoff, P=P)
    B = U_minor.T  # (P x (n+m))
    W_mn = _compute_W_from_B(B, n=n, m=m, ridge=ridge)

    B1 = B[:, :n]
    B2 = B[:, n:]

    return {
        "W": W_mn.astype(np.float64),
        "mu": mu.astype(np.float64),
        "n": int(n),
        "m": int(m),
        "center": bool(center),
        "P_used": int(idx_minor.shape[0]),
        "idx_minor": idx_minor.astype(int),
        "B1": B1.astype(np.float64),
        "B2": B2.astype(np.float64),
        "evals": eigvals.astype(np.float64)
    }

def mca_predict_from_X1(x1_vec, mca):
    """x1_vec shape (n,), returns x2_hat shape (m,) using stored W and mu."""
    n, m = mca["n"], mca["m"]
    assert x1_vec.shape[0] == n, f"expected n={n}, got {x1_vec.shape[0]}"
    mu_full = np.asarray(mca["mu"])
    if mca.get("center", True):
        mu1 = mu_full[:n]
        mu2 = mu_full[n:]
        x1c = x1_vec - mu1
        x2_hat = mca["W"] @ x1c
        return x2_hat + mu2
    return mca["W"] @ x1_vec

def build_train_windows_1d(series, n, m):
    total = n + m
    if len(series) < total:
        raise ValueError(f"Series too short ({len(series)}) for window {total}.")
    win = np.lib.stride_tricks.sliding_window_view(series, total)  # [N, total]
    X1 = win[:, :n].copy()
    X2 = win[:, n:].copy()
    return X1, X2

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
    """Parse a single model config line. Supports extras for LR scheduling.
       Example:
       'sequence_length:400; output_size:160; hidden_sizes:[512,256]; x_seconds:3; w:2; batch_size:64; epochs:120; learning_rate:0.001; lr_scheduler:step; step_size:40; gamma:0.5'
    """
    config = {}
    params = [p for p in config_line.split(';') if p.strip()]
    for param in params:
        key, value = param.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == 'hidden_sizes':
            config[key] = list(map(int, re.findall(r'\d+', value)))
        elif key in ['sequence_length', 'output_size', 'x_seconds', 'w', 'batch_size', 'epochs', 'step_size', 'T_max']:
            config[key] = int(value)
        elif key in ['learning_rate', 'decay', 'gamma']:
            config[key] = float(value)
        elif key in ['lr_scheduler']:
            config[key] = value.lower()
        else:
            # keep any unknown key as raw string (optional)
            config[key] = value
    return config

def load_data(file_list, data_dir=TRAIN_MOCAP_DIR):
    """Load 1D sequences from CSV files (first column) and return a single concatenated numpy array."""
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data, dtype=np.float32)

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def get_scheduler(optimizer, config):
    """Return a configured LR scheduler. Defaults to exponential decay."""
    sched_type = config.get("lr_scheduler", "exponential")
    if sched_type == "exponential":
        decay = config.get("decay", 0.99)
        return LambdaLR(optimizer, lr_lambda=lambda e: decay ** e)
    elif sched_type == "step":
        step_size = config.get("step_size", 50)
        gamma = config.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == "cosine":
        T_max = config.get("T_max", config.get("epochs", 100))
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type}")

def latest_checkpoint_pt(model_folder):
    """Return the path to the latest epoch_*.pt checkpoint if present; else '' (legacy helper)."""
    pts = glob.glob(os.path.join(model_folder, "epoch_*.pt"))
    if not pts:
        return ""
    def _epoch_num(p):
        m = re.search(r'epoch_(\d+)\.pt$', os.path.basename(p))
        return int(m.group(1)) if m else -1
    pts = sorted(pts, key=_epoch_num)
    return pts[-1]

# --------- resume helpers (supports .pt and .pth; prefers higher epoch, then .pt) ---------

def _extract_epoch(path):
    m = re.search(r'epoch_(\d+)\.(pt|pth)$', os.path.basename(path))
    return int(m.group(1)) if m else None

def find_latest_checkpoint_any(model_folder):
    """
    Find newest checkpoint among *.pt and *.pth.
    Returns (path, kind, epoch) where kind in {'pt','pth'}; or (None, None, None).
    Preference: higher epoch; if tie, prefer .pt over .pth.
    """
    pts  = glob.glob(os.path.join(model_folder, "epoch_*.pt"))
    pths = glob.glob(os.path.join(model_folder, "epoch_*.pth"))
    candidates = []
    for p in pts:
        ep = _extract_epoch(p)
        if ep is not None:
            candidates.append((ep, 'pt', p))
    for p in pths:
        ep = _extract_epoch(p)
        if ep is not None:
            candidates.append((ep, 'pth', p))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda t: (t[0], 0 if t[1] == 'pt' else 1))  # epoch asc, pt before pth
    ep, kind, path = candidates[-1]
    return path, kind, ep

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
        h = [torch.zeros(1, batch_size, hidden_size, device=device) for hidden_size in self.hidden_sizes]
        return h

# ----------------- ERROR LOGGING -------------------------------------------

ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")

# Redirect stdout and stderr to a timestamped run log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_log_file = os.path.join(LOG_DIR, f"run_{timestamp}.log")
sys.stdout = Logger(run_log_file)
sys.stderr = Logger(run_log_file)

# Catch uncaught exceptions and log them
def log_exception(exc_type, exc_value, exc_traceback):
    with open(ERROR_LOG_FILE, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        f.write("="*80 + "\n")
    # Also write to the run log
    with open(run_log_file, "a") as rlog:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=rlog)

sys.excepthook = log_exception

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
    test_data = pd.read_csv(TEST_FILE_PATH, header=None).iloc[:, 0].values.astype(np.float32)
    meters_to_cm = METERS_TO_CM

    # Prepare model summary file (append mode)
    summary_file = open(MODEL_SUMMARY_PATH, 'a')

    # Iterate through model configurations
    for config_line in model_configs:
        try:
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
            noise_std       = NOISE_STD_DEFAULT  # single global noise hyperparam for all models

            # Build model name and per-model paths
            model_name   = f"noisy_D1_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}"
            print("Training for:", model_name, file=sys.__stdout__)
            # print("Training for:", model_name)

            model_folder = os.path.join(MODEL_ROOT_DIR, model_name)
            os.makedirs(model_folder, exist_ok=True)

            # Initialize model/optimizer/etc.
            model = GRUModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = get_scheduler(optimizer, config)

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_log_path = os.path.join(LOG_DIR, f"{model_name}.log")
            with open(model_log_path, 'a') as mlog:
                # Header for this model
                mlog.write('='*80 + '\n')
                mlog.write(f"[{now_str()}] Starting model: {model_name}\n")
                mlog.write(f"Config: {json.dumps(config)}\n")
                mlog.write(f"Device: {device} | GPU_INDEX={GPU_INDEX}\n")
                mlog.write(f"Trainable parameters: {n_params:,}\n")
                mlog.write(f"noise_std={noise_std}, meters_to_cm={meters_to_cm}\n")
                mlog.write(f"USE_MCA={USE_MCA}, MCA_P={MCA_P}, MCA_ENERGY_CUTOFF={MCA_ENERGY_CUTOFF}, MCA_CENTER={MCA_CENTER}, MCA_RIDGE={MCA_RIDGE}\n")
                mlog.write(f"Checkpoints every {SAVE_EVERY} epochs | Logs every {LOG_EVERY} epochs\n")
                mlog.write('='*80 + '\n')
                mlog.flush()

            # --------------------- Prepare training data ----------------------
            csv_files   = os.listdir(TRAIN_DIR)
            train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
            train_data  = load_data(train_files, data_dir=TRAIN_DIR)

            # Build sequences (fast, without Python loop)
            total = sequence_length + output_size
            if len(train_data) < total:
                raise ValueError(f"Training data too short ({len(train_data)}) for total window size {total}.")
            windows = np.lib.stride_tricks.sliding_window_view(train_data, total)  # [N, total]
            X_np = windows[:, :sequence_length][..., None].copy()  # [N, seq_len, 1]
            y_np = windows[:, sequence_length:].copy()              # [N, out_size]

            X_train = torch.from_numpy(X_np)
            y_train = torch.from_numpy(y_np)

            # ---------------- MCA residual target construction (added) ----------------
            if USE_MCA:
                n_mca = MCA_N or sequence_length
                m_mca = MCA_M or output_size
                # Build clean training windows for MCA
                X1_clean = X_np[:, :n_mca] if X_np.shape[1] >= n_mca else np.ascontiguousarray(X_np.squeeze(-1))[:, :n_mca]
                X1_clean = X1_clean.reshape(X1_clean.shape[0], n_mca)  # [N, n]
                X2_clean = y_np[:, :m_mca]  # [N, m]
                train_windows_clean = np.concatenate([X1_clean, X2_clean], axis=1)  # [N, n+m]
                mca_model = compute_mca_predictor(train_windows_clean, n=n_mca, m=m_mca, center=MCA_CENTER, energy_cutoff=MCA_ENERGY_CUTOFF, P=MCA_P, ridge=MCA_RIDGE)
                # Save MCA model
                with open(os.path.join(model_folder, 'mca_model.json'), 'w') as fjson:
                    fjson.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in mca_model.items()}))
                # Compute baseline predictions for training windows and residual targets (match mca_brute mean handling)
                X2_hat = _mca_predict_block(
                    mca_model['W'],
                    train_windows_clean[:, :n_mca],
                    mca_model['mu'],
                    n=n_mca,
                    m=m_mca,
                    centered=MCA_CENTER
                )  # [N, m]
                y_train_residual_np = y_np[:, :m_mca] - X2_hat
                # Pad residual targets to match output_size if m_mca < output_size
                if m_mca != output_size:
                    pad = output_size - m_mca
                    if pad > 0:
                        y_train_residual_np = np.hstack([y_train_residual_np, np.zeros((y_train_residual_np.shape[0], pad), dtype=y_train_residual_np.dtype)])
                    else:
                        y_train_residual_np = y_train_residual_np[:, :output_size]
                y_train_residual = torch.from_numpy(y_train_residual_np.astype(np.float32))

            # Add Gaussian noise to X_train
            noise = torch.randn_like(X_train) * noise_std
            X_train_noisy = X_train + noise

            # Dataset/loader
            train_dataset = TensorDataset(X_train_noisy, y_train)
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            if USE_MCA:
                train_dataset = TensorDataset(X_train_noisy, y_train_residual)
                train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            # ------------------- RESUME (per-config) -------------------
            start_epoch = 1
            final_resume_path = None

            if RESUME:
                if RESUME_PATH and os.path.isfile(RESUME_PATH):
                    final_resume_path = RESUME_PATH
                    final_resume_kind = 'pt' if RESUME_PATH.endswith('.pt') else 'pth' if RESUME_PATH.endswith('.pth') else None
                    final_resume_epoch = _extract_epoch(RESUME_PATH)
                else:
                    rp, rk, re = find_latest_checkpoint_any(model_folder)
                    final_resume_path, final_resume_kind, final_resume_epoch = rp, rk, re

                if final_resume_path and final_resume_kind:
                    if final_resume_kind == 'pt':
                        ckpt = torch.load(final_resume_path, map_location=device)
                        model.load_state_dict(ckpt["model_state_dict"])
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        if scheduler and ckpt.get("scheduler_state_dict"):
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        stored_epoch = int(ckpt.get("epoch", final_resume_epoch or 0))
                        start_epoch = stored_epoch + 1
                    elif final_resume_kind == 'pth':
                        state_dict = torch.load(final_resume_path, map_location=device)
                        model.load_state_dict(state_dict)
                        start_epoch = (final_resume_epoch + 1) if final_resume_epoch is not None else 1

                    with open(model_log_path, 'a') as mlog:
                        mlog.write(f"[{now_str()}] RESUME=True | Resuming from '{final_resume_path}' "
                                   f"(kind={final_resume_kind}, parsed_epoch={final_resume_epoch}) "
                                   f"→ start_epoch={start_epoch}\n")
                        mlog.flush()
                else:
                    with open(model_log_path, 'a') as mlog:
                        mlog.write(f"[{now_str()}] RESUME=True | No checkpoint found for '{model_name}'. Starting fresh.\n")
                        mlog.flush()
            else:
                with open(model_log_path, 'a') as mlog:
                    mlog.write(f"[{now_str()}] RESUME=False | Starting fresh from epoch 1.\n")
                    mlog.flush()

            # -------------------------- Training loop -------------------------
            start_time = time.time()
            x_time_steps = x_seconds * 20  # Steps for weighted loss partition

            for epoch in range(start_epoch, epochs + 1):
                model.train()
                total_loss = 0.0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    h = model.init_hidden(inputs.size(0))
                    optimizer.zero_grad()

                    outputs, h = model(inputs, h)
                    h = [h_i.detach() for h_i in h]

                    # Enforce hard continuity on first point
                    last_input_point = inputs[:, -1, 0]
                    outputs = _enforce_first_point(outputs, last_input_point)

                    # Weighted loss for first x_seconds and remaining
                    first_x_steps   = targets[:, :x_time_steps]
                    remaining_steps = targets[:, x_time_steps:]

                    loss_first_x    = criterion(outputs[:, :x_time_steps], first_x_steps) * w
                    loss_remaining  = criterion(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else 0.0

                    # Smoothness/continuity loss on output sequence
                    continuity_loss = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)

                    # Strong smoothness/derivative matching on first x_seconds
                    first_seg = min(x_time_steps, output_size)
                    if first_seg >= 2:
                        pred_diff  = outputs[:, 1:first_seg] - outputs[:, :first_seg-1]
                        true_diff  = targets[:, 1:first_seg] - targets[:, :first_seg-1]
                        deriv_loss_first = criterion(pred_diff, true_diff)
                    else:
                        deriv_loss_first = 0.0
                    if first_seg >= 3:
                        curvature = outputs[:, 2:first_seg] - 2*outputs[:, 1:first_seg-1] + outputs[:, :first_seg-2]
                        curvature_loss_first = torch.mean(curvature ** 2)
                    else:
                        curvature_loss_first = 0.0

                    loss = (
                        loss_first_x
                        + loss_remaining
                        + CONTINUITY_WEIGHT * continuity_loss
                        + DERIV_WEIGHT_FIRST * deriv_loss_first
                        + CURV_WEIGHT_FIRST  * curvature_loss_first
                    )
                    if not torch.isfinite(loss):
                        with open(model_log_path, 'a') as mlog:
                            mlog.write(f"[{now_str()}] Non-finite loss detected; skipping batch. "
                                       f"loss_first={float(loss_first_x)} "
                                       f"loss_rem={float(loss_remaining) if isinstance(loss_remaining, torch.Tensor) else loss_remaining} "
                                       f"cont={float(continuity_loss)} "
                                       f"deriv_first={float(deriv_loss_first) if isinstance(deriv_loss_first, torch.Tensor) else deriv_loss_first} "
                                       f"curv_first={float(curvature_loss_first) if isinstance(curvature_loss_first, torch.Tensor) else curvature_loss_first}\n")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += float(loss.item())

                avg_epoch_loss = total_loss / max(1, len(train_loader))

                # Step LR scheduler AFTER epoch
                scheduler.step()

                # Save & log every LOG_EVERY / SAVE_EVERY epochs (no prints)
                if (epoch % LOG_EVERY == 0) or (epoch == epochs):
                    elapsed = time.time() - start_time
                    cur_lr  = get_current_lr(optimizer)

                    # Save checkpoint (.pt = full checkpoint; .pth = weights-only)
                    base = f"epoch_{epoch}"
                    ckpt_pt  = os.path.join(model_folder, base + ".pt")
                    ckpt_pth = os.path.join(model_folder, base + ".pth")

                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "loss": avg_epoch_loss,
                    }
                    torch.save(checkpoint, ckpt_pt)
                    torch.save(model.state_dict(), ckpt_pth)

                    with open(model_log_path, 'a') as mlog:
                        mlog.write(f"[{now_str()}] epoch={epoch} | avg_loss={avg_epoch_loss:.6f} "
                                   f"| elapsed_s={elapsed:.2f} | lr={cur_lr:.6e} "
                                   f"| ckpt_full='{ckpt_pt}' | ckpt_weights='{ckpt_pth}'\n")
                        mlog.flush()

            training_time = time.time() - start_time
            print(f"Training completed for {model_name} in {training_time/60:.2f} minutes.", file=sys.__stdout__)
            # ---------------------------- Testing -----------------------------
            # Prefer the newest .pt/.pth in this model folder
            final_ckpt_path, final_kind, _ = find_latest_checkpoint_any(model_folder)
            if final_ckpt_path and final_kind == 'pt':
                ckpt = torch.load(final_ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                final_ckpt_display = final_ckpt_path
            elif final_ckpt_path and final_kind == 'pth':
                model.load_state_dict(torch.load(final_ckpt_path, map_location=device))
                final_ckpt_display = final_ckpt_path
            else:
                # fall back to weights of final epoch (expected to exist)
                final_ckpt_pth = os.path.join(model_folder, f"epoch_{epochs}.pth")
                model.load_state_dict(torch.load(final_ckpt_pth, map_location=device))
                final_ckpt_display = final_ckpt_pth

            model.eval()

            prediction_times = []
            absolute_errors  = []
            errors_3s, errors_4s, errors_5s = [], [], []

            steps_3s = 3 * 20
            steps_4s = 4 * 20
            steps_5s = 5 * 20

            # <<< NEW: robust length handling for 1D or 2D test arrays
            T_test = test_data.shape[0] if isinstance(test_data, np.ndarray) else len(test_data)
            total_steps = T_test - sequence_length - output_size - 1
            start_index = sequence_length
            end_index   = start_index + total_steps
            if end_index + output_size > T_test:
                end_index   = T_test - output_size

            h = model.init_hidden(1)  # batch size 1 for inference

            # Video writer
            fig = plt.figure(figsize=(12, 8))
            writer = animation.FFMpegWriter(fps=VIDEO_FPS)

            video_out_path = os.path.join(PLOT_DIR, f"{model_name}.mp4")
            with torch.no_grad(), writer.saving(fig, video_out_path, dpi=VIDEO_DPI):
                for i in range(start_index, end_index):
                    noisy_series = np.full(len(test_data), np.nan, dtype=np.float32)

                    # ---------- inside the testing loop, REPLACE your per-frame block with this ----------
                    # one point per step (streaming) + test-time noise
                    val = torch.tensor([[[test_data[i]]]], dtype=torch.float32, device=device)  # [1,1,1]
                    input_tensor_noisy = val + torch.randn_like(val) * noise_std

                    # record the actual noisy value we fed (convert to cm for plotting)
                    noisy_val_cm = float(input_tensor_noisy.squeeze().detach().cpu().numpy()) * meters_to_cm
                    noisy_series[i] = noisy_val_cm

                    t0 = time.perf_counter()
                    output, h = model(input_tensor_noisy, h)  # stateful inference: one datapoint per tick
                    # detach hidden state (LSTM has detach_state; GRU uses list detach)
                    h = model.detach_state(h) if hasattr(model, "detach_state") else [hh.detach() for hh in h]
                    t1 = time.perf_counter()
                    prediction_times.append(t1 - t0)

                    predicted = output.detach().cpu().numpy().flatten()
                    # Enforce hard continuity at inference: first prediction equals last clean reading
                    predicted[0] = float(test_data[i])
                    true_future      = test_data[i + 1:i + 1 + output_size] * meters_to_cm
                    # ---------------- MCA baseline at inference (added) ----------------
                    if USE_MCA:
                        try:
                            global MCA_CACHE
                            if MCA_CACHE is None:
                                with open(os.path.join(model_folder, 'mca_model.json'), 'r') as fjson:
                                    _m = json.load(fjson)
                                MCA_CACHE = {k: (np.array(v) if isinstance(v, list) else v) for k, v in _m.items()}
                            mca_model = MCA_CACHE
                            n_mca = int(mca_model['n']); m_mca = int(mca_model['m'])
                            # Build last n_mca past window from CLEAN test_data
                            if i - n_mca + 1 >= 0:
                                x1 = test_data[i - n_mca + 1: i + 1].astype(np.float64).copy()
                            else:
                                # pad with earliest value if not enough past
                                pad = n_mca - (i + 1)
                                x1 = np.concatenate([np.full(pad, test_data[0], dtype=np.float64), test_data[:i+1].astype(np.float64)])
                            # Predict baseline with MCA
                            x2_hat = mca_predict_from_X1(x1, mca_model)  # shape (m,)
                            # Ensure length equals output_size
                            if m_mca != output_size:
                                if m_mca > output_size: x2_hat = x2_hat[:output_size]
                                else: x2_hat = np.hstack([x2_hat, np.zeros(output_size - m_mca, dtype=x2_hat.dtype)])
                            predicted_future = (predicted + x2_hat) * meters_to_cm
                        except Exception as _e:
                            # fall back to GRU-only if MCA state missing
                            predicted_future = predicted * meters_to_cm
                    else:
                        predicted_future = predicted * meters_to_cm
                    # Re-enforce first-point continuity after any MCA blending
                    predicted_future[0] = float(test_data[i] * meters_to_cm)
                    abs_error        = np.abs(true_future - predicted_future)

                    absolute_errors.append(abs_error.mean())
                    if steps_3s > 0: errors_3s.append(float(np.mean(abs_error[:steps_3s])))
                    if steps_4s > 0: errors_4s.append(float(np.mean(abs_error[:steps_4s])))
                    if steps_5s > 0: errors_5s.append(float(np.mean(abs_error[:steps_5s])))

                    # ---- Build history window (exactly `sequence_length` points if available) ----
                    hist_start = max(0, i - sequence_length + 1)
                    hist_x = np.arange(hist_start, i + 1) / 20.0
                    hist_clean_cm = test_data[hist_start : i + 1] * meters_to_cm
                    hist_noisy_cm = noisy_series[hist_start : i + 1]  # contains NaNs for early steps

                    # ---- Future time axis ----
                    fut_x = np.arange(i + 1, i + 1 + output_size) / 20.0

                    # ---- Draw frame: show history (clean + noisy) and future (true + predicted) ----
                    fig.clear()
                    ax1 = fig.add_subplot(2, 1, 1)

                    # history (clean) — what the underlying signal actually was
                    ax1.plot(hist_x, hist_clean_cm, 'k', linewidth=1.2, label=f'History clean ({len(hist_clean_cm)} steps)')

                    # history (noisy) — EXACT values fed to the model at each tick
                    # we mask NaNs so the line starts when data is first available
                    mask = ~np.isnan(hist_noisy_cm)
                    if np.any(mask):
                        ax1.plot(hist_x[mask], hist_noisy_cm[mask], '--', linewidth=1.0, label='History noisy (fed)', alpha=0.9)

                    # future (true vs predicted)
                    ax1.plot(fut_x, true_future, 'g--', linewidth=1.2, label=f'True (+{output_size/20:.1f}s)')
                    ax1.plot(fut_x, predicted_future, 'r', linewidth=1.2, label='Predicted')

                    ax1.set_title(f"{model_name} | i={i} | ckpt={os.path.basename(final_ckpt_display)}")
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Value (cm)')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='upper right', fontsize=8)

                    # error panel (current horizon only)
                    ax2 = fig.add_subplot(2, 1, 2)
                    horizon_axis = np.arange(1, output_size + 1) / 20.0
                    ax2.plot(horizon_axis, abs_error, linewidth=1.0, label='|Pred-True| (cm)')
                    ax2.set_xlabel('Prediction horizon (s)')
                    ax2.set_ylabel('Error (cm)')
                    ax2.set_xlim(0, output_size/20.0)
                    ax2.set_ylim(0, max(1.0, float(np.nanmax(abs_error)) * 1.1))
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper right', fontsize=8)

                    writer.grab_frame()

            # ----------------- Summaries -----------------
            avg_pred_time_ms = np.mean(prediction_times) * 1000.0 if prediction_times else float('nan')
            mean_abs_err_cm  = float(np.mean(absolute_errors)) if absolute_errors else float('nan')
            mae_3s = float(np.mean(errors_3s)) if errors_3s else float('nan')
            mae_4s = float(np.mean(errors_4s)) if errors_4s else float('nan')
            mae_5s = float(np.mean(errors_5s)) if errors_5s else float('nan')

            # with open(model_log_path, 'a') as mlog:
            #     mlog.write(f"[{now_str()}] DONE training time={training_time:.2f}s, "
            #                f"avg_pred_time_ms={avg_pred_time_ms:.3f}, "
            #                f"MAE_cm={mean_abs_err_cm:.3f}, MAE_3s={mae_3s:.3f}, MAE_4s={mae_4s:.3f}, MAE_5s={mae_5s:.3f}\n")
            #     mlog.flush()
            with open(model_log_path, 'a') as mlog:
                mlog.write('-'*80 + '\n')
                mlog.write(f"[{now_str()}] Training finished in {training_time:.2f} s\n")
                mlog.write(f"Final checkpoint used: {final_ckpt_display}\n")
                mlog.write(f"Test video saved: {video_out_path}\n")
                mlog.write(f"Average Prediction Time: {avg_pred_time_ms:.6f} ms\n")
                mlog.write(f"Average Absolute Error (Total): {mean_abs_err_cm:.6f} cm\n")
                mlog.write(f"Average Absolute Error (3s): {mae_3s:.6f} cm\n")
                mlog.write(f"Average Absolute Error (4s): {mae_4s:.6f} cm\n")
                mlog.write(f"Average Absolute Error (5s): {mae_5s:.6f} cm\n")
                mlog.write('-'*80 + '\n\n')
                mlog.flush()
            # Write to model summary file
            summary_file.write(
                f"{model_name}, params={n_params}, train_time_s={training_time:.2f}, "
                f"pred_ms={avg_pred_time_ms:.3f}, mae_cm={mean_abs_err_cm:.3f}, "
                f"mae_3s={mae_3s:.3f}, mae_4s={mae_4s:.3f}, mae_5s={mae_5s:.3f}\n"
            )
            summary_file.flush()

        except Exception as e:
            # goes to run log file (stdout is redirected) and errors.log via excepthook
            print(f"Error processing config '{config_line}': {e}")

    summary_file.close()

# ----------------------------- CLI PARSER ---------------------------------

def _build_argparser():
    parser = argparse.ArgumentParser(description="Noisy GRU training & evaluation")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume per-config from the latest checkpoint in its model folder."
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Explicit checkpoint file (.pt or .pth) to resume from; overrides auto-detection."
    )
    parser.add_argument('--use-mca', action='store_true', help='Enable MCA residual learning and inference.')
    parser.add_argument('--mca-n', type=int, default=None, help='MCA n past samples (default: sequence_length).')
    parser.add_argument('--mca-m', type=int, default=None, help='MCA m future samples (default: output_size).')
    parser.add_argument('--mca-center', action='store_true', default=True, help='Row-center windows for MCA.')
    parser.add_argument('--mca-energy-cutoff', type=float, default=0.01, help='Minor-subspace energy fraction.')
    parser.add_argument('--mca-P', type=int, default=None, help='Use exactly P minor components.')
    parser.add_argument('--mca-ridge', type=float, default=1e-6, help='Ridge for (B2^T B2 + λI).')
    return parser

# Entry point
if __name__ == '__main__':
    args = _build_argparser().parse_args()
    # override globals before main() uses them
    RESUME = bool(args.resume)
    RESUME_PATH = args.resume_path or RESUME_PATH
    USE_MCA = bool(args.use_mca)
    MCA_N = args.mca_n or MCA_N
    MCA_M = args.mca_m or MCA_M
    MCA_CENTER = bool(args.mca_center)
    MCA_ENERGY_CUTOFF = float(args.mca_energy_cutoff)
    MCA_P = args.mca_P if args.mca_P is not None else MCA_P
    MCA_RIDGE = float(args.mca_ridge)
    main()
