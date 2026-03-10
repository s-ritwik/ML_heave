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
import json
import time
import math
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import traceback
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR

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
#CONFIG_FILE_PATH = 'model_configs_seq2.txt'
CONFIG_FILE_PATH = 'model_configs_seq.txt'
MODEL_SUMMARY_PATH = 'model_summary.txt'

# Training data locations (kept consistent; original code mixed two dirs)
TRAIN_DIR = 'train_data_normalised'
TRAIN_MOCAP_DIR = 'train_data_normalised_mocap'  # used for test file below

# <<< NEW: velocity-augmented dataset directory
TRAIN_DIR_WITH_VEL = 'train_data_mekf_vel'  # z + velocity files live here

# Testing
TEST_FILE_PATH = os.path.join(TRAIN_MOCAP_DIR, 'D1H3_normalised.csv')

# Noise and scaling constants
NOISE_STD_DEFAULT = 0.09        # Gaussian noise std on inputs
METERS_TO_CM      = 25          # conversion factor as used in the original code

# Saving / logging cadence
SAVE_EVERY = 20                 # epochs
LOG_EVERY  = 20                 # epochs (same as SAVE_EVERY per request)

# Video writer settings
VIDEO_FPS = 20
VIDEO_DPI = 100

# --------------------- RESUME CONTROLS (overridden by CLI) ---------------------
RESUME = False
RESUME_PATH = ""   # e.g. "noisyGRU_models_seq/noisy_D1_GRU_20_8_512_256/epoch_100.pt"

# <<< NEW (set by argparse at the bottom)
USE_VEL = False

# # ======== DEBUG HARNESS (paste after imports) ========
# import os, sys, logging, signal, faulthandler, atexit, traceback, time
# LOGFILE = os.path.expanduser("~/gru_debug.log")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # catch async CUDA errors
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout),
#               logging.FileHandler(LOGFILE, mode="a")]
# )
# log = logging.getLogger("GRU_BRUTE")
# log.info("PID=%s starting; log=%s", os.getpid(), LOGFILE)
# faulthandler.enable(open(LOGFILE, "a"))  # also mirrors to stderr
# try:
#     faulthandler.register(signal.SIGUSR1, file=open(LOGFILE, "a"), all_threads=True)
# except Exception:
#     pass
# def _excepthook(exc_type, exc, tb):
#     log.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
#     traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
#     os._exit(1)
# sys.excepthook = _excepthook
# def _sig_handler(signum, frame):
#     log.error("Received signal %s; dumping stack...", signum)
#     traceback.print_stack(frame, file=sys.stderr)
#     faulthandler.dump_traceback(file=open(LOGFILE, "a"), all_threads=True)
#     os._exit(128 + signum)
# for s in (signal.SIGTERM, signal.SIGINT):
#     try: signal.signal(s, _sig_handler)
#     except Exception: pass
# def _heartbeat():
#     log.debug("heartbeat: alive at %s", time.strftime("%H:%M:%S"))
#     sys.stdout.flush()
#     sys.stderr.flush()
#     import threading; threading.Timer(60.0, _heartbeat).start()
# _heartbeat()
# @atexit.register
# def _on_exit():
#     log.info("Process exiting (atexit). If unexpected, scroll above for cause.")
# # ======== END DEBUG HARNESS ========


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

# <<< NEW: helper to detect header (best-effort)
def has_header(csv_path):
    try:
        first = pd.read_csv(csv_path, nrows=1, header=None)
        return not np.all(np.isfinite(pd.to_numeric(first.iloc[0, :], errors='coerce')))
    except Exception:
        return False

# <<< NEW: 1-col and 2-col loaders for the velocity dataset
def load_data_1col(file_list, data_dir):
    data = []
    for file_name in file_list:
        df = pd.read_csv(os.path.join(data_dir, file_name), header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data, dtype=np.float32)

def load_data_2col(file_list, data_dir):
    chunks = []
    for file_name in file_list:
        path = os.path.join(data_dir, file_name)
        df = pd.read_csv(path, header=0 if has_header(path) else None)
        arr = df.iloc[:, :2].astype(np.float32).values  # take first two columns: [z, vz]
        chunks.append(arr)
    return np.vstack(chunks)

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def get_scheduler(optimizer, config):
    """Return a configured LR scheduler. Defaults to exponential decay."""
    sched_type = config.get("lr_scheduler", "exponential")
    if sched_type == "exponential":
        decay = config.get("decay", 0.998)
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

class Logger(object):
    """File-only logger: captures stdout/stderr to a file (no terminal echo)."""
    def __init__(self, logfile):
        self.log = open(logfile, "a", buffering=1)  # line-buffered
    def write(self, message):
        self.log.write(message)
    def flush(self):
        self.log.flush()

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

    # <<< NEW: pick sources based on --vel
    if USE_VEL:
        train_dir = TRAIN_DIR_WITH_VEL
        test_candidates = sorted(glob.glob(os.path.join(TRAIN_DIR_WITH_VEL, "D1H3*.csv")))
        if not test_candidates:
            raise FileNotFoundError("No D1H3*.csv found in TRAIN_DIR_WITH_VEL.")
        test_file = test_candidates[0]
        input_size = 2
        in_tag = "in2_vel"
        # load test data as 2-col (z+vz)
        df_test = pd.read_csv(test_file, header=0 if has_header(test_file) else None)
        test_data = df_test.iloc[:, :2].astype(np.float32).values  # [T,2]
    else:
        train_dir = TRAIN_DIR
        test_file = TEST_FILE_PATH
        input_size = 1
        in_tag = "in1"
        # Load test data (kept as original)
        test_data = pd.read_csv(test_file, header=None).iloc[:, 0].values.astype(np.float32)  # [T]
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
            model_name   = f"noisy_D1_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}_{in_tag}"  # <<< NEW (suffix)
            print("Training for:", model_name, file=sys.__stdout__)
            print("Training for:", model_name)

            model_folder = os.path.join(MODEL_ROOT_DIR, model_name)
            os.makedirs(model_folder, exist_ok=True)

            # Initialize model/optimizer/etc.
            model = GRUModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(device)  # <<< NEW (input_size)
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
                mlog.write(f"Checkpoints every {SAVE_EVERY} epochs | Logs every {LOG_EVERY} epochs\n")
                mlog.write(f"Input size: {input_size} ({'z' if input_size==1 else 'z+vz'})\n")  # <<< NEW
                mlog.write('='*80 + '\n')
                mlog.flush()

            # --------------------- Prepare training data ----------------------
            csv_files   = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
            train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]

            total = sequence_length + output_size

            if input_size == 1:
                series = load_data_1col(train_files, data_dir=train_dir)  # [T]
                if len(series) < total:
                    raise ValueError(f"Training data too short ({len(series)}) for total window size {total}.")
                windows = np.lib.stride_tricks.sliding_window_view(series, total)  # [N, total]
                X_np = windows[:, :sequence_length][..., None].copy()              # [N, seq_len, 1]
                y_np = windows[:, sequence_length:].copy()                         # [N, out_size]
            else:
                series = load_data_2col(train_files, data_dir=train_dir)  # [T, 2]
                if series.shape[0] < total:
                    raise ValueError(f"Training data too short ({series.shape[0]}) for total window size {total}.")
                win = np.lib.stride_tricks.sliding_window_view(series, (total, series.shape[1]))  # [N,1,total,2]
                win = win.reshape(-1, total, series.shape[1])                                     # [N,total,2]
                X_np = win[:, :sequence_length, :].copy()                                         # [N,seq,2]
                y_np = win[:, sequence_length:, 0].copy()                                         # [N,out] future z only

            X_train = torch.from_numpy(X_np)
            y_train = torch.from_numpy(y_np)

            # Add Gaussian noise to X_train
            noise = torch.randn_like(X_train) * noise_std
            X_train_noisy = X_train + noise

            # Dataset/loader
            train_dataset = TensorDataset(X_train_noisy, y_train)
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            # ------------------- RESUME (per-config) -------------------
            start_epoch = 1
            final_resume_path = None
            final_resume_kind = None
            final_resume_epoch = None

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

                    # Weighted loss for first x_seconds and remaining
                    first_x_steps   = targets[:, :x_time_steps]
                    remaining_steps = targets[:, x_time_steps:]

                    loss_first_x    = criterion(outputs[:, :x_time_steps], first_x_steps) * w
                    loss_remaining  = criterion(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else 0.0

                    # Smoothness/continuity loss on output sequence
                    continuity_loss = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)

                    loss = loss_first_x + loss_remaining + 0.2 * continuity_loss
                    if not torch.isfinite(loss):
                        with open(model_log_path, 'a') as mlog:
                            mlog.write(f"[{now_str()}] Non-finite loss detected; skipping batch. "
                                       f"loss_first={float(loss_first_x)} "
                                       f"loss_rem={float(loss_remaining) if isinstance(loss_remaining, torch.Tensor) else loss_remaining} "
                                       f"cont={float(continuity_loss)}\n")
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
            print(f"Testing {model_name}: generating video '{video_out_path}' ...", file=sys.__stdout__)
            with torch.no_grad(), writer.saving(fig, video_out_path, dpi=VIDEO_DPI):
                for i in range(start_index, end_index):
                    noisy_series = np.full(T_test, np.nan, dtype=np.float32)  # <<< NEW (uses T_test)

                    # ---------- stateful, one point per step (+ test-time noise) ----------
                    if input_size == 1:
                        val_np = np.array([[[test_data[i]]]], dtype=np.float32)         # [1,1,1]
                    else:
                        val_np = np.array([[ test_data[i] ]], dtype=np.float32)          # [1,1,2] (z, vz)
                    val = torch.from_numpy(val_np).to(device)

                    input_tensor_noisy = val + torch.randn_like(val) * noise_std

                    # record the actual noisy z we fed (convert to cm for plotting)
                    if input_size == 1:
                        noisy_val_z = float(input_tensor_noisy.squeeze().detach().cpu().numpy())
                    else:
                        noisy_val_z = float(input_tensor_noisy.squeeze().detach().cpu().numpy()[0])  # channel 0 = z
                    noisy_val_cm = noisy_val_z * meters_to_cm
                    noisy_series[i] = noisy_val_cm

                    t0 = time.perf_counter()
                    output, h = model(input_tensor_noisy, h)  # stateful inference
                    h = [hh.detach() for hh in h]
                    t1 = time.perf_counter()
                    prediction_times.append(t1 - t0)

                    predicted = output.detach().cpu().numpy().flatten()

                    # ground truth future z (column 0 if using vel)
                    if input_size == 1:
                        true_future      = test_data[i + 1:i + 1 + output_size] * meters_to_cm
                    else:
                        true_future      = test_data[i + 1:i + 1 + output_size, 0] * meters_to_cm

                    predicted_future = predicted * meters_to_cm
                    abs_error        = np.abs(true_future - predicted_future)

                    absolute_errors.append(abs_error.mean())
                    if steps_3s > 0: errors_3s.append(float(np.mean(abs_error[:steps_3s])))
                    if steps_4s > 0: errors_4s.append(float(np.mean(abs_error[:steps_4s])))
                    if steps_5s > 0: errors_5s.append(float(np.mean(abs_error[:steps_5s])))

                    # ---- Build history window (exactly `sequence_length` points if available) ----
                    hist_start = max(0, i - sequence_length + 1)
                    hist_x = np.arange(hist_start, i + 1) / 20.0

                    if input_size == 1:
                        hist_clean_cm = test_data[hist_start : i + 1] * meters_to_cm
                    else:
                        hist_clean_cm = test_data[hist_start : i + 1, 0] * meters_to_cm  # z only

                    hist_noisy_cm = noisy_series[hist_start : i + 1]  # contains NaNs for early steps

                    # ---- Future time axis ----
                    fut_x = np.arange(i + 1, i + 1 + output_size) / 20.0

                    # ---- Draw frame: show history (clean + noisy) and future (true + predicted) ----
                    fig.clear()
                    ax1 = fig.add_subplot(2, 1, 1)

                    # history (clean) — what the underlying signal actually was
                    ax1.plot(hist_x, hist_clean_cm, 'k', linewidth=1.2, label=f'History clean ({len(hist_clean_cm)} steps)')

                    # history (noisy) — EXACT values fed to the model at each tick
                    mask = ~np.isnan(hist_noisy_cm)
                    if np.any(mask):
                        ax1.plot(hist_x[mask], hist_noisy_cm[mask], '--', linewidth=1.0, label='History noisy (fed)', alpha=0.9)

                    # future (true vs predicted)
                    ax1.plot(fut_x, true_future, 'g--', linewidth=1.2, label=f'True (+{output_size/20:.1f}s)')
                    ax1.plot(fut_x, predicted_future, 'r', linewidth=1.2, label='Predicted')

                    # lock x-range to show exactly history+future
                    xmin = (i - sequence_length + 1) / 20.0
                    xmax = (i + output_size) / 20.0
                    ax1.set_xlim(xmin, xmax)
                    ax1.set_ylim(-30, 30)
                    ax1.set_title(f"t = {i/20.0:.2f}s | Window: {sequence_length} hist + {output_size} fut")
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Position (cm)')
                    ax1.legend(loc='upper left')

                    # annotate per-tick prediction time
                    if prediction_times:
                        ax1.text(0.01, 0.95,
                                f"pred time: {prediction_times[-1]*1000:.2f} ms",
                                transform=ax1.transAxes, va='top', ha='left')

                    # error subplot
                    ax2 = fig.add_subplot(2, 1, 2)
                    ax2.plot(fut_x, abs_error, 'b', linewidth=1.2, label='Absolute Error (cm)')
                    ax2.set_xlim(xmin, xmax)
                    ax2.set_ylim(0, 15)
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Error (cm)')
                    ax2.legend(loc='upper left')

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
                mlog.write(f"Final checkpoint used: {final_ckpt_display}\n")
                mlog.write(f"Test video saved: {video_out_path}\n")
                mlog.write(f"Average Prediction Time: {avg_prediction_time:.6f} s\n")
                mlog.write(f"Average Absolute Error (Total): {avg_error:.6f} cm\n")
                mlog.write(f"Average Absolute Error (3s): {avg_error_3s:.6f} cm\n")
                mlog.write(f"Average Absolute Error (4s): {avg_error_4s:.6f} cm\n")
                mlog.write(f"Average Absolute Error (5s): {avg_error_5s:.6f} cm\n")
                mlog.write('-'*80 + '\n\n')
                mlog.flush()

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
    # <<< NEW: enable z+velocity inputs from train_data_normalised_with_vel/
    parser.add_argument(
        "--vel",
        action="store_true",
        help="Use z + velocity inputs from 'train_data_normalised_with_vel/'."
    )
    return parser

# Entry point
if __name__ == '__main__':
    args = _build_argparser().parse_args()
    # override globals before main() uses them
    RESUME = bool(args.resume)
    RESUME_PATH = args.resume_path or RESUME_PATH
    USE_VEL = bool(args.vel)  # <<< NEW
    main()
