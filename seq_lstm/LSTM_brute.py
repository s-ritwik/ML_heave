# -*- coding: utf-8 -*-
"""
Noisy LSTM training & evaluation with periodic checkpointing and file logging.

Key features:
- Saves to `noisyLSTM_models_seq/<model_name>/`
- Full & weights-only checkpoints every SAVE_EVERY epochs
- File-only logging; no console prints (run logs + per-model logs + error log)
- Exponential / Step / Cosine LR schedulers
- Weighted loss for early-horizon steps + continuity regularizer
- Streaming one-tick inference using LSTM hidden states (h, c)
- Test-time video with predicted vs. ground-truth traces

NEW:
- Resume flag: if RESUME=True, for each config we auto-detect the newest checkpoint
  (.pt preferred, else .pth) in that config’s folder and resume from it.
  If RESUME_PATH is set, it takes priority.
"""
try:
    import os
    import re
    import sys
    import json
    import time
    import glob
    import torch
    import argparse
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
    from sklearn.preprocessing import MinMaxScaler  # kept for compatibility
except Exception as e:
    print(f"{e} Some imports failed. Ensure you have the required packages installed.")
# -------------------------------------------------------------------------
# -------------------------- GLOBAL CONSTANTS ------------------------------
# -------------------------------------------------------------------------

# Root directories
MODEL_ROOT_DIR = 'noisyLSTM_models_seq'
PLOT_DIR       = 'noisyprediction_videos'
LOG_DIR        = 'log'

# Ensure directories exist
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# GPU selection (0 or 1)
GPU_INDEX = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global data/config files
CONFIG_FILE_PATH   = 'model_configs_1.txt'
MODEL_SUMMARY_PATH = 'model_summary.txt'

# Training data locations
TRAIN_DIR      = 'train_data_normalised'
TRAIN_MOCAP_DIR = 'train_data_normalised_mocap'  # used for test file below

# Testing
TEST_FILE_PATH = os.path.join(TRAIN_MOCAP_DIR, 'D1H3_normalised.csv')

# Noise and scaling constants
NOISE_STD_DEFAULT = 0.05        # Gaussian noise std on inputs
METERS_TO_CM      = 25          # project-specific conversion factor

# Saving / logging cadence
SAVE_EVERY = 20                 # epochs
LOG_EVERY  = 10                 # epochs (same as SAVE_EVERY)

# Video writer settings
VIDEO_FPS = 20
VIDEO_DPI = 100

# --------------------- RESUME CONTROLS ---------------------
RESUME = False
RESUME_PATH = ""

def _build_argparser():
    parser = argparse.ArgumentParser(description="Noisy LSTM training & evaluation")
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
    return parser


# # ======== DEBUG HARNESS (paste after imports) ========
# import os, sys, logging, signal, faulthandler, atexit, traceback, time
# LOGFILE = os.path.expanduser("~/LSTM_debug.log")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # catch async CUDA errors

# # log to both console and file
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout),
#               logging.FileHandler(LOGFILE, mode="a")]
# )
# log = logging.getLogger("GRU_BRUTE")
# log.info("PID=%s starting; log=%s", os.getpid(), LOGFILE)

# # faulthandler will dump traces on fatal signals
# faulthandler.enable(open(LOGFILE, "a"))  # also mirrors to stderr

# # dump on demand: kill -USR1 <pid>
# try:
#     faulthandler.register(signal.SIGUSR1, file=open(LOGFILE, "a"), all_threads=True)
# except Exception:
#     pass

# # catch ANY uncaught exception
# def _excepthook(exc_type, exc, tb):
#     log.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
#     # ensure we see something even if logging breaks
#     traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
#     os._exit(1)
# sys.excepthook = _excepthook

# # log on SIGTERM/SIGINT to know if something external stops us
# def _sig_handler(signum, frame):
#     log.error("Received signal %s; dumping stack...", signum)
#     traceback.print_stack(frame, file=sys.stderr)
#     faulthandler.dump_traceback(file=open(LOGFILE, "a"), all_threads=True)
#     os._exit(128 + signum)
# for s in (signal.SIGTERM, signal.SIGINT):
#     try: signal.signal(s, _sig_handler)
#     except Exception: pass

# # heartbeat so you know if we hang vs exit
# def _heartbeat():
#     log.debug("heartbeat: alive at %s", time.strftime("%H:%M:%S"))
#     sys.stdout.flush()
#     sys.stderr.flush()
#     # schedule another beat
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
       'sequence_length:400; output_size:160; hidden_sizes:[512,256];
        x_seconds:3; w:2; batch_size:64; epochs:120; learning_rate:0.001;
        lr_scheduler:step; step_size:40; gamma:0.5'
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
            config[key] = value
    return config

def load_data(file_list, data_dir=TRAIN_DIR):
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

# --------------------------- Resume helpers ---------------------------

def _extract_epoch(path):
    """Extract epoch number from filenames like 'epoch_123.pt' or 'epoch_123.pth'."""
    m = re.search(r'epoch_(\d+)\.(pt|pth)$', os.path.basename(path))
    return int(m.group(1)) if m else None

def find_latest_checkpoint_any(model_folder):
    """
    Find the newest checkpoint in `model_folder` among *.pt and *.pth.
    Return a tuple: (path, kind, epoch) where kind in {'pt','pth'}; or (None, None, None) if none found.
    Preference is simply by highest epoch; if tie, prefer .pt.
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

    # sort by epoch asc, then prefer pt over pth at same epoch
    candidates.sort(key=lambda t: (t[0], 0 if t[1] == 'pt' else 1))
    ep, kind, path = candidates[-1]
    return path, kind, ep

# -------------------------------------------------------------------------
# ---------------------------- MODEL DEFINITION ---------------------------
# -------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.lstm_layers = nn.ModuleList()
        # first layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        # stacked layers
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i - 1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, state):
        """
        x:     [B, T, 1]  (T can be 1 during streaming ticks)
        state: list of (h, c) tuples, one per layer, where each h/c is [1, B, H_i]
        """
        next_state = []
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            h_i, c_i = state[i]
            out, (h_o, c_o) = lstm(out, (h_i, c_i))
            next_state.append((h_o, c_o))
        # last timestep features -> horizon
        out = out[:, -1, :]           # [B, H_last]
        out = self.fc(out)            # [B, output_size]
        out = self.tanh(out)
        return out, next_state

    def init_hidden(self, batch_size):
        """Returns list[(h0,c0), ...] on correct device."""
        return [
            (torch.zeros(1, batch_size, hs, device=device),
             torch.zeros(1, batch_size, hs, device=device))
            for hs in self.hidden_sizes
        ]

    @staticmethod
    def detach_state(state):
        """Detach per-layer (h, c) to break graph between iterations."""
        return [(h.detach(), c.detach()) for (h, c) in state]

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
            model_name   = f"noisy_D1_LSTM_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}"
            print("Training for:", model_name, file=sys.__stdout__)  # also print to original stdout
            model_folder = os.path.join(MODEL_ROOT_DIR, model_name)
            os.makedirs(model_folder, exist_ok=True)

            # Initialize model/optimizer/etc.
            model = LSTMModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = get_scheduler(optimizer, config)

            # Count parameters + setup per-model log
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
                mlog.write('='*80 + '\n')
                mlog.flush()

            # --------------------- Prepare training data ----------------------
            csv_files   = os.listdir(TRAIN_DIR)
            train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
            train_data  = load_data(train_files, data_dir=TRAIN_DIR)

            # Build sequences with sliding window
            total = sequence_length + output_size
            if len(train_data) < total:
                raise ValueError(f"Training data too short ({len(train_data)}) for total window size {total}.")
            windows = np.lib.stride_tricks.sliding_window_view(train_data, total)  # [N, total]
            X_np = windows[:, :sequence_length][..., None].copy()  # [N, seq_len, 1]
            y_np = windows[:, sequence_length:].copy()              # [N, out_size]

            X_train = torch.from_numpy(X_np)
            y_train = torch.from_numpy(y_np)

            # Add Gaussian noise to X_train
            noise = torch.randn_like(X_train) * noise_std
            X_train_noisy = X_train + noise

            # Dataset/loader
            train_dataset = TensorDataset(X_train_noisy, y_train)
            train_loader  = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available()
            )

            # ------------------- RESUME (per-config) -------------------
            start_epoch = 1
            final_resume_path = None
            final_resume_kind = None
            final_resume_epoch = None

            if RESUME:
                if RESUME_PATH and os.path.isfile(RESUME_PATH):
                    # Explicit resume path has priority
                    final_resume_path = RESUME_PATH
                    final_resume_kind = 'pt' if RESUME_PATH.endswith('.pt') else 'pth' if RESUME_PATH.endswith('.pth') else None
                    final_resume_epoch = _extract_epoch(RESUME_PATH)
                else:
                    # Auto-detect newest ckpt in this config's folder
                    rp, rk, re = find_latest_checkpoint_any(model_folder)
                    final_resume_path, final_resume_kind, final_resume_epoch = rp, rk, re

                if final_resume_path and final_resume_kind:
                    if final_resume_kind == 'pt':
                        ckpt = torch.load(final_resume_path, map_location=device)
                        model.load_state_dict(ckpt["model_state_dict"])
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        if scheduler and ckpt.get("scheduler_state_dict"):
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        # prefer stored epoch; fallback to parsed epoch
                        stored_epoch = int(ckpt.get("epoch", final_resume_epoch or 0))
                        start_epoch = stored_epoch + 1
                    elif final_resume_kind == 'pth':
                        state_dict = torch.load(final_resume_path, map_location=device)
                        model.load_state_dict(state_dict)
                        # only weights available; parse epoch from filename if possible
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
            x_time_steps = min(x_seconds * 20, output_size)  # clamp to horizon

            for epoch in range(start_epoch, epochs + 1):
                model.train()
                total_loss = 0.0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    h = model.init_hidden(inputs.size(0))
                    optimizer.zero_grad()

                    outputs, h = model(inputs, h)
                    h = model.detach_state(h)

                    # Weighted loss for first x_seconds and remaining
                    first_x_steps   = targets[:, :x_time_steps]
                    remaining_steps = targets[:, x_time_steps:]

                    loss_first_x    = nn.functional.mse_loss(outputs[:, :x_time_steps], first_x_steps) * w
                    loss_remaining  = nn.functional.mse_loss(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else torch.tensor(0.0, device=device)

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
            print(f"Training completed for {model_name} in {training_time:.2f} seconds.", file=sys.__stdout__)
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
                # fall back to weights of final epoch
                final_ckpt_pth = os.path.join(model_folder, f"epoch_{epochs}.pth")
                model.load_state_dict(torch.load(final_ckpt_pth, map_location=device))
                final_ckpt_display = final_ckpt_pth

            model.eval()

            prediction_times = []
            absolute_errors  = []
            steps_3s = min(3 * 20, output_size)
            steps_4s = min(4 * 20, output_size)
            steps_5s = min(5 * 20, output_size)
            errors_3s, errors_4s, errors_5s = [], [], []
             # Streaming setup
            total_steps = len(test_data/100) - sequence_length - output_size - 1
            start_index = sequence_length
            end_index   = start_index + max(0, total_steps)
            if end_index + output_size > len(test_data):
                end_index = len(test_data) - output_size

            h = model.init_hidden(1)  # batch size 1 for inference

            # Video writer
            fig = plt.figure(figsize=(12, 8))
            writer = animation.FFMpegWriter(fps=VIDEO_FPS)
            video_out_path = os.path.join(PLOT_DIR, f"{model_name}.mp4")
            print("Generating test video:", video_out_path, file=sys.__stdout__)
            with torch.no_grad(), writer.saving(fig, video_out_path, dpi=VIDEO_DPI):
                for i in range(start_index, end_index):
                    # one point per step (streaming) + test-time noise
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
                    true_future      = test_data[i + 1:i + 1 + output_size] * meters_to_cm
                    predicted_future = predicted * meters_to_cm
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
                f"Average Prediction Time: {avg_prediction_time:.6f} s\n"
                f"Average Absolute Error (Total): {avg_error:.6f} cm\n"
                f"Average Absolute Error (3s): {avg_error_3s:.6f} cm\n"
                f"Average Absolute Error (4s): {avg_error_4s:.6f} cm\n"
                f"Average Absolute Error (5s): {avg_error_5s:.6f} cm\n"
                f"------------------------------------------------------------\n\n"
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

# Entry point
if __name__ == '__main__':
    args = _build_argparser().parse_args()
    # override globals before main() uses them
    RESUME = bool(args.resume)
    RESUME_PATH = args.resume_path or RESUME_PATH
    main()
