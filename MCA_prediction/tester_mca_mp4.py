#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tester_mca_mp4.py
-----------------
Visual tester for the MCA linear predictor trained by mca_brute.py.
Loads a saved model directory (containing W.npy, mu.npy, config.json),
runs rolling multi-step predictions on D1H3 (by default), and exports an MP4
showing history, predicted future vs ground truth, and per-horizon absolute error.

Example:
    python tester_mca_mp4.py --model-dir ./mca_models/MCA_n800_m120_stride1_P6_cut0.015_ridge1e-06 \
                             --test-time 150 \
                             --sampling-rate 20

Options:
    --model-dir       Path to saved model folder from mca_brute.py (required)
    --test-file       CSV to test on (defaults to config["test_file"] or /mnt/data/D1H3_normalised.csv)
    --sampling-rate   Frames-per-second and time axis scaling (default 20)
    --test-time       Duration in seconds to render (default 150)
    --noise-std       Optional Gaussian noise std to add to the input history (in data units) (default 0.0)
    --meters-to-cm    Scale factor for plotting (default 25.0)
    --ylim-error      Y-limit (max) for bottom error plot (default 18.0)
    --outdir          Directory to save the MP4 (default: <model_dir>/videos)
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# ------------------------------ Helpers ------------------------------------
def robust_read_numeric_series(csv_path: str, col_index: int = 0) -> np.ndarray:
    """
    Load a CSV and return 1D numpy array for the selected numeric column.
    Tries normal read (with dtypes), then falls back to header=None.
    """
    try:
        df = pd.read_csv(csv_path)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            # Fall back: assume no header and first column is numeric
            df2 = pd.read_csv(csv_path, header=None)
            return df2.iloc[:, col_index].astype(float).to_numpy()
        if col_index < 0 or col_index >= num_df.shape[1]:
            raise ValueError(f"col_index {col_index} out of range for file {csv_path} with {num_df.shape[1]} numeric cols.")
        return num_df.iloc[:, col_index].astype(float).to_numpy()
    except Exception:
        # Final fallback
        df2 = pd.read_csv(csv_path, header=None)
        return df2.iloc[:, col_index].astype(float).to_numpy()


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


def build_output_path(model_dir: str, sampling_rate: int, test_time: int, n: int, m: int) -> str:
    vids_dir = os.path.join(model_dir, "videos")
    os.makedirs(vids_dir, exist_ok=True)
    model_name = os.path.basename(model_dir.rstrip("/"))
    fname = f"{model_name}_{sampling_rate}Hz_{test_time}s_n{n}_m{m}.mp4"
    return os.path.join(vids_dir, fname)


# ------------------------------- Main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=str, help="Folder with W.npy, mu.npy, config.json")
    ap.add_argument("--test-file", type=str, default=None, help="CSV to test on (defaults to config['test_file'] or D1H3_normalised.csv)")
    ap.add_argument("--sampling-rate", type=int, default=20, help="Samples per second (fps for video)")
    ap.add_argument("--test-time", type=int, default=150, help="Seconds of video to write")
    ap.add_argument("--noise-std", type=float, default=0.0, help="Stddev of Gaussian noise added to input history")
    ap.add_argument("--meters-to-cm", type=float, default=25.0, help="Scale factor from data units to cm for plotting")
    ap.add_argument("--ylim-error", type=float, default=18.0, help="Y max for error subplot")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory (default: <model_dir>/videos)")
    args = ap.parse_args()

    # Load model
    W, mu, config, n, m, centered, col_index = load_mca_model(args.model_dir)

    # Determine test file
    test_file = args.test_file or config.get("test_file") or "/mnt/data/D1H3_normalised.csv"
    series = robust_read_numeric_series(test_file, col_index=col_index).astype(float)

    sr = int(args.sampling_rate)
    T_total = int(args.test_time)
    meters_to_cm = float(args.meters_to_cm)

    # Video output path
    out_mp4 = build_output_path(args.model_dir, sr, T_total, n, m)
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        out_mp4 = os.path.join(args.outdir, os.path.basename(out_mp4))

    # Indexing for streaming-like loop
    start_i = n - 1
    max_i   = len(series) - 1 - m  # ensure future [i+1 .. i+m] exists
    if max_i < start_i:
        raise RuntimeError("Test series too short for given n and m.")

    total_steps = min(T_total * sr, max_i - start_i + 1)

    # For metrics & plotting
    steps_3s, steps_4s, steps_5s = min(3*sr, m), min(4*sr, m), min(5*sr, m)
    pred_times = []
    mean_abs_errors = []
    err_3s, err_4s, err_5s = [], [], []

    # Matplotlib setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    metadata = dict(title='MCA forecast', artist='Matplotlib', comment='MCA rolling predictions')
    writer   = FFMpegWriter(fps=sr, metadata=metadata)

    t_hist_base   = np.arange(-n+1, 1) / sr      # [-Tin+1..0] s
    t_future_base = np.arange(1, m+1) / sr       # [1..m]/sr s
    Tin_s, Tout_s = n / sr, m / sr

    t0_global = time.time()
    with writer.saving(fig, out_mp4, dpi=300):
        for step in range(total_steps):
            i = start_i + step

            # Prepare input history
            x1 = series[i-n+1:i+1].copy()  # length n
            if args.noise_std > 0.0:
                x1 = x1 + np.random.normal(0.0, args.noise_std, size=x1.shape)

            # Predict
            t0 = time.perf_counter()
            yhat = predict_once(W, x1, mu=mu, n=n, m=m, centered=centered)
            t1 = time.perf_counter()
            pred_times.append(t1 - t0)

            ytrue = series[i+1:i+1+m]

            # Errors
            abs_err = np.abs((yhat - ytrue) * meters_to_cm)  # in cm for plotting
            mean_abs_errors.append(abs_err.mean())
            err_3s.append(abs_err[:steps_3s].mean())
            err_4s.append(abs_err[:steps_4s].mean())
            err_5s.append(abs_err[:steps_5s].mean())

            # ------------- Plotting -------------
            ax1.clear(); ax2.clear()

            hist_cm = x1 * meters_to_cm
            ax1.plot(t_hist_base, hist_cm, label='Input history (cm)')
            ax1.plot(t_future_base, ytrue * meters_to_cm, 'g--', label='True future (cm)')
            ax1.plot(t_future_base, yhat  * meters_to_cm, 'r', label='Predicted (cm)')
            ax1.axvline(0.0, linestyle=':', linewidth=1)
            ax1.set_ylim(-25, 25)
            ax1.set_xlim(-Tin_s, Tout_s)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (cm)')
            ax1.legend(loc='upper left')

            total_elapsed = time.time() - t0_global
            avg_ms = (np.mean(pred_times)*1000.0) if pred_times else 0.0
            cur_ms = (t1 - t0)*1000.0
            fig.suptitle(
                f"Elapsed: {total_elapsed:.2f}s / {T_total}s   |   Model: {os.path.basename(args.model_dir)}",
                fontsize=12
            )
            ax1.text(
                0.99, 0.02,
                f"Pred time: {cur_ms:.2f} ms  (avg {avg_ms:.2f} ms)\nNoise σ={args.noise_std:.3f}",
                transform=ax1.transAxes, ha='right', va='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')
            )

            # Error subplot
            ax2.plot(t_future_base, abs_err, label='Absolute error (cm)')
            ax2.axvline(0.0, linestyle=':', linewidth=1)
            ax2.set_xlim(0.0, Tout_s)
            ax2.set_ylim(0.0, float(args.ylim_error))
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Error (cm)')
            ax2.legend(loc='upper left')

            writer.grab_frame()

    plt.close(fig)

    # --------------------------- METRICS & PRINTS -----------------------------
    if mean_abs_errors:
        avg_pred_time = float(np.mean(pred_times))
        avg_abs_err   = float(np.mean(mean_abs_errors))
        avg_err_3s    = float(np.mean(err_3s))
        avg_err_4s    = float(np.mean(err_4s))
        avg_err_5s    = float(np.mean(err_5s))

        print(f"\nSaved video: {out_mp4}")
        print(f"Average Prediction Time: {avg_pred_time*1000.0:.3f} ms")
        print(f"Average Absolute Error (first 3s): {avg_err_3s:.4f} cm")
        print(f"Average Absolute Error (first 4s): {avg_err_4s:.4f} cm")
        print(f"Average Absolute Error (first 5s): {avg_err_5s:.4f} cm")
        print(f"Total Average Absolute Error: {avg_abs_err:.4f} cm")
    else:
        print("\nNo errors were recorded. Ensure test duration and indices are valid.")
        print(f"Saved video: {out_mp4}")


if __name__ == "__main__":
    main()
