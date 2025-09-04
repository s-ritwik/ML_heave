#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCA-based linear predictor for ship deck motion
------------------------------------------------
Replicates the Minor Component Analysis (MCA) method described in the paper,
using your dataset organization and windowing (n past, m future) similar to GRU_brute.py.

Features
- Hyperparameters at top or via CLI (stride, energy cutoff for minor components, ridge lambda, etc.)
- Trains on D1H1/2/4/5 (default) and evaluates on D1H3 (default)
- Uses SVD to find minor components of the autocorrelation matrix
- Builds a closed-form linear predictor W to map X1 (past) -> X2 (future)
- Reports overall and per-horizon MAE/RMSE
- Saves model artifacts (W, mean vector, config) in an output folder

Example (defaults):
    python mca_brute.py

Override some settings:
    python mca_brute.py --stride-train 5 --energy-cutoff 0.015 --ridge 1e-6

Specify P directly (overrides energy cutoff):
    python mca_brute.py --P 8

Author: ChatGPT (MCA replication for your GRU_brute workflow)
"""

import os
import json
import time
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------ Defaults ------------------------------------
DEFAULT_TRAIN_FILES = [
    "MCA_prediction/train_data_normalised_mocap/D1H1_normalised.csv",
    "MCA_prediction/train_data_normalised_mocap/D1H2_normalised.csv",
    "MCA_prediction/train_data_normalised_mocap/D1H4_normalised.csv",
    "MCA_prediction/train_data_normalised_mocap/D1H5_normalised.csv",
]
DEFAULT_TEST_FILE = "MCA_prediction/train_data_normalised_mocap/D1H3_normalised.csv"

# Window sizes
DEFAULT_N = 800   # past samples (40 s @ 20 Hz)
DEFAULT_M = 120   # future samples (6 s @ 20 Hz)

# Strides
DEFAULT_STRIDE_TRAIN = 5
DEFAULT_STRIDE_TEST = 5

# MCA settings
DEFAULT_CENTER = True          # subtract training mean per time-offset (row-wise mean)
DEFAULT_ENERGY_CUTOFF = 0.015  # fraction of total energy assigned to minor subspace
DEFAULT_P = None               # if not None, overrides energy cutoff and uses exactly P minor comps
DEFAULT_RIDGE = 1e-6           # Tikhonov regularization for stability in (B2^T B2 + λI)^{-1}

# Data column selection
DEFAULT_COL_INDEX = 0          # which numeric column to use if CSV has multiple numeric columns

# Output
DEFAULT_OUTDIR = "./mca_models"
DEFAULT_MODEL_NAME = None      # if None, auto-generate from hyperparameters

# Save models next to this script, under MCA_prediction/models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, "models")
# ------------------------------- Utils --------------------------------------
def read_numeric_series(csv_path: str, col_index: int = 0) -> np.ndarray:
    """Load a CSV and return 1D numpy array for the selected numeric column.
    If multiple numeric columns exist, choose by index (default 0)."""
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {csv_path}.")
    if col_index < 0 or col_index >= num_df.shape[1]:
        raise ValueError(f"col_index {col_index} out of range for file {csv_path} with {num_df.shape[1]} numeric cols.")
    arr = num_df.iloc[:, col_index].to_numpy(dtype=float)
    return np.asarray(arr)


def build_windows(series: np.ndarray, n: int, m: int, stride: int) -> np.ndarray:
    """Construct overlapping windows of length n+m from a 1D series with a given stride.
    Returns an array of shape (n+m, K) where K is the number of windows."""
    N = n + m
    T = series.shape[0]
    if T < N:
        return np.zeros((N, 0), dtype=float)
    starts = range(0, T - N + 1, stride)
    K = math.floor((T - N) / stride) + 1
    X = np.empty((N, K), dtype=float)
    k = 0
    for s in starts:
        X[:, k] = series[s:s+N]
        k += 1
    return X


def concat_train_windows(files: List[str], n: int, m: int, stride: int, col_index: int) -> np.ndarray:
    """Read all training files and horizontally stack all windows into X_all (N x M_total)."""
    mats = []
    for f in files:
        s = read_numeric_series(f, col_index=col_index)
        X = build_windows(s, n, m, stride)
        if X.shape[1] > 0:
            mats.append(X)
    if len(mats) == 0:
        return np.zeros((n+m, 0), dtype=float)
    return np.concatenate(mats, axis=1)


def center_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the mean over samples for each row (time-offset)."""
    mu = X.mean(axis=1, keepdims=True) if X.shape[1] > 0 else np.zeros((X.shape[0], 1))
    return X - mu, mu.squeeze()


def select_minor_components(eigvals: np.ndarray,
                            U: np.ndarray,
                            energy_cutoff: float,
                            P: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Select minor component indices.
    If P is provided, pick the P smallest eigenvalues. Otherwise, pick the smallest number whose
    cumulative energy fraction <= energy_cutoff (e.g., 0.015 for ~1.5%)."""
    N = eigvals.shape[0]
    order = np.argsort(eigvals)  # ascending: smallest first
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
        # Degenerate case: fall back to single minor comp
        idx_minor = order[:1]
        return idx_minor, U[:, idx_minor]
    csum = 0.0
    chosen = []
    for j in order:
        csum += eigvals[j]
        chosen.append(j)
        if (csum / total) >= energy_cutoff:
            break
    # Ensure at least 1 component
    if len(chosen) == 0:
        chosen = [order[0]]
    idx_minor = np.array(chosen, dtype=int)
    return idx_minor, U[:, idx_minor]


def compute_W_from_B(B: np.ndarray, n: int, m: int, ridge: float) -> np.ndarray:
    """Given B (P x (n+m)), split B into B1 (P x n) and B2 (P x m) and compute
    W = -(B2^T B2 + λI)^{-1} B2^T B1  of shape (m x n)."""
    P, N = B.shape
    assert N == n + m, "B must have shape (P, n+m)"
    B1 = B[:, :n]
    B2 = B[:, n:]
    # Shapes: B2^T B2 => (m x m), B2^T B1 => (m x n)
    BtB = B2.T @ B2
    BtB_reg = BtB + ridge * np.eye(BtB.shape[0])
    BtB_B1 = B2.T @ B1
    W = -np.linalg.solve(BtB_reg, BtB_B1)
    return W


def predict_block(W: np.ndarray, X1: np.ndarray, mu_full: np.ndarray, n: int, m: int, centered: bool) -> np.ndarray:
    """Predict future block for many windows at once.
    X1: shape (n, K), returns X2_hat: shape (m, K).
    If centered=True, we assume training used row-centering and we must subtract mu[:n] and add mu[n:] back."""
    if centered:
        mu1 = mu_full[:n].reshape(n, 1)
        mu2 = mu_full[n:].reshape(m, 1)
        X1c = X1 - mu1
        X2c_hat = W @ X1c
        return X2c_hat + mu2
    else:
        return W @ X1


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE overall and per-horizon.
    y_* shapes: (m, K)."""
    assert y_true.shape == y_pred.shape
    m, K = y_true.shape
    diff = y_pred - y_true
    mae_h = np.mean(np.abs(diff), axis=1)          # per horizon
    rmse_h = np.sqrt(np.mean(diff**2, axis=1))     # per horizon
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = float(np.sqrt(np.mean(diff**2)))
    return {
        "mae_all": mae_all,
        "rmse_all": rmse_all,
        "mae_h": mae_h.tolist(),
        "rmse_h": rmse_h.tolist(),
    }


def save_model(outdir: str,
               model_name: str,
               W: np.ndarray,
               mu: np.ndarray,
               idx_minor: np.ndarray,
               config: dict):
    """Save artifacts: W, mu, idx_minor, config.json."""
    model_dir = os.path.join(outdir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "W.npy"), W)
    np.save(os.path.join(model_dir, "mu.npy"), mu)
    np.save(os.path.join(model_dir, "idx_minor.npy"), idx_minor)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return model_dir


# ------------------------------- Main ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MCA linear predictor trainer/evaluator")
    parser.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES, help="Training CSV files")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE, help="Test CSV file")
    parser.add_argument("--col-index", type=int, default=DEFAULT_COL_INDEX, help="Numeric column index to use")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Past window length")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="Future window length")
    parser.add_argument("--stride-train", type=int, default=DEFAULT_STRIDE_TRAIN, help="Training stride")
    parser.add_argument("--stride-test", type=int, default=DEFAULT_STRIDE_TEST, help="Test stride")
    parser.add_argument("--center", action="store_true", default=DEFAULT_CENTER, help="Enable row-centering on training windows")
    parser.add_argument("--no-center", dest="center", action="store_false", help="Disable centering")
    parser.add_argument("--energy-cutoff", type=float, default=DEFAULT_ENERGY_CUTOFF, help="Minor energy fraction cutoff (ignored if P given)")
    parser.add_argument("--P", type=int, default=None, help="Number of minor components to use (overrides energy cutoff)")
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE, help="Ridge (Tikhonov) regularization")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model name; autogenerated if None")
    args = parser.parse_args()

    n, m = args.n, args.m
    N = n + m

    t0 = time.time()
    print("=== MCA Linear Predictor Training ===")
    print(f"Train files: {args.train_files}")
    print(f"Test file:   {args.test_file}")
    print(f"Using numeric column index: {args.col_index}")
    print(f"Window sizes: n={n}, m={m} (N={N})")
    print(f"Strides: train={args.stride_train}, test={args.stride_test}")
    print(f"Center rows: {args.center}")
    print(f"Energy cutoff (minor): {args.energy_cutoff}  |  P override: {args.P}")
    print(f"Ridge lambda: {args.ridge}")

    # -------------------- Load and assemble training windows -----------------
    X_all = concat_train_windows(args.train_files, n, m, args.stride_train, args.col_index)
    M_total = X_all.shape[1]
    if M_total == 0:
        raise RuntimeError("No training windows constructed. Check file paths/lengths/stride.")
    print(f"Training windows: {M_total}  (matrix shape: {X_all.shape})")

    # -------------------- Center (optional) ----------------------------------
    if args.center:
        Xc, mu = center_rows(X_all)
    else:
        Xc = X_all.copy()
        mu = np.zeros((N,), dtype=float)

    # -------------------- SVD on training matrix -----------------------------
    # SVD of Xc / sqrt(M) is numerically in line with eigen of sample autocovariance
    print("Computing SVD...")
    t_svd0 = time.time()
    # Use economy SVD; Xc is (N x M_total)
    # Scale by sqrt(M_total) so that eigvals ~ variance
    Xs = Xc / math.sqrt(M_total)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    eigvals = S**2  # proportional to eigenvalues of R
    t_svd1 = time.time()
    print(f"SVD done in {t_svd1 - t_svd0:.3f} s")

    # -------------------- Select minor components ----------------------------
    idx_minor, U_minor = select_minor_components(eigvals, U, args.energy_cutoff, args.P)
    P_used = idx_minor.shape[0]
    print(f"Minor components selected: P={P_used} | indices: {idx_minor.tolist()}")
    # Form B = U_minor^T  with shape (P x N)
    B = U_minor.T

    # -------------------- Compute W ------------------------------------------
    print("Computing W = -(B2^T B2 + λI)^{-1} B2^T B1 ...")
    W = compute_W_from_B(B, n=n, m=m, ridge=args.ridge)
    print(f"W shape: {W.shape}")

    # -------------------- Save model -----------------------------------------
    if args.model_name is None:
        model_name = f"MCA_n{n}_m{m}_stride{args.stride_train}_P{P_used}_cut{args.energy_cutoff}_ridge{args.ridge}"
    else:
        model_name = args.model_name

    config = {
        "train_files": args.train_files,
        "test_file": args.test_file,
        "col_index": args.col_index,
        "n": n,
        "m": m,
        "stride_train": args.stride_train,
        "stride_test": args.stride_test,
        "center": args.center,
        "energy_cutoff": args.energy_cutoff,
        "P_used": int(P_used),
        "ridge": args.ridge,
    }
    model_dir = save_model(args.outdir, model_name, W=W, mu=mu, idx_minor=idx_minor, config=config)
    print(f"Model saved to: {model_dir}")

    # -------------------- Evaluation on test file ----------------------------
    print("\n=== Evaluation on test set ===")
    series_test = read_numeric_series(args.test_file, col_index=args.col_index)
    X_test = build_windows(series_test, n, m, args.stride_test)
    K_test = X_test.shape[1]
    if K_test == 0:
        raise RuntimeError("No test windows constructed. Check test file length/stride.")
    X1 = X_test[:n, :]
    Y_true = X_test[n:, :]
    Y_pred = predict_block(W, X1, mu_full=mu, n=n, m=m, centered=args.center)

    metrics = compute_metrics(Y_true, Y_pred)

    print(f"Test windows: {K_test}")
    print(f"MAE (overall):  {metrics['mae_all']:.6f}")
    print(f"RMSE (overall): {metrics['rmse_all']:.6f}")

    # Print a few horizon checkpoints (1s, 3s, 6s) if plausible
    hz = 20  # 20 Hz sampling
    for sec in [1, 3, 6]:
        idx = min(sec*hz - 1, m-1)
        mae_s = metrics["mae_h"][idx]
        rmse_s = metrics["rmse_h"][idx]
        print(f"H={sec:>2d}s  step={idx+1:>3d}:  MAE={mae_s:.6f}  RMSE={rmse_s:.6f}")

    # Also save predictions for further plotting/analysis
    np.savez(os.path.join(model_dir, "test_predictions.npz"),
             Y_true=Y_true, Y_pred=Y_pred, X1=X1)

    dt = time.time() - t0
    print(f"\nDone in {dt:.3f} s.")
    print("Artifacts saved:")
    print(f" - {os.path.join(model_dir, 'W.npy')}")
    print(f" - {os.path.join(model_dir, 'mu.npy')}")
    print(f" - {os.path.join(model_dir, 'idx_minor.npy')}")
    print(f" - {os.path.join(model_dir, 'config.json')}")
    print(f" - {os.path.join(model_dir, 'test_predictions.npz')}")

if __name__ == "__main__":
    main()
