#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCA-based linear predictor with P and lambda tuning
---------------------------------------------------
Extends your MCA script to grid-search P (minor components) and ridge λ on a
temporal validation split built from the training files, then evaluates the
best model on the test file.

Defaults:
- Train on D1H1/2/4/5, test on D1H3
- n=800, m=120, stride=1, centering on
- P grid = [4, 8, 12, 16, 24]
- ridge grid = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
- Validation split = last 20% windows of each training file

Artifacts:
- Saves best W, mu, idx_minor, and config.json under ./models/<auto-name>
- Saves test predictions to test_predictions.npz

Run:
  python mca_tune.py
  python mca_tune.py --P-grid 6,10,14,18,22 --ridge-grid 1e-7,1e-6,1e-5 --val-split 0.25
"""

import os
import json
import time
import math
import argparse
from typing import List, Tuple, Optional, Dict

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
DEFAULT_STRIDE_TRAIN = 1
DEFAULT_STRIDE_TEST  = 1

# MCA settings
DEFAULT_CENTER = True          # row-centering
DEFAULT_ENERGY_CUTOFF = 0.01   # used only if you pass --use-cutoff
DEFAULT_P = None               # when tuning, P comes from grid
DEFAULT_RIDGE = 1e-6

# Validation split (temporal)
DEFAULT_VAL_SPLIT = 0.20       # last 20% windows per train file

# Tuning grids
DEFAULT_P_GRID = "4,8,12,16,24"
DEFAULT_RIDGE_GRID = "1e-8,1e-7,1e-6,1e-5,1e-4"

# Output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, "models")

# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------

def parse_grid(s: str, cast=float) -> List[float]:
    s = s.strip()
    if not s:
        return []
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    return vals

def read_numeric_series(csv_path: str, col_index: int = 0) -> np.ndarray:
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError(f"No numeric columns in {csv_path}.")
    if col_index < 0 or col_index >= num_df.shape[1]:
        raise ValueError(f"col_index {col_index} out of range for {csv_path}.")
    arr = num_df.iloc[:, col_index].to_numpy(dtype=float)
    return np.asarray(arr)

def build_windows(series: np.ndarray, n: int, m: int, stride: int) -> np.ndarray:
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

def temporal_train_val_split(X: np.ndarray, val_split: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split windows temporally along columns: first part train, tail part val."""
    K = X.shape[1]
    if K == 0 or val_split <= 0:
        return X, np.zeros((X.shape[0], 0), dtype=X.dtype)
    k_val = int(round(K * val_split))
    k_val = min(max(k_val, 1), K-1)  # ensure non-empty train/val if possible
    X_tr = X[:, :K - k_val]
    X_va = X[:, K - k_val:]
    return X_tr, X_va

def concat_train_windows_with_val(files: List[str], n: int, m: int, stride: int, col_index: int, val_split: float) -> Tuple[np.ndarray, np.ndarray]:
    """Read each training file, build windows, split temporally, and concatenate across files."""
    mats_tr, mats_va = [], []
    for f in files:
        s = read_numeric_series(f, col_index=col_index)
        X = build_windows(s, n, m, stride)
        Xtr, Xva = temporal_train_val_split(X, val_split)
        if Xtr.shape[1] > 0:
            mats_tr.append(Xtr)
        if Xva.shape[1] > 0:
            mats_va.append(Xva)
    X_tr = np.concatenate(mats_tr, axis=1) if mats_tr else np.zeros((n+m, 0), dtype=float)
    X_va = np.concatenate(mats_va, axis=1) if mats_va else np.zeros((n+m, 0), dtype=float)
    return X_tr, X_va

def center_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=1, keepdims=True) if X.shape[1] > 0 else np.zeros((X.shape[0], 1))
    return X - mu, mu.squeeze()

def svd_from_train(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Economy SVD of centered training matrix scaled by sqrt(M)."""
    M = max(Xc.shape[1], 1)
    Xs = Xc / math.sqrt(M)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    eigvals = S**2
    return U, eigvals

def minor_indices_from_P(eigvals: np.ndarray, P: int) -> np.ndarray:
    order = np.argsort(eigvals)  # ascending
    P = int(P)
    P = max(1, min(P, eigvals.shape[0]-1))
    return order[:P]

def minor_indices_from_cut(eigvals: np.ndarray, cutoff: float) -> np.ndarray:
    order = np.argsort(eigvals)
    total = eigvals.sum()
    if total <= 0:
        return order[:1]
    csum = 0.0
    chosen = []
    for j in order:
        csum += eigvals[j]
        chosen.append(j)
        if (csum / total) >= cutoff:
            break
    if not chosen:
        chosen = [order[0]]
    return np.array(chosen, dtype=int)

def compute_W_from_B(B: np.ndarray, n: int, m: int, ridge: float) -> np.ndarray:
    """B: (P x (n+m)); W = -(B2^T B2 + λI)^{-1} B2^T B1  ∈ R^{m×n}."""
    P, N = B.shape
    assert N == n + m, "B must be (P, n+m)"
    B1 = B[:, :n]
    B2 = B[:, n:]
    BtB   = B2.T @ B2        # (m×m)
    BtB1  = B2.T @ B1        # (m×n)
    BtB  += ridge * np.eye(BtB.shape[0])
    W = -np.linalg.solve(BtB, BtB1)
    return W

def predict_block(W: np.ndarray, X1: np.ndarray, mu_full: np.ndarray, n: int, m: int, centered: bool) -> np.ndarray:
    """X1: (n×K) → X2_hat: (m×K)"""
    if centered:
        mu1 = mu_full[:n].reshape(n, 1)
        mu2 = mu_full[n:].reshape(m, 1)
        X1c = X1 - mu1
        X2c_hat = W @ X1c
        return X2c_hat + mu2
    else:
        return W @ X1

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {"mae": mae, "rmse": rmse}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MCA linear predictor with P, λ tuning")
    ap.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES, help="Training CSV files")
    ap.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE, help="Test CSV file")
    ap.add_argument("--col-index", type=int, default=0, help="Numeric column index to use")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Past window length")
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="Future window length")
    ap.add_argument("--stride-train", type=int, default=DEFAULT_STRIDE_TRAIN, help="Training stride")
    ap.add_argument("--stride-test", type=int, default=DEFAULT_STRIDE_TEST, help="Test stride")
    ap.add_argument("--center", action="store_true", default=DEFAULT_CENTER, help="Enable row-centering")
    ap.add_argument("--no-center", dest="center", action="store_false", help="Disable row-centering")
    ap.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Temporal validation split (0..1)")
    ap.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse"], help="Metric to tune on")
    # Tuning grids
    ap.add_argument("--P-grid", type=str, default=DEFAULT_P_GRID, help="Comma list, e.g., '4,8,12,16,24'")
    ap.add_argument("--ridge-grid", type=str, default=DEFAULT_RIDGE_GRID, help="Comma list, e.g., '1e-8,1e-7,1e-6'")
    # Optional: use cutoff to generate a single P from energy (not typical for tuning, but provided)
    ap.add_argument("--use-cutoff", action="store_true", help="Use --energy-cutoff to derive P instead of P-grid")
    ap.add_argument("--energy-cutoff", type=float, default=DEFAULT_ENERGY_CUTOFF, help="Minor energy fraction cutoff")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    ap.add_argument("--model-name", type=str, default=None, help="Custom model folder name")
    args = ap.parse_args()

    n, m = int(args.n), int(args.m)
    N = n + m

    print("=== MCA Tuning ===")
    print(f"Train files: {args.train_files}")
    print(f"Test file:   {args.test_file}")
    print(f"n={n}, m={m}, stride_train={args.stride_train}, stride_test={args.stride_test}")
    print(f"Center={args.center}, val_split={args.val_split}, metric={args.metric}")
    print(f"P-grid={args.P_grid}, ridge-grid={args.ridge_grid}, use-cutoff={args.use_cutoff}, cutoff={args.energy_cutoff}")

    t0 = time.time()

    # -------------------- Build train/val windows ----------------------------
    X_tr_all, X_va_all = concat_train_windows_with_val(
        args.train_files, n, m, args.stride_train, args.col_index, args.val_split
    )
    Ktr, Kva = X_tr_all.shape[1], X_va_all.shape[1]
    if Ktr == 0:
        raise RuntimeError("No training windows. Check files/lengths/stride.")
    if Kva == 0:
        raise RuntimeError("No validation windows. Increase val_split or check files.")

    print(f"Train windows: {Ktr} | Val windows: {Kva}")

    # -------------------- Center on TRAIN only -------------------------------
    if args.center:
        Xc_tr, mu = center_rows(X_tr_all)  # mu over train
    else:
        Xc_tr = X_tr_all.copy()
        mu = np.zeros((N,), dtype=float)

    # -------------------- SVD on TRAIN --------------------------------------
    print("Computing SVD on train...")
    U, eigvals = svd_from_train(Xc_tr)
    print("SVD done.")

    # Pre-split to X1/X2 for val
    X1_va = X_va_all[:n, :]
    Y_va_true = X_va_all[n:, :]

    # Candidate grids
    ridges = parse_grid(args.ridge_grid, cast=float)
    if args.use_cutoff:
        # derive a single P from cutoff
        idx_minor = minor_indices_from_cut(eigvals, args.energy_cutoff)
        P_grid = [int(idx_minor.shape[0])]
        idx_map = {P_grid[0]: idx_minor}
    else:
        P_grid = [int(p) for p in parse_grid(args.P_grid, cast=int)]
        idx_map = {P: minor_indices_from_P(eigvals, P) for P in P_grid}

    # -------------------- Grid search ---------------------------------------
    results = []  # list of dicts
    best = {"score": float("inf"), "P": None, "ridge": None, "W": None, "idx_minor": None}

    for P in P_grid:
        idx_minor = idx_map[P]
        B = U[:, idx_minor].T  # (P × N)
        for ridge in ridges:
            # Compute W
            W = compute_W_from_B(B, n=n, m=m, ridge=float(ridge))
            # Predict on val
            Y_va_pred = predict_block(W, X1_va, mu_full=mu, n=n, m=m, centered=args.center)
            mets = compute_metrics(Y_va_true, Y_va_pred)
            score = mets[args.metric]
            results.append({"P": P, "ridge": float(ridge), **mets})
            if score < best["score"]:
                best.update({"score": score, "P": P, "ridge": float(ridge), "W": W.copy(), "idx_minor": idx_minor.copy()})

    # Print grid summary sorted by chosen metric
    print("\nGrid results (sorted by {}):".format(args.metric))
    results_sorted = sorted(results, key=lambda d: d[args.metric])
    for r in results_sorted:
        print(f"P={r['P']:>3d}  ridge={r['ridge']:<10.1e}  val_mae={r['mae']:.6f}  val_rmse={r['rmse']:.6f}")

    print("\nBest:")
    print(f"P={best['P']}, ridge={best['ridge']:.1e}, val_{args.metric}={best['score']:.6f}")

    # -------------------- Save best model -----------------------------------
    model_name = args.model_name
    if model_name is None:
        model_name = f"MCA_TUNED_n{n}_m{m}_stride{args.stride_train}_P{best['P']}_ridge{best['ridge']}"
    model_dir = os.path.join(args.outdir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(model_dir, "W.npy"), best["W"])
    np.save(os.path.join(model_dir, "mu.npy"), mu)
    np.save(os.path.join(model_dir, "idx_minor.npy"), best["idx_minor"])

    config = {
        "train_files": args.train_files,
        "test_file": args.test_file,
        "col_index": args.col_index,
        "n": n, "m": m,
        "stride_train": args.stride_train,
        "stride_test": args.stride_test,
        "center": bool(args.center),
        "val_split": float(args.val_split),
        "metric": args.metric,
        "P_grid": P_grid,
        "ridge_grid": ridges,
        "use_cutoff": bool(args.use_cutoff),
        "energy_cutoff": float(args.energy_cutoff),
        "P_used": int(best["P"]),
        "ridge_used": float(best["ridge"]),
        "tuning_results": results_sorted[:50],  # top 50 for quick view
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved tuned model to: {model_dir}")

    # -------------------- Final test evaluation -----------------------------
    print("\n=== Evaluation on test set with tuned (P, λ) ===")
    series_test = read_numeric_series(args.test_file, col_index=args.col_index)
    X_test = build_windows(series_test, n, m, args.stride_test)
    K_test = X_test.shape[1]
    if K_test == 0:
        raise RuntimeError("No test windows. Check test length/stride.")

    X1_te = X_test[:n, :]
    Y_te_true = X_test[n:, :]
    Y_te_pred = predict_block(best["W"], X1_te, mu_full=mu, n=n, m=m, centered=args.center)
    mets_test = compute_metrics(Y_te_true, Y_te_pred)

    print(f"Test windows: {K_test}")
    print(f"TEST  MAE={mets_test['mae']:.6f}  RMSE={mets_test['rmse']:.6f}")

    np.savez(os.path.join(model_dir, "test_predictions.npz"),
             Y_true=Y_te_true, Y_pred=Y_te_pred, X1=X1_te)

    dt = time.time() - t0
    print(f"\nDone in {dt:.3f} s.")
    print("Artifacts:")
    print(f" - {os.path.join(model_dir, 'W.npy')}")
    print(f" - {os.path.join(model_dir, 'mu.npy')}")
    print(f" - {os.path.join(model_dir, 'idx_minor.npy')}")
    print(f" - {os.path.join(model_dir, 'config.json')}")
    print(f" - {os.path.join(model_dir, 'test_predictions.npz')}")

if __name__ == "__main__":
    main()
