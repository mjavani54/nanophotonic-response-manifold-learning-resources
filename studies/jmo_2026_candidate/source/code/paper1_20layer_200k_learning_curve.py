#!/usr/bin/env python3
"""
Paper 1 — 20-Layer 200k Learning-Curve Experiment

Purpose
-------
Estimate the training-sample requirement for the 20-layer nanophotonic system
using a fixed neural-network architecture and fixed optimization protocol.

The script varies ONLY the number of training samples:
    10k, 20k, 30k, 50k, 75k, 100k, 150k, 200k (subject to holdout requirements)

Key safeguards
--------------
1. One fixed test set is held out before any training-size subsampling.
2. One fixed validation set is held out before any training-size subsampling.
3. Training subsets are nested: larger training sets contain the smaller ones
   for a given seed.
4. Multiple random seeds are used.
5. The network architecture and optimizer are held fixed across all training sizes.
6. Outputs include mean/std metrics and threshold-crossing estimates.

Expected CSV orientation
------------------------
DS: 20 x N
RS: 200 x N
where columns are samples.

Typical usage
-------------
python paper1_20layer_200k_learning_curve.py \
    --zip "Datasets20.zip" \
    --out paper1_20layer_200k_learning_curve

If your zip contains filenames that do not include "20" and "DS"/"RS",
use --ds-name and --rs-name with exact filenames.
"""

import argparse
import json
import math
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Reproducibility ----------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- File discovery ------------------------------ #

def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    extract_root = out_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".done"
    if not marker.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text("ok\n")
    return extract_root


def discover_csvs(root: Path, ds_name=None, rs_name=None):
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {root}")

    if ds_name:
        ds_matches = [p for p in csvs if p.name == ds_name]
    else:
        ds_matches = [
            p for p in csvs
            if "ds" in p.stem.lower() and "20" in p.stem.lower()
        ]
        if not ds_matches:
            ds_matches = [p for p in csvs if "ds" in p.stem.lower()]

    if rs_name:
        rs_matches = [p for p in csvs if p.name == rs_name]
    else:
        rs_matches = [
            p for p in csvs
            if "rs" in p.stem.lower() and "20" in p.stem.lower()
        ]
        if not rs_matches:
            rs_matches = [p for p in csvs if "rs" in p.stem.lower()]

    if len(ds_matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify DS file.\n"
            f"Candidates: {[str(p) for p in ds_matches]}\n"
            "Use --ds-name with the exact filename."
        )
    if len(rs_matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify RS file.\n"
            f"Candidates: {[str(p) for p in rs_matches]}\n"
            "Use --rs-name with the exact filename."
        )
    return ds_matches[0], rs_matches[0]


# ------------------------------ Data loading ------------------------------- #

def load_matrix(path: Path) -> np.ndarray:
    # Fast CSV loading with pandas.
    arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    return arr


def qc_and_orient(ds: np.ndarray, rs: np.ndarray):
    """
    Orient to samples x features:
        X: N x 20
        Y: N x 200
    """
    original_shapes = {"ds_raw": list(ds.shape), "rs_raw": list(rs.shape)}

    # DS should have one dimension equal to 20.
    if ds.shape[0] == 20:
        X = ds.T
    elif ds.shape[1] == 20:
        X = ds
    else:
        raise ValueError(f"DS must have 20 design variables. Raw shape: {ds.shape}")

    # RS should have one dimension equal to 200.
    if rs.shape[0] == 200:
        Y = rs.T
    elif rs.shape[1] == 200:
        Y = rs
    else:
        raise ValueError(f"RS must have 200 response points. Raw shape: {rs.shape}")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample-count mismatch after orientation: X={X.shape}, Y={Y.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError("DS contains NaN or Inf values.")
    if not np.isfinite(Y).all():
        raise ValueError("RS contains NaN or Inf values.")

    # Exact duplicate sample checks.
    dup_x = pd.DataFrame(X).duplicated().sum()
    dup_y = pd.DataFrame(Y).duplicated().sum()

    qc = {
        **original_shapes,
        "X_oriented": list(X.shape),
        "Y_oriented": list(Y.shape),
        "n_samples": int(X.shape[0]),
        "x_min": float(np.min(X)),
        "x_max": float(np.max(X)),
        "y_min": float(np.min(Y)),
        "y_max": float(np.max(Y)),
        "duplicate_design_samples": int(dup_x),
        "duplicate_response_samples": int(dup_y),
    }
    return X.astype(np.float32), Y.astype(np.float32), qc


# ------------------------------ Standardizer ------------------------------- #

class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x):
        return (x * self.std + self.mean).astype(np.float32)


# ------------------------------- Neural net -------------------------------- #

class MLP(nn.Module):
    def __init__(self, in_dim=20, out_dim=200, width=896, depth=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------- Metrics ----------------------------------- #

def metrics(y_true, y_pred):
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    resid = yt - yp

    mse = np.mean(resid ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(resid))

    sse = np.sum(resid ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)
    r2_global = 1.0 - sse / sst if sst > 0 else np.nan

    # Per-spectrum R2, then mean. This is useful but can be unstable
    # if an individual spectrum has tiny variance.
    denom = np.sum((yt - yt.mean(axis=1, keepdims=True)) ** 2, axis=1)
    numer = np.sum(resid ** 2, axis=1)
    valid = denom > 1e-12
    r2_per_sample = np.full(len(yt), np.nan, dtype=np.float64)
    r2_per_sample[valid] = 1.0 - numer[valid] / denom[valid]

    return {
        "r2_global": float(r2_global),
        "r2_per_sample_mean": float(np.nanmean(r2_per_sample)),
        "rmse": float(rmse),
        "mae": float(mae),
    }


# ------------------------------- Training ---------------------------------- #

def train_one(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    seed, device,
    width=896, depth=4,
    batch_size=1024,
    lr=1e-3,
    max_epochs=1200,
    patience=100,
    min_delta=1e-7,
    weight_decay=0.0,
    num_workers=0,
):
    set_seed(seed)

    # Fit scaling only on the current training subset.
    sx = Standardizer().fit(X_train)
    sy = Standardizer().fit(Y_train)

    xtr = sx.transform(X_train)
    ytr = sy.transform(Y_train)
    xva = sx.transform(X_val)
    yva = sy.transform(Y_val)
    xte = sx.transform(X_test)

    train_ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(ytr)
    )
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = MLP(width=width, depth=depth).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Plateau scheduler; same protocol for every training size.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        min_lr=1e-6,
    )

    loss_fn = nn.MSELoss()

    Xv = torch.from_numpy(xva).to(device)
    Yv = torch.from_numpy(yva).to(device)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            running += loss.item() * bs
            seen += bs

        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, Yv).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append((epoch, train_loss, val_loss, current_lr))

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    def predict_np(x_raw, chunk=8192):
        x_std = sx.transform(x_raw)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_std), chunk):
                xb = torch.from_numpy(x_std[i:i+chunk]).to(device)
                preds.append(model(xb).cpu().numpy())
        pred_std = np.concatenate(preds, axis=0)
        return sy.inverse_transform(pred_std)

    pred_train = predict_np(X_train)
    pred_val = predict_np(X_val)
    pred_test = predict_np(X_test)

    mtr = metrics(Y_train, pred_train)
    mva = metrics(Y_val, pred_val)
    mte = metrics(Y_test, pred_test)

    return {
        "seed": seed,
        "n_parameters": n_params,
        "epochs_trained": int(history[-1][0]),
        "best_val_standardized_mse": float(best_val),
        "final_lr": float(history[-1][3]),
        "train_r2": mtr["r2_global"],
        "val_r2": mva["r2_global"],
        "test_r2": mte["r2_global"],
        "test_r2_per_sample_mean": mte["r2_per_sample_mean"],
        "test_rmse": mte["rmse"],
        "test_mae": mte["mae"],
        "history": history,
    }


# ---------------------------- Main experiment ------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=None,
                    help="ZIP containing 20-layer DS/RS CSV files.")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Folder containing DS/RS CSV files.")
    ap.add_argument("--ds-name", type=str, default=None)
    ap.add_argument("--rs-name", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default="paper1_20layer_200k_learning_curve")

    ap.add_argument("--train-sizes", type=int, nargs="+",
                    default=[10000, 20000, 30000, 50000, 75000, 100000, 150000, 180000],
                    help="Training sizes. Largest must leave room for val/test.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])

    ap.add_argument("--test-size", type=int, default=10000)
    ap.add_argument("--val-size", type=int, default=10000)

    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.zip:
        root = extract_zip(Path(args.zip), out)
    elif args.data_root:
        root = Path(args.data_root)
    else:
        raise ValueError("Provide either --zip or --data-root.")

    ds_path, rs_path = discover_csvs(
        root,
        ds_name=args.ds_name,
        rs_name=args.rs_name
    )
    print(f"DS: {ds_path}")
    print(f"RS: {rs_path}")

    ds = load_matrix(ds_path)
    rs = load_matrix(rs_path)
    X, Y, qc = qc_and_orient(ds, rs)

    n_total = len(X)
    if args.test_size + args.val_size >= n_total:
        raise ValueError("Validation + test sizes consume the entire dataset.")

    max_train_available = n_total - args.test_size - args.val_size
    requested_sizes = sorted(set(args.train_sizes))
    train_sizes = [n for n in requested_sizes if n <= max_train_available]

    if not train_sizes:
        raise ValueError(
            f"No requested training size fits. Maximum available is {max_train_available}."
        )

    if requested_sizes[-1] > max_train_available:
        print(
            f"NOTE: requested size {requested_sizes[-1]} exceeds maximum "
            f"{max_train_available} after holdouts and will be skipped."
        )

    qc.update({
        "ds_file": str(ds_path),
        "rs_file": str(rs_path),
        "device": str(device),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "max_train_available": max_train_available,
        "train_sizes_used": train_sizes,
        "width": args.width,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "seeds": args.seeds,
    })
    (out / "learning_curve_qc.json").write_text(json.dumps(qc, indent=2))

    all_rows = []

    # IMPORTANT:
    # For each seed, use one permutation:
    # first test, next val, remainder training pool.
    # Every larger train subset is nested within the previous one.
    for seed in args.seeds:
        print(f"\n===== Seed {seed} =====")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)

        test_idx = perm[:args.test_size]
        val_idx = perm[args.test_size:args.test_size + args.val_size]
        pool_idx = perm[args.test_size + args.val_size:]

        X_test, Y_test = X[test_idx], Y[test_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        for n_train in train_sizes:
            print(f"Seed {seed} | n_train={n_train:,}")
            tr_idx = pool_idx[:n_train]
            X_train, Y_train = X[tr_idx], Y[tr_idx]

            res = train_one(
                X_train, Y_train,
                X_val, Y_val,
                X_test, Y_test,
                seed=seed,
                device=device,
                width=args.width,
                depth=args.depth,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
            )

            hist = pd.DataFrame(
                res.pop("history"),
                columns=["epoch", "train_loss", "val_loss", "lr"]
            )
            hist.to_csv(
                out / f"history_seed{seed}_n{n_train}.csv",
                index=False
            )

            row = {
                "layers": 20,
                "n_total_dataset": n_total,
                "n_train": n_train,
                "n_val": args.val_size,
                "n_test": args.test_size,
                "width": args.width,
                "depth": args.depth,
                **res,
            }
            all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(
                out / "learning_curve_results.csv",
                index=False
            )

    results = pd.DataFrame(all_rows)
    results.to_csv(out / "learning_curve_results.csv", index=False)

    agg = results.groupby("n_train").agg(
        n_seeds=("seed", "count"),
        n_parameters=("n_parameters", "first"),
        epochs_mean=("epochs_trained", "mean"),
        epochs_std=("epochs_trained", "std"),
        train_r2_mean=("train_r2", "mean"),
        train_r2_std=("train_r2", "std"),
        val_r2_mean=("val_r2", "mean"),
        val_r2_std=("val_r2", "std"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
    ).reset_index()

    agg["generalization_gap_mean"] = (
        agg["train_r2_mean"] - agg["test_r2_mean"]
    )
    agg.to_csv(out / "learning_curve_summary.csv", index=False)

    # Threshold decisions
    threshold_rows = []
    for threshold in [0.99, 0.995, 0.999]:
        passing = agg.loc[agg["test_r2_mean"] >= threshold]
        if len(passing):
            first = passing.iloc[0]
            threshold_rows.append({
                "threshold_r2": threshold,
                "resolved": True,
                "minimum_training_samples": int(first["n_train"]),
                "mean_test_r2_at_crossing": float(first["test_r2_mean"]),
                "std_test_r2_at_crossing": float(first["test_r2_std"]),
            })
        else:
            threshold_rows.append({
                "threshold_r2": threshold,
                "resolved": False,
                "minimum_training_samples": np.nan,
                "mean_test_r2_at_crossing": np.nan,
                "std_test_r2_at_crossing": np.nan,
            })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out / "learning_curve_thresholds.csv", index=False)

    # Simple scaling fits for extrapolation/description.
    # Fit against log10(N) only when enough points exist.
    fit_info = {}
    if len(agg) >= 3:
        x = np.log10(agg["n_train"].to_numpy(dtype=float))
        y = agg["test_r2_mean"].to_numpy(dtype=float)
        coeff = np.polyfit(x, y, deg=1)
        yhat = np.polyval(coeff, x)
        sse = np.sum((y - yhat) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r2_fit = 1 - sse / sst if sst > 0 else np.nan
        fit_info = {
            "model": "test_r2_mean = a*log10(n_train) + b",
            "a": float(coeff[0]),
            "b": float(coeff[1]),
            "r2": float(r2_fit),
        }
    (out / "learning_curve_fit.json").write_text(json.dumps(fit_info, indent=2))

    # Figure 1: R2 vs training size
    fig = plt.figure(figsize=(8, 5.5))
    plt.errorbar(
        agg["n_train"],
        agg["test_r2_mean"],
        yerr=agg["test_r2_std"].fillna(0),
        marker="o",
        capsize=4
    )
    plt.axhline(0.99, linestyle="--", linewidth=1.2)
    plt.xlabel("Number of training samples")
    plt.ylabel(r"Test $R^2$")
    plt.title("20-Layer Learning Curve")
    plt.xscale("log")
    plt.tight_layout()
    fig.savefig(out / "learning_curve_test_r2.png", dpi=300)
    plt.close(fig)

    # Figure 2: train/test comparison
    fig = plt.figure(figsize=(8, 5.5))
    plt.plot(
        agg["n_train"], agg["train_r2_mean"],
        marker="o", label="Train"
    )
    plt.plot(
        agg["n_train"], agg["test_r2_mean"],
        marker="o", label="Test"
    )
    plt.axhline(0.99, linestyle="--", linewidth=1.2)
    plt.xlabel("Number of training samples")
    plt.ylabel(r"$R^2$")
    plt.title("20-Layer Train vs Test Performance")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "learning_curve_train_vs_test.png", dpi=300)
    plt.close(fig)

    # Figure 3: generalization gap
    fig = plt.figure(figsize=(8, 5.5))
    plt.plot(
        agg["n_train"],
        agg["generalization_gap_mean"],
        marker="o"
    )
    plt.xlabel("Number of training samples")
    plt.ylabel(r"Train $R^2$ - Test $R^2$")
    plt.title("20-Layer Generalization Gap")
    plt.xscale("log")
    plt.tight_layout()
    fig.savefig(out / "learning_curve_generalization_gap.png", dpi=300)
    plt.close(fig)

    print("\n===== SUMMARY =====")
    print(agg.to_string(index=False))
    print("\n===== THRESHOLDS =====")
    print(thresholds.to_string(index=False))
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
