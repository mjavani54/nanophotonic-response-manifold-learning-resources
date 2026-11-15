#!/usr/bin/env python3
"""
Paper 1 — Low-Sample Refinement for 8–14 Layer Structures

Purpose
-------
Resolve the low-end censoring in the controlled cross-structure learning curves.

Structures:
    8, 10, 12, 14 layers

Training sizes:
    1k, 2k, 3k, 4k, 5k

Fixed architectures (same as controlled experiment):
    8L:  width 64, depth 4
    10L: width 64, depth 4
    12L: width 128, depth 4
    14L: width 128, depth 4

Same optimizer/training protocol:
    Adam, lr=1e-3
    ReduceLROnPlateau
    max 1200 epochs
    patience 100
    3 seeds

Typical usage
-------------
python paper1_low_sample_refinement.py \
    --zip-small "Datasets(2).zip" \
    --out paper1_low_sample_refinement
"""

import argparse
import json
import math
import random
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_zip_fresh(zip_path: Path, out_root: Path) -> Path:
    target = out_root / "extracted" / "small"
    target.mkdir(parents=True, exist_ok=True)

    csvs = list(target.rglob("*.csv"))
    if not csvs:
        if any(target.iterdir()):
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)

    csvs = list(target.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found after extracting {zip_path} into {target}"
        )
    return target


def find_layer_file(root: Path, layer: int, kind: str) -> Path:
    kind_low = kind.lower()
    csvs = list(root.rglob("*.csv"))

    pats = [
        re.compile(rf"(?<!\d){layer}[\s_-]*layer(?!\d)", re.I),
        re.compile(rf"layer[\s_-]*{layer}(?!\d)", re.I),
    ]

    cands = []
    for p in csvs:
        stem = p.stem
        stem_low = stem.lower()
        kind_ok = (
            re.search(rf"(^|[\s_-]){re.escape(kind_low)}($|[\s_-])",
                      stem_low, re.I) is not None
            or stem_low.endswith(kind_low)
        )
        layer_ok = any(pat.search(stem) for pat in pats)
        if kind_ok and layer_ok:
            cands.append(p)

    if len(cands) != 1:
        raise RuntimeError(
            f"Could not uniquely identify {kind} for {layer} layers.\n"
            f"Candidates: {[str(p) for p in cands]}"
        )
    return cands[0]


def load_csv(path: Path):
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def orient_pair(ds, rs, layer):
    if ds.shape[0] == layer:
        X = ds.T
    elif ds.shape[1] == layer:
        X = ds
    else:
        raise ValueError(f"{layer}L DS shape invalid: {ds.shape}")

    if rs.shape[0] == 200:
        Y = rs.T
    elif rs.shape[1] == 200:
        Y = rs
    else:
        raise ValueError(f"{layer}L RS shape invalid: {rs.shape}")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{layer}L sample mismatch: X={X.shape}, Y={Y.shape}")

    if not np.isfinite(X).all():
        raise ValueError(f"{layer}L DS contains NaN/Inf")
    if not np.isfinite(Y).all():
        raise ValueError(f"{layer}L RS contains NaN/Inf")

    return X.astype(np.float32), Y.astype(np.float32)


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


class MLP(nn.Module):
    def __init__(self, in_dim, width, depth=4, out_dim=200):
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


def regression_metrics(y_true, y_pred):
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    resid = yt - yp

    sse = np.sum(resid ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    mse = np.mean(resid ** 2)
    return {
        "r2": float(r2),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(resid))),
    }


def train_one(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    layer, width, seed, device,
    depth=4,
    batch_size=512,
    lr=1e-3,
    max_epochs=1200,
    patience=100,
    min_delta=1e-7,
):
    set_seed(seed)

    sx = Standardizer().fit(X_train)
    sy = Standardizer().fit(Y_train)

    xtr = sx.transform(X_train)
    ytr = sy.transform(Y_train)
    xva = sx.transform(X_val)
    yva = sy.transform(Y_val)

    train_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = MLP(layer, width, depth=depth).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            seen += bs

        train_loss = total / max(seen, 1)

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
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid best state found.")

    model.load_state_dict(best_state)
    model.eval()

    def predict(raw_x, chunk=8192):
        xs = sx.transform(raw_x)
        outs = []
        with torch.no_grad():
            for i in range(0, len(xs), chunk):
                xb = torch.from_numpy(xs[i:i+chunk]).to(device)
                outs.append(model(xb).cpu().numpy())
        pred_std = np.concatenate(outs, axis=0)
        return sy.inverse_transform(pred_std)

    pred_train = predict(X_train)
    pred_val = predict(X_val)
    pred_test = predict(X_test)

    mtr = regression_metrics(Y_train, pred_train)
    mva = regression_metrics(Y_val, pred_val)
    mte = regression_metrics(Y_test, pred_test)

    return {
        "n_parameters": n_params,
        "epochs_trained": int(history[-1][0]),
        "best_val_standardized_mse": float(best_val),
        "final_lr": float(history[-1][3]),
        "train_r2": mtr["r2"],
        "val_r2": mva["r2"],
        "test_r2": mte["r2"],
        "test_rmse": mte["rmse"],
        "test_mae": mte["mae"],
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-small", required=True)
    ap.add_argument("--out", default="paper1_low_sample_refinement")
    ap.add_argument("--train-sizes", nargs="+", type=int,
                    default=[1000,2000,3000,4000,5000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = extract_zip_fresh(Path(args.zip_small), out)

    widths = {
        8: 64,
        10: 64,
        12: 128,
        14: 128,
    }

    all_rows = []
    qc_rows = []

    for layer in [8,10,12,14]:
        print(f"\n================ {layer} layers ================")

        ds_path = find_layer_file(root, layer, "DS")
        rs_path = find_layer_file(root, layer, "RS")
        print(f"DS: {ds_path}")
        print(f"RS: {rs_path}")

        ds = load_csv(ds_path)
        rs = load_csv(rs_path)
        X, Y = orient_pair(ds, rs, layer)

        qc_rows.append({
            "layers": layer,
            "n_total": len(X),
            "x_dim": X.shape[1],
            "y_dim": Y.shape[1],
            "x_min": float(X.min()),
            "x_max": float(X.max()),
            "y_min": float(Y.min()),
            "y_max": float(Y.max()),
            "width": widths[layer],
            "depth": args.depth,
        })

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")

            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X))

            test_idx = perm[:args.test_size]
            val_idx = perm[args.test_size:args.test_size + args.val_size]
            pool_idx = perm[args.test_size + args.val_size:]

            X_test, Y_test = X[test_idx], Y[test_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            for n_train in args.train_sizes:
                print(
                    f"{layer}L | seed={seed} | n_train={n_train:,} | "
                    f"width={widths[layer]}"
                )

                idx = pool_idx[:n_train]
                X_train, Y_train = X[idx], Y[idx]

                res = train_one(
                    X_train, Y_train,
                    X_val, Y_val,
                    X_test, Y_test,
                    layer=layer,
                    width=widths[layer],
                    seed=seed,
                    device=device,
                    depth=args.depth,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                )

                hist = pd.DataFrame(
                    res.pop("history"),
                    columns=["epoch","train_loss","val_loss","lr"]
                )
                hist.to_csv(
                    out / f"history_L{layer}_seed{seed}_n{n_train}.csv",
                    index=False
                )

                row = {
                    "layers": layer,
                    "seed": seed,
                    "n_train": n_train,
                    "n_val": args.val_size,
                    "n_test": args.test_size,
                    "width": widths[layer],
                    "depth": args.depth,
                    **res,
                }
                all_rows.append(row)

                pd.DataFrame(all_rows).to_csv(
                    out / "low_sample_refinement_results.csv",
                    index=False
                )

    pd.DataFrame(qc_rows).to_csv(out / "low_sample_refinement_qc.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(out / "low_sample_refinement_results.csv", index=False)

    summary = results.groupby(["layers","n_train"]).agg(
        n_seeds=("seed","count"),
        width=("width","first"),
        n_parameters=("n_parameters","first"),
        train_r2_mean=("train_r2","mean"),
        train_r2_std=("train_r2","std"),
        val_r2_mean=("val_r2","mean"),
        val_r2_std=("val_r2","std"),
        test_r2_mean=("test_r2","mean"),
        test_r2_std=("test_r2","std"),
        test_rmse_mean=("test_rmse","mean"),
        test_rmse_std=("test_rmse","std"),
        test_mae_mean=("test_mae","mean"),
        test_mae_std=("test_mae","std"),
    ).reset_index()

    summary["generalization_gap_mean"] = (
        summary["train_r2_mean"] - summary["test_r2_mean"]
    )
    summary.to_csv(out / "low_sample_refinement_summary.csv", index=False)

    threshold_rows = []
    for layer in [8,10,12,14]:
        sub = summary[summary.layers == layer].sort_values("n_train")
        for threshold in [0.99,0.995,0.999]:
            passing = sub[sub.test_r2_mean >= threshold]
            if len(passing):
                first = passing.iloc[0]
                threshold_rows.append({
                    "layers": layer,
                    "threshold_r2": threshold,
                    "resolved": True,
                    "minimum_training_samples": int(first.n_train),
                    "test_r2_mean_at_crossing": float(first.test_r2_mean),
                    "test_r2_std_at_crossing": float(first.test_r2_std),
                })
            else:
                threshold_rows.append({
                    "layers": layer,
                    "threshold_r2": threshold,
                    "resolved": False,
                    "minimum_training_samples": np.nan,
                    "test_r2_mean_at_crossing": np.nan,
                    "test_r2_std_at_crossing": np.nan,
                })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out / "low_sample_refinement_thresholds.csv", index=False)

    # Figure
    fig = plt.figure(figsize=(8.5,6))
    for layer in [8,10,12,14]:
        sub = summary[summary.layers == layer]
        plt.errorbar(
            sub.n_train,
            sub.test_r2_mean,
            yerr=sub.test_r2_std.fillna(0),
            marker="o",
            capsize=3,
            label=f"{layer} layers",
        )
    plt.axhline(0.99, linestyle="--", linewidth=1.2)
    plt.xlabel("Number of training samples")
    plt.ylabel(r"Test $R^2$")
    plt.title("Low-Sample Learning-Curve Refinement")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "fig_low_sample_refinement.png", dpi=300)
    plt.close(fig)

    metadata = {
        "device": str(device),
        "train_sizes": args.train_sizes,
        "seeds": args.seeds,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "widths": widths,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n================ R2 >= 0.99 THRESHOLDS ================")
    print(
        thresholds[thresholds.threshold_r2 == 0.99]
        .to_string(index=False)
    )
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
