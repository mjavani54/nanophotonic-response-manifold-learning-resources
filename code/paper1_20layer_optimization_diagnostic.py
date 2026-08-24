#!/usr/bin/env python3
"""
Paper 1 — 20-layer optimization diagnostic
===========================================
Purpose
-------
Test whether the 20-layer forward surrogate's apparent R^2 plateau near 0.987
is caused by an inadequate optimization/training protocol rather than the
available 50,000-sample dataset.

The architecture is intentionally held fixed at width=896, depth=4 (~2.61M
parameters), because this was the best-performing region in the preceding
capacity diagnostic. Only optimizer/training settings are changed.

Decision logic
--------------
* If any training configuration reaches mean test R^2 >= 0.99 across seeds,
  the 50k dataset is sufficient and the previous failure was optimization-limited.
* If all well-tuned configurations remain below 0.99 and cluster near the same
  plateau, additional 20-layer data is scientifically justified.

Example
-------
python paper1_20layer_optimization_diagnostic.py \
    --zip "Datasets(2).zip" \
    --out paper1_20layer_optimization
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit("PyTorch is required. Install it in the active environment first.") from e

LAYER = 20
EXPECTED_SAMPLES = 50_000
EXPECTED_RESPONSE_DIM = 200
WIDTH = 896
DEPTH = 4
DEFAULT_SEEDS = [11, 29, 47]
TARGET_R2 = 0.99

# Structured optimization sweep. Architecture and data split remain fixed.
# The baseline is deliberately retained for comparability with the earlier run.
CONFIGS = [
    {
        "name": "adam_1e-3_long",
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "none",
    },
    {
        "name": "adam_5e-4_long",
        "optimizer": "adam",
        "lr": 5e-4,
        "weight_decay": 0.0,
        "scheduler": "none",
    },
    {
        "name": "adam_1e-3_plateau",
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "plateau",
    },
    {
        "name": "adamw_1e-3_wd1e-5_plateau",
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "scheduler": "plateau",
    },
    {
        "name": "adamw_5e-4_wd1e-5_plateau",
        "optimizer": "adamw",
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "scheduler": "plateau",
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_matrix(path: Path) -> np.ndarray:
    # CSV layout: features x samples. Network layout: samples x features.
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32).T


def locate_dataset_root(zip_path: Path | None, data_dir: Path | None, out: Path) -> Path:
    if data_dir is not None:
        return data_dir
    if zip_path is None:
        raise ValueError("Provide --zip or --data-dir")

    extract_dir = out / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    target = f"SiO2_TiO2_{LAYER}_layer_50K_200_points_RS.csv"
    candidates = list(extract_dir.rglob(target))
    if not candidates:
        raise FileNotFoundError(f"Could not locate {target} after extraction.")
    return candidates[0].parent


def dataset_paths(root: Path) -> tuple[Path, Path]:
    ds = root / f"SiO2_TiO2_{LAYER}_layer_50K_200_points_DS.csv"
    rs = root / f"SiO2_TiO2_{LAYER}_layer_50K_200_points_RS.csv"
    if not ds.exists() or not rs.exists():
        raise FileNotFoundError("Missing 20-layer DS/RS files")
    return ds, rs


def fixed_split(n: int, seed: int = 20260816):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.20, random_state=seed, shuffle=True)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, shuffle=True)
    return tr, va, te


def normalize_design(Xtr, Xv, Xte):
    lo = Xtr.min(axis=0, keepdims=True)
    hi = Xtr.max(axis=0, keepdims=True)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (Xtr - lo) / span, (Xv - lo) / span, (Xte - lo) / span


def response_metrics(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    nmse = mse / var if var > 0 else np.nan
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return mse, mae, nmse, r2


class ForwardNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        dims = [input_dim] + [width] * depth + [output_dim]
        modules = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            modules.extend([nn.Linear(a, b), nn.ReLU()])
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


def make_scheduler(opt, cfg, scheduler_patience: int, scheduler_factor: float, min_lr: float):
    if cfg["scheduler"] == "none":
        return None
    if cfg["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )
    raise ValueError(f"Unknown scheduler: {cfg['scheduler']}")


def train_model(
    model,
    Xtr,
    Ytr,
    Xv,
    Yv,
    *,
    cfg,
    epochs,
    batch_size,
    patience,
    scheduler_patience,
    scheduler_factor,
    min_lr,
    device,
):
    model.to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr.astype(np.float32)),
            torch.from_numpy(Ytr.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    Yv_t = torch.from_numpy(Yv.astype(np.float32)).to(device)

    opt = make_optimizer(model, cfg)
    scheduler = make_scheduler(opt, cfg, scheduler_patience, scheduler_factor, min_lr)
    loss_fn = nn.MSELoss()

    best = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            seen += len(xb)

        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), Yv_t).item()

        current_lr = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "lr": current_lr,
        })

        if scheduler is not None:
            scheduler.step(val_loss)

        # Relative/absolute tolerance is deliberately tiny; we want to preserve
        # small but real gains near the final plateau.
        if val_loss < best - 1e-9:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best, history


def predict_numpy(model, X, device, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
            out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def save_learning_curve(history, config_name: str, seed: int, figdir: Path):
    h = pd.DataFrame(history)
    h.to_csv(figdir.parent / f"history_{config_name}_seed{seed}.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.semilogy(h["epoch"], h["train_mse"], label="train")
    plt.semilogy(h["epoch"], h["val_mse"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / f"learning_curve_{config_name}_seed{seed}.png", dpi=220)
    plt.close()


def qc(X, Y, out: Path):
    row = {
        "layers": LAYER,
        "X_shape": str(X.shape),
        "Y_shape": str(Y.shape),
        "shape_ok": X.shape == (EXPECTED_SAMPLES, LAYER)
                    and Y.shape == (EXPECTED_SAMPLES, EXPECTED_RESPONSE_DIM),
        "X_nan": int(np.isnan(X).sum()),
        "Y_nan": int(np.isnan(Y).sum()),
        "X_inf": int(np.isinf(X).sum()),
        "Y_inf": int(np.isinf(Y).sum()),
    }
    pd.DataFrame([row]).to_csv(out / "optimization_qc.csv", index=False)
    if not row["shape_ok"] or any(row[k] != 0 for k in ["X_nan", "Y_nan", "X_inf", "Y_inf"]):
        raise RuntimeError("QC failed. Inspect optimization_qc.csv")


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    figdir = args.out / "figures"
    figdir.mkdir(exist_ok=True)

    root = locate_dataset_root(args.zip, args.data_dir, args.out)
    ds_path, rs_path = dataset_paths(root)
    X = read_csv_matrix(ds_path)
    Y = read_csv_matrix(rs_path)
    qc(X, Y, args.out)

    tr, va, te = fixed_split(len(X))
    Xtr, Xv, Xte = X[tr], X[va], X[te]
    Ytr, Yv, Yte = Y[tr], Y[va], Y[te]
    Xtr, Xv, Xte = normalize_design(Xtr, Xv, Xte)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset root: {root}")
    print(f"20-layer split: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")
    print(f"Architecture: width={WIDTH}, depth={DEPTH}")

    probe_model = ForwardNet(LAYER, EXPECTED_RESPONSE_DIM)
    nparams = sum(p.numel() for p in probe_model.parameters())
    print(f"Trainable parameters: {nparams:,}")
    del probe_model

    raw_rows = []
    raw_path = args.out / "optimization_results.csv"

    for cfg in CONFIGS:
        print(f"\nConfiguration: {cfg['name']}")
        for seed in args.seeds:
            print(f"  seed={seed}")
            seed_everything(seed)
            model = ForwardNet(LAYER, EXPECTED_RESPONSE_DIM)

            model, best_epoch, best_val_mse, history = train_model(
                model,
                Xtr,
                Ytr,
                Xv,
                Yv,
                cfg=cfg,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                min_lr=args.min_lr,
                device=device,
            )

            pred_tr = predict_numpy(model, Xtr, device)
            pred_te = predict_numpy(model, Xte, device)
            tr_mse, tr_mae, tr_nmse, tr_r2 = response_metrics(Ytr, pred_tr)
            te_mse, te_mae, te_nmse, te_r2 = response_metrics(Yte, pred_te)

            final_lr = history[-1]["lr"] if history else cfg["lr"]
            row = {
                "layers": LAYER,
                "width": WIDTH,
                "depth": DEPTH,
                "parameters": nparams,
                "config": cfg["name"],
                "optimizer": cfg["optimizer"],
                "initial_lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "scheduler": cfg["scheduler"],
                "seed": seed,
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
                "final_recorded_lr": final_lr,
                "train_mse": tr_mse,
                "train_nmse": tr_nmse,
                "train_r2": tr_r2,
                "test_mse": te_mse,
                "test_mae": te_mae,
                "test_nmse": te_nmse,
                "test_r2": te_r2,
            }
            raw_rows.append(row)
            pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
            save_learning_curve(history, cfg["name"], seed, figdir)
            print(
                f"    best_epoch={best_epoch}, val_mse={best_val_mse:.6g}, "
                f"train_R2={tr_r2:.6f}, test_R2={te_r2:.6f}"
            )

            del model, pred_tr, pred_te
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(raw_rows)
    summary = raw.groupby(
        ["config", "optimizer", "initial_lr", "weight_decay", "scheduler"],
        as_index=False,
    ).agg(
        test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"),
        test_r2_min=("test_r2", "min"),
        test_r2_max=("test_r2", "max"),
        test_nmse_mean=("test_nmse", "mean"),
        train_r2_mean=("train_r2", "mean"),
        best_val_mse_mean=("best_val_mse", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        epochs_run_mean=("epochs_run", "mean"),
    )
    summary = summary.sort_values("test_r2_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(args.out / "optimization_summary.csv", index=False)

    best = summary.iloc[0]
    configs_reaching = summary[summary["test_r2_mean"] >= args.target_r2]
    individual_success_rate = float((raw["test_r2"] >= args.target_r2).mean())

    if len(configs_reaching) > 0:
        diagnosis = "optimization_limited_50k_sufficient"
        recommendation = (
            "At least one fixed-architecture optimization protocol reaches the R2 target on mean across seeds. "
            "Do not generate a larger 20-layer dataset for this target."
        )
    else:
        # Quantify whether configurations converge to a common plateau.
        top3 = summary.head(min(3, len(summary)))
        top3_range = float(top3["test_r2_mean"].max() - top3["test_r2_mean"].min())
        train_test_gap = float(best["train_r2_mean"] - best["test_r2_mean"])
        diagnosis = "data_expansion_justified_after_optimization_check"
        recommendation = (
            "No well-tuned fixed-architecture protocol reaches the R2 target. "
            "Generate additional 20-layer samples and run a controlled learning-curve experiment; "
            "start with 75k and 100k total samples before committing to 150k-200k."
        )

    top3 = summary.head(min(3, len(summary)))
    top3_range = float(top3["test_r2_mean"].max() - top3["test_r2_mean"].min())
    train_test_gap = float(best["train_r2_mean"] - best["test_r2_mean"])

    decision = pd.DataFrame([{
        "layers": LAYER,
        "target_r2": args.target_r2,
        "architecture_width": WIDTH,
        "architecture_depth": DEPTH,
        "parameters": nparams,
        "best_config": best["config"],
        "best_test_r2_mean": float(best["test_r2_mean"]),
        "best_test_r2_std": float(best["test_r2_std"]),
        "best_train_r2_mean": float(best["train_r2_mean"]),
        "best_train_test_r2_gap": train_test_gap,
        "top3_config_mean_r2_range": top3_range,
        "configs_reaching_target": int(len(configs_reaching)),
        "individual_run_success_rate": individual_success_rate,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }])
    decision.to_csv(args.out / "optimization_decision.csv", index=False)

    # Summary figure
    plot_df = summary.sort_values("test_r2_mean")
    plt.figure(figsize=(9, 5.5))
    y = np.arange(len(plot_df))
    plt.errorbar(
        plot_df["test_r2_mean"],
        y,
        xerr=plot_df["test_r2_std"].fillna(0),
        fmt="o",
        capsize=3,
    )
    plt.axvline(args.target_r2, linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["config"])
    plt.xlabel("Test R² (mean ± SD across seeds)")
    plt.tight_layout()
    plt.savefig(figdir / "optimization_comparison.png", dpi=300)
    plt.savefig(figdir / "optimization_comparison.pdf")
    plt.close()

    print("\nOptimization summary:\n")
    print(summary[["config", "test_r2_mean", "test_r2_std", "train_r2_mean", "best_epoch_mean"]].to_string(index=False))
    print("\nDecision:\n")
    print(decision[["best_config", "best_test_r2_mean", "diagnosis", "recommendation"]].to_string(index=False))
    print(f"\nDone. Results written to: {args.out.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("paper1_20layer_optimization"))
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--scheduler-patience", type=int, default=25)
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--target-r2", type=float, default=TARGET_R2)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
