#!/usr/bin/env python3
"""
Paper 1 — targeted high-capacity diagnostic
===========================================
Purpose
-------
Determine whether the 14/16/18/20-layer forward-model results are limited by
network capacity or by the available 50,000-sample dataset.

The experiment holds the data split and training protocol fixed, uses the full
available training partition (40,000 samples), and sweeps a finer family of
4-hidden-layer ReLU networks. Three seeds are used by default.

Interpretation
--------------
* If R^2 >= target is reached as width increases, the previous result was
  capacity-limited at that target.
* If the largest models still improve materially with width, capacity remains
  unresolved; do NOT generate more data yet.
* If the largest models plateau below target across seeds, the existing data /
  optimization protocol is the next likely bottleneck. That is the point at
  which a larger dataset becomes scientifically justified.

Example
-------
python paper1_high_capacity_diagnostic.py --zip "Datasets(2).zip" --out paper1_capacity_diagnostic

Or, if already extracted:
python paper1_high_capacity_diagnostic.py --data-dir /path/to/Datasets --out paper1_capacity_diagnostic
"""

from __future__ import annotations
import argparse, math, random, shutil, zipfile
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

TARGET_LAYERS = [14, 16, 18, 20]
EXPECTED_SAMPLES = 50_000
EXPECTED_RESPONSE_DIM = 200
DEFAULT_SEEDS = [11, 29, 47]

# A fixed depth avoids mixing width and depth when estimating the capacity threshold.
# Grids are centered on the transition regions observed in the first full run.
WIDTH_GRID = {
    14: [128, 160, 192, 224, 256, 320, 384],
    16: [256, 320, 384, 448, 512, 576, 640, 768],
    18: [320, 384, 448, 512, 576, 640, 768, 896],
    20: [448, 512, 576, 640, 768, 896, 1024, 1280],
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_matrix(path: Path) -> np.ndarray:
    # Files are features x samples; models use samples x features.
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
    candidates = list(extract_dir.rglob("SiO2_TiO2_14_layer_50K_200_points_RS.csv"))
    if not candidates:
        raise FileNotFoundError("Could not locate the multilayer CSV files after extraction.")
    return candidates[0].parent


def paths_for_layer(root: Path, layer: int):
    ds = root / f"SiO2_TiO2_{layer}_layer_50K_200_points_DS.csv"
    rs = root / f"SiO2_TiO2_{layer}_layer_50K_200_points_RS.csv"
    if not ds.exists() or not rs.exists():
        raise FileNotFoundError(f"Missing DS/RS files for {layer} layers")
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
    return (Xtr-lo)/span, (Xv-lo)/span, (Xte-lo)/span


def response_metrics(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    nmse = mse / var if var > 0 else np.nan
    sse = float(np.sum(err**2))
    sst = float(np.sum((y_true-y_true.mean())**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    return mse, mae, nmse, r2


class ForwardNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int = 4):
        super().__init__()
        dims = [input_dim] + [width]*depth + [output_dim]
        modules = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            modules += [nn.Linear(a, b), nn.ReLU()]
        modules += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def train_model(model, Xtr, Ytr, Xv, Yv, *, epochs, batch_size, lr, patience, device):
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),
                      torch.from_numpy(Ytr.astype(np.float32))),
        batch_size=batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available())
    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    Yv_t = torch.from_numpy(Yv.astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = math.inf
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv_t), Yv_t).item()
        if vloss < best - 1e-8:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epoch+1, best


def predict_numpy(model, X, device, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(device)
            out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def qc(root: Path, out: Path):
    rows = []
    for L in TARGET_LAYERS:
        ds, rs = paths_for_layer(root, L)
        X, Y = read_csv_matrix(ds), read_csv_matrix(rs)
        ok = X.shape == (EXPECTED_SAMPLES, L) and Y.shape == (EXPECTED_SAMPLES, EXPECTED_RESPONSE_DIM)
        rows.append({"layers": L, "X_shape": str(X.shape), "Y_shape": str(Y.shape), "shape_ok": ok,
                     "X_nan": int(np.isnan(X).sum()), "Y_nan": int(np.isnan(Y).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out/"capacity_diagnostic_qc.csv", index=False)
    if not df["shape_ok"].all() or (df[["X_nan","Y_nan"]].to_numpy() != 0).any():
        raise RuntimeError("QC failed. Inspect capacity_diagnostic_qc.csv")


def run_diagnostic(root: Path, out: Path, device: str, seeds, epochs: int, patience: int,
                   batch_size: int, lr: float):
    rows = []
    result_path = out/"high_capacity_results.csv"
    for L in TARGET_LAYERS:
        ds, rs = paths_for_layer(root, L)
        X, Y = read_csv_matrix(ds), read_csv_matrix(rs)
        tr, va, te = fixed_split(len(X))
        Xtr, Xv, Xte = X[tr], X[va], X[te]
        Ytr, Yv, Yte = Y[tr], Y[va], Y[te]
        Xtr, Xv, Xte = normalize_design(Xtr, Xv, Xte)
        print(f"\n{L} layers: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")
        for width in WIDTH_GRID[L]:
            for seed in seeds:
                seed_everything(seed)
                model = ForwardNet(L, EXPECTED_RESPONSE_DIM, width=width, depth=4)
                nparams = sum(p.numel() for p in model.parameters())
                print(f"  width={width:4d}, params={nparams:,}, seed={seed}")
                model, ne, bv = train_model(
                    model, Xtr, Ytr, Xv, Yv,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    patience=patience, device=device)
                pred = predict_numpy(model, Xte, device)
                mse, mae, nmse, r2 = response_metrics(Yte, pred)
                rows.append({
                    "layers": L, "depth": 4, "width": width,
                    "parameters": nparams, "train_samples": len(Xtr),
                    "seed": seed, "epochs": ne, "best_val_mse": bv,
                    "test_mse": mse, "test_mae": mae,
                    "test_nmse": nmse, "test_r2": r2,
                })
                pd.DataFrame(rows).to_csv(result_path, index=False)
                print(f"    test R2={r2:.6f}, NMSE={nmse:.6g}, epochs={ne}")
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, out: Path, target: float, plateau_delta: float):
    agg = raw.groupby(["layers","depth","width","parameters"], as_index=False).agg(
        r2_mean=("test_r2","mean"), r2_std=("test_r2","std"),
        nmse_mean=("test_nmse","mean"), nmse_std=("test_nmse","std"),
        epochs_mean=("epochs","mean"))
    agg.to_csv(out/"high_capacity_summary.csv", index=False)

    decisions = []
    for L, g in agg.groupby("layers"):
        g = g.sort_values("parameters").reset_index(drop=True)
        ok = g[g.r2_mean >= target]
        threshold_params = int(ok.parameters.iloc[0]) if len(ok) else np.nan
        threshold_width = int(ok.width.iloc[0]) if len(ok) else np.nan
        best_idx = g.r2_mean.idxmax()
        best = g.loc[best_idx]

        # Assess whether the largest-capacity end has flattened. We intentionally
        # use the last three widths, not just the final pair, to avoid one noisy run.
        tail = g.tail(min(3, len(g)))
        tail_gain = float(tail.r2_mean.max() - tail.r2_mean.min()) if len(tail) >= 2 else np.nan
        last_gain = float(g.r2_mean.iloc[-1] - g.r2_mean.iloc[-2]) if len(g) >= 2 else np.nan

        if len(ok):
            diagnosis = "capacity_threshold_resolved"
            next_action = "No larger dataset needed for the R2 target; use this capacity family for learning-curve tests."
        elif np.isfinite(tail_gain) and tail_gain <= plateau_delta and best.r2_mean < target:
            diagnosis = "likely_data_or_optimization_limited"
            next_action = "Capacity appears to plateau below target; larger data is now justified, but verify optimization before simulation expansion."
        else:
            diagnosis = "capacity_still_unresolved"
            next_action = "Performance still changes with capacity; extend capacity/optimization before generating more data."

        decisions.append({
            "layers": L,
            "target_r2": target,
            "min_width_reaching_target": threshold_width,
            "min_parameters_reaching_target": threshold_params,
            "best_width": int(best.width),
            "best_parameters": int(best.parameters),
            "best_r2_mean": float(best.r2_mean),
            "best_r2_std": float(best.r2_std) if np.isfinite(best.r2_std) else np.nan,
            "largest_width": int(g.width.iloc[-1]),
            "largest_model_r2": float(g.r2_mean.iloc[-1]),
            "last_step_r2_gain": last_gain,
            "tail_r2_range": tail_gain,
            "diagnosis": diagnosis,
            "next_action": next_action,
        })
    dec = pd.DataFrame(decisions)
    dec.to_csv(out/"capacity_diagnostic_decision.csv", index=False)
    return agg, dec


def make_figures(agg: pd.DataFrame, out: Path, target: float):
    figdir = out/"figures"
    figdir.mkdir(exist_ok=True)

    plt.figure(figsize=(8,6))
    for L, g in agg.groupby("layers"):
        g = g.sort_values("parameters")
        plt.semilogx(g.parameters, g.r2_mean, "o-", label=f"{L} layers")
    plt.axhline(target, linestyle="--", linewidth=1, label=f"R²={target}")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Test R² (mean across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir/"capacity_diagnostic_r2.png", dpi=300)
    plt.savefig(figdir/"capacity_diagnostic_r2.pdf")
    plt.close()

    plt.figure(figsize=(8,6))
    for L, g in agg.groupby("layers"):
        g = g.sort_values("parameters")
        plt.loglog(g.parameters, g.nmse_mean, "o-", label=f"{L} layers")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Test NMSE (mean across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir/"capacity_diagnostic_nmse.png", dpi=300)
    plt.savefig(figdir/"capacity_diagnostic_nmse.pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("paper1_capacity_diagnostic"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--target-r2", type=float, default=0.99)
    ap.add_argument("--plateau-delta", type=float, default=0.001,
                    help="If the range of mean R2 across the three largest widths is <= this and target is missed, flag a likely plateau.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    root = locate_dataset_root(args.zip, args.data_dir, args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset root: {root}")
    qc(root, args.out)
    print("QC passed")

    raw = run_diagnostic(root, args.out, device, args.seeds, args.epochs,
                         args.patience, args.batch_size, args.lr)
    agg, dec = summarize(raw, args.out, args.target_r2, args.plateau_delta)
    make_figures(agg, args.out, args.target_r2)

    print("\nDecision summary:\n")
    cols = ["layers","min_parameters_reaching_target","best_r2_mean","tail_r2_range","diagnosis"]
    print(dec[cols].to_string(index=False))
    print(f"\nDone. Results written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
