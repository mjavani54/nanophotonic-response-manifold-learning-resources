#!/usr/bin/env python3
"""
Paper 1 — Controlled Cross-Structure Learning-Curve Experiment

Goal
----
Test whether intrinsic response-manifold complexity predicts training-sample
requirements once model-capacity limitations are removed.

Structures:
    8, 10, 12, 14, 16, 18, 20 layers

Data:
    --zip-small : ZIP containing the original 8–18 layer datasets
    --zip-20    : ZIP containing the 200k 20-layer dataset

Design
------
* Same validation size and test size for every structure.
* Same optimizer, scheduler, stopping rule, batch size, and preprocessing.
* Nested training subsets for each seed.
* Three seeds by default.
* Capacity-appropriate architecture for each structure is FIXED before the
  learning-curve sweep.

Default architecture widths (4 hidden layers):
    8L:   64
    10L:  64
    12L:  128
    14L:  128
    16L:  256
    18L:  448
    20L:  896

These widths are deliberately conservative: they are intended to remove
under-capacity as the dominant limitation while still preserving a monotonic,
predeclared capacity schedule.

Default training sizes:
    5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k

The 8–18 datasets contain 50k total samples, so with 5k validation and 5k test,
40k is the maximum available training size. The 20-layer dataset is larger, but
the same 40k ceiling is used in the PRIMARY controlled comparison so that all
structures are compared on exactly the same sample-size grid.

Primary endpoint:
    minimum N_train achieving mean test R^2 >= 0.99

Secondary endpoints:
    thresholds at 0.995 and 0.999 when resolved
    generalization gap
    correlation between intrinsic dimensionality and required sample count

Typical usage
-------------
python paper1_cross_structure_learning_curves.py \
    --zip-small "Datasets(2).zip" \
    --zip-20 "Datasets20.zip" \
    --out paper1_cross_structure_learning_curves
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

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ----------------------------- Reproducibility ----------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ZIP extraction ------------------------------ #

def extract_zip_fresh(zip_path: Path, out_root: Path, tag: str) -> Path:
    """
    Extracts ZIP into a tag-specific folder.
    Re-extracts automatically if no CSV files are found.
    """
    target = out_root / "extracted" / tag
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
            f"No CSV files were found after extracting {zip_path} into {target}"
        )
    return target


# ----------------------------- File discovery ------------------------------ #

def find_layer_file(root: Path, layer: int, kind: str) -> Path:
    """
    kind = 'DS' or 'RS'

    Match the layer count as an actual layer token, so 8-layer does not
    accidentally match 18-layer.
    """
    kind_low = kind.lower()
    csvs = list(root.rglob("*.csv"))

    # Accept common forms such as:
    #   8_layer, 8-layer, 8 layer, 8layer, layer_8, layer8
    layer_patterns = [
        re.compile(rf"(?<!\d){layer}[\s_-]*layer(?!\d)", re.I),
        re.compile(rf"layer[\s_-]*{layer}(?!\d)", re.I),
    ]

    candidates = []
    for p in csvs:
        stem = p.stem
        stem_low = stem.lower()

        # Require DS/RS as a distinct filename token where possible.
        kind_ok = (
            re.search(rf"(^|[\s_-]){re.escape(kind_low)}($|[\s_-])", stem_low, re.I)
            is not None
            or stem_low.endswith(kind_low)
        )
        layer_ok = any(pat.search(stem) for pat in layer_patterns)

        if kind_ok and layer_ok:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify {kind} file for {layer} layers.\n"
        f"Candidates: {[str(p) for p in candidates]}"
    )


def find_20_file(root: Path, kind: str) -> Path:
    kind_low = kind.lower()
    csvs = list(root.rglob("*.csv"))
    cands = [p for p in csvs if kind_low in p.stem.lower()]

    if len(cands) == 1:
        return cands[0]

    cands20 = [p for p in cands if "20" in p.stem.lower()]
    if len(cands20) == 1:
        return cands20[0]

    raise RuntimeError(
        f"Could not uniquely identify 20-layer {kind} file.\n"
        f"Candidates: {[str(p) for p in cands]}"
    )


# ------------------------------ Data loading ------------------------------- #

def load_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def orient_pair(ds: np.ndarray, rs: np.ndarray, layer: int):
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
        raise ValueError(
            f"{layer}L sample mismatch after orientation: X={X.shape}, Y={Y.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{layer}L DS contains NaN/Inf.")
    if not np.isfinite(Y).all():
        raise ValueError(f"{layer}L RS contains NaN/Inf.")

    return X.astype(np.float32), Y.astype(np.float32)


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


# --------------------------------- Model ------------------------------------ #

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=200, width=128, depth=4):
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


# -------------------------------- Metrics ---------------------------------- #

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


# -------------------------------- Training --------------------------------- #

def train_one(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    layer, width, depth,
    seed, device,
    batch_size=1024,
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
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = MLP(in_dim=layer, width=width, depth=depth).to(device)
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

        for xb, yb in train_loader:
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
        raise RuntimeError("No valid model state was obtained.")

    model.load_state_dict(best_state)
    model.eval()

    def predict(raw_x, chunk=8192):
        xs = sx.transform(raw_x)
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), chunk):
                xb = torch.from_numpy(xs[i:i+chunk]).to(device)
                out.append(model(xb).cpu().numpy())
        pred_std = np.concatenate(out, axis=0)
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


# ------------------------------ Main routine -------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-small", required=True,
                    help="ZIP containing 8–18 layer datasets.")
    ap.add_argument("--zip-20", required=True,
                    help="ZIP containing 200k 20-layer dataset.")
    ap.add_argument("--out", default="paper1_cross_structure_learning_curves")

    ap.add_argument("--train-sizes", nargs="+", type=int,
                    default=[5000,10000,15000,20000,25000,30000,35000,40000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])

    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=5000)

    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    small_root = extract_zip_fresh(Path(args.zip_small), out, "small")
    root20 = extract_zip_fresh(Path(args.zip_20), out, "layer20")

    # Capacity-appropriate fixed widths.
    widths = {
        8: 64,
        10: 64,
        12: 128,
        14: 128,
        16: 256,
        18: 448,
        20: 896,
    }

    # Intrinsic dimensionality values from the completed full analysis.
    intrinsic = pd.DataFrame({
        "layers": [8,10,12,14,16,18,20],
        "participation_ratio": [
            2.358282,3.148455,4.200912,5.357589,6.619628,7.933762,9.275467
        ],
        "pca_dim_99": [9,14,19,26,34,43,53],
        "pca_dim_995": [11,16,23,31,40,52,66],
        "pca_dim_999": [14,21,31,43,59,79,101],
        "ae_dim_99": [5,6,8,10,14,20,30],
    })
    intrinsic.to_csv(out / "intrinsic_dimension_reference.csv", index=False)

    all_results = []
    qc_rows = []

    for layer in [8,10,12,14,16,18,20]:
        print(f"\n================ {layer} layers ================")

        if layer < 20:
            ds_path = find_layer_file(small_root, layer, "DS")
            rs_path = find_layer_file(small_root, layer, "RS")
        else:
            ds_path = find_20_file(root20, "DS")
            rs_path = find_20_file(root20, "RS")

        print(f"DS: {ds_path}")
        print(f"RS: {rs_path}")

        ds = load_csv(ds_path)
        rs = load_csv(rs_path)
        X, Y = orient_pair(ds, rs, layer)

        n_total = len(X)
        max_train = n_total - args.val_size - args.test_size
        valid_sizes = [n for n in args.train_sizes if n <= max_train]

        if not valid_sizes:
            raise RuntimeError(
                f"{layer}L has no valid training sizes after holdouts."
            )

        qc_rows.append({
            "layers": layer,
            "n_total": n_total,
            "x_dim": X.shape[1],
            "y_dim": Y.shape[1],
            "x_min": float(X.min()),
            "x_max": float(X.max()),
            "y_min": float(Y.min()),
            "y_max": float(Y.max()),
            "val_size": args.val_size,
            "test_size": args.test_size,
            "max_train": max_train,
            "width": widths[layer],
            "depth": args.depth,
        })

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_total)

            test_idx = perm[:args.test_size]
            val_idx = perm[args.test_size:args.test_size + args.val_size]
            pool_idx = perm[args.test_size + args.val_size:]

            X_test, Y_test = X[test_idx], Y[test_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            for n_train in valid_sizes:
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
                    depth=args.depth,
                    seed=seed,
                    device=device,
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
                    "n_total_dataset": n_total,
                    "n_train": n_train,
                    "n_val": args.val_size,
                    "n_test": args.test_size,
                    "width": widths[layer],
                    "depth": args.depth,
                    **res,
                }
                all_results.append(row)

                pd.DataFrame(all_results).to_csv(
                    out / "cross_structure_learning_curve_results.csv",
                    index=False
                )

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out / "cross_structure_qc.csv", index=False)

    results = pd.DataFrame(all_results)
    results.to_csv(out / "cross_structure_learning_curve_results.csv", index=False)

    summary = results.groupby(["layers","n_train"]).agg(
        n_seeds=("seed","count"),
        width=("width","first"),
        depth=("depth","first"),
        n_parameters=("n_parameters","first"),
        epochs_mean=("epochs_trained","mean"),
        epochs_std=("epochs_trained","std"),
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
    summary.to_csv(out / "cross_structure_learning_curve_summary.csv", index=False)

    # Threshold crossing estimates.
    threshold_rows = []
    for layer in sorted(summary.layers.unique()):
        sub = summary[summary.layers == layer].sort_values("n_train")
        for threshold in [0.99, 0.995, 0.999]:
            passing = sub[sub.test_r2_mean >= threshold]
            if len(passing):
                row = passing.iloc[0]
                threshold_rows.append({
                    "layers": layer,
                    "threshold_r2": threshold,
                    "resolved": True,
                    "minimum_training_samples": int(row.n_train),
                    "test_r2_mean_at_crossing": float(row.test_r2_mean),
                    "test_r2_std_at_crossing": float(row.test_r2_std),
                    "n_parameters": int(row.n_parameters),
                })
            else:
                threshold_rows.append({
                    "layers": layer,
                    "threshold_r2": threshold,
                    "resolved": False,
                    "minimum_training_samples": np.nan,
                    "test_r2_mean_at_crossing": np.nan,
                    "test_r2_std_at_crossing": np.nan,
                    "n_parameters": int(sub.n_parameters.iloc[0]),
                })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out / "cross_structure_thresholds.csv", index=False)

    # Primary R2=0.99 table merged with intrinsic dimensions.
    t99 = thresholds[thresholds.threshold_r2 == 0.99].copy()
    master = intrinsic.merge(t99, on="layers", how="left")
    master.to_csv(out / "cross_structure_master_summary.csv", index=False)

    # Correlations only for resolved structures.
    corr_rows = []
    resolved = master[
        (master["resolved"] == True) &
        master["minimum_training_samples"].notna()
    ].copy()

    metrics = [
        "participation_ratio",
        "pca_dim_99",
        "pca_dim_995",
        "pca_dim_999",
        "ae_dim_99",
    ]

    if SCIPY_OK and len(resolved) >= 4:
        y = resolved["minimum_training_samples"].to_numpy(float)
        for m in metrics:
            x = resolved[m].to_numpy(float)
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            corr_rows.append({
                "complexity_metric": m,
                "n_structures": len(resolved),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
            })

            # Also log10 sample-count correlation.
            ylog = np.log10(y)
            prl, ppl = pearsonr(x, ylog)
            srl, spl = spearmanr(x, ylog)
            corr_rows.append({
                "complexity_metric": m + "_vs_log10_samples",
                "n_structures": len(resolved),
                "pearson_r": float(prl),
                "pearson_p": float(ppl),
                "spearman_rho": float(srl),
                "spearman_p": float(spl),
            })

    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(out / "sample_complexity_correlations.csv", index=False)

    # ------------------------------- Figures ------------------------------- #

    # 1. All learning curves.
    fig = plt.figure(figsize=(9,6))
    for layer in sorted(summary.layers.unique()):
        sub = summary[summary.layers == layer].sort_values("n_train")
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
    plt.title("Controlled Learning Curves Across Structural Complexity")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "fig_learning_curves_all_layers.png", dpi=300)
    plt.close(fig)

    # 2. Required samples vs intrinsic dimension.
    if len(resolved) >= 2:
        for m in metrics:
            fig = plt.figure(figsize=(7.5,5.5))
            plt.scatter(resolved[m], resolved.minimum_training_samples)
            for _, r in resolved.iterrows():
                plt.annotate(
                    f"{int(r.layers)}L",
                    (r[m], r.minimum_training_samples),
                    xytext=(5,5),
                    textcoords="offset points"
                )
            plt.xlabel(m.replace("_"," "))
            plt.ylabel(r"Minimum samples for mean test $R^2 \geq 0.99$")
            plt.title("Intrinsic Complexity vs Sample Requirement")
            plt.tight_layout()
            fig.savefig(out / f"fig_samples_vs_{m}.png", dpi=300)
            plt.close(fig)

    # 3. Generalization gap.
    fig = plt.figure(figsize=(9,6))
    for layer in sorted(summary.layers.unique()):
        sub = summary[summary.layers == layer].sort_values("n_train")
        plt.plot(
            sub.n_train,
            sub.generalization_gap_mean,
            marker="o",
            label=f"{layer} layers"
        )
    plt.xlabel("Number of training samples")
    plt.ylabel(r"Train $R^2$ - Test $R^2$")
    plt.title("Generalization Gap Across Structural Complexity")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "fig_generalization_gap_all_layers.png", dpi=300)
    plt.close(fig)

    # Metadata.
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
        "primary_threshold": 0.99,
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n================ PRIMARY THRESHOLD RESULTS ================")
    print(
        thresholds[thresholds.threshold_r2 == 0.99]
        .to_string(index=False)
    )

    if len(correlations):
        print("\n================ SAMPLE-COMPLEXITY CORRELATIONS ================")
        print(correlations.to_string(index=False))
    else:
        print(
            "\nNo correlation table was produced. This means too few structures "
            "resolved the R2=0.99 threshold or SciPy is unavailable."
        )

    print(f"\nAll outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
