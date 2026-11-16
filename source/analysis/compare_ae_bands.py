#!/usr/bin/env python3
"""Compare 20-layer autoencoder reconstruction across two spectral grids.

Uses the exact AutoEncoder, split, optimizer, early stopping, and R2 functions
from the Response Complexity Tracks Learning Resources repository. The legacy
response generator supplies the two crossed, same-design bandwidth conditions.

Requires numpy and scikit-learn for --inspect-only. Training also requires
PyTorch as required by the repository's original AE pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

OLD_DS = "old data sets/Dataset_20/SiO2_TiO2_20layer_DS.csv"
CURRENT_DS = "Datasets/SiO2_TiO2_20_layer_50K_200_points_DS.csv"
CURRENT_RS = "Datasets/SiO2_TiO2_20_layer_50K_200_points_RS.csv"
ALL_CASES = ("old_narrow", "old_wide", "current_narrow", "current_wide")
FIELDS = ["case", "latent_dim", "seed", "epochs", "val_mse", "test_mse", "test_mae", "test_nmse", "test_r2"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrices(path: Path, ds_name: str, rs_name: str | None = None):
    with zipfile.ZipFile(path) as z:
        with z.open(ds_name) as f:
            design = np.loadtxt(f, delimiter=",", dtype=np.float64)
        if rs_name is not None:
            with z.open(rs_name) as f:
                response = np.loadtxt(f, delimiter=",", dtype=np.float32).T.copy()
        else:
            response = None
    if design.shape != (20, 50_000) or (response is not None and response.shape != (50_000, 200)):
        raise ValueError(f"Unexpected 20-layer shapes: {design.shape}, {None if response is None else response.shape}")
    if not np.isfinite(design).all() or (response is not None and not np.isfinite(response).all()):
        raise ValueError("Nonfinite values in source archive")
    return design, response


def calculate(gen, designs: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    gen.FREQ_THz = np.linspace(bounds[0], bounds[1], 200, dtype=np.float64)
    return gen.legacy_response(designs, chunk_size=2048).T.astype(np.float32)


def pca_check(y: np.ndarray) -> dict:
    # Same 80/10/10 index split and global R2 denominator as the AE pipeline.
    tr, tmp = train_test_split(np.arange(len(y)), test_size=.20, random_state=20260816, shuffle=True)
    _, te = train_test_split(tmp, test_size=.50, random_state=20260816, shuffle=True)
    train = y[tr].astype(np.float64)
    test = y[te].astype(np.float64)
    mu = train.mean(0)
    centered = train - mu
    eig, vec = np.linalg.eigh(centered.T @ centered / len(train))
    eig = np.clip(eig[::-1], 0, None)
    vec = vec[:, ::-1]
    cumulative = np.cumsum(eig) / eig.sum()
    scores = (test - mu) @ vec
    total = np.sum((test - test.mean()) ** 2)
    result = {}
    for k in (10, 20, 30):
        residual = np.sum(scores[:, k:] ** 2)
        result[str(k)] = round(float(1 - residual / total), 8)
    return {
        "pca99_train_centered_variance": int(np.searchsorted(cumulative, .99) + 1),
        "heldout_global_r2_by_pca_dim": result,
    }


def load_cases(args, gen):
    # Ignore historical narrow response labels: they use epsilon=1.46.
    old_designs, _ = matrices(args.old, OLD_DS)
    current_designs, current_wide = matrices(args.current, CURRENT_DS, CURRENT_RS)
    current_check = calculate(gen, current_designs[:, [0, 1, 49, 499, 4999, 24999, 49999]], (250, 1050))
    current_mae = float(np.mean(np.abs(current_check - current_wide[[0, 1, 49, 499, 4999, 24999, 49999]])))
    if current_mae >= 2e-5:
        raise ValueError(f"Corrected labels fail corrected-generator spot check: {current_mae}")
    cases = {
        "old_narrow": calculate(gen, old_designs, (350, 750)),
        "old_wide": calculate(gen, old_designs, (250, 1050)),
        "current_narrow": calculate(gen, current_designs, (350, 750)),
        "current_wide": current_wide,
    }
    return cases, {"corrected_current_spotcheck_mae": current_mae,
                   "old_historical_response_csv_used": False}


def training(args, cases, pipeline, metadata):
    import torch

    output = args.output / "ae_results.csv"
    meta_path = args.output / "experiment.json"
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but PyTorch reports no CUDA device")
    if device == "mps" and not mps_available:
        raise ValueError("MPS requested but PyTorch reports no available Apple GPU")
    metadata["device"] = device
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError("Existing output has a different experiment definition; choose another --output directory")
    else:
        if output.exists():
            raise ValueError("Existing result CSV lacks matching experiment metadata")
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    completed = set()
    if output.exists():
        with output.open(newline="") as f:
            completed = {(r["case"], int(r["latent_dim"]), int(r["seed"])) for r in csv.DictReader(f)}
    print("Training device:", device, flush=True)
    tr, va, te = pipeline.fixed_split(50_000)
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        for case in args.cases:
            y = cases[case]
            ytr, yv, yte = y[tr], y[va], y[te]
            for z in args.latents:
                for seed in args.seeds:
                    if (case, z, seed) in completed:
                        continue
                    pipeline.seed_everything(seed)
                    model = pipeline.AutoEncoder(200, z)
                    model, epochs, val_mse = pipeline.train_torch_model(
                        model, ytr, ytr, yv, yv, epochs=args.epochs,
                        batch_size=512, lr=1e-3, patience=15, device=device,
                    )
                    pred = pipeline.predict_numpy(model, yte, device)
                    mse, mae, nmse, r2 = pipeline.response_metrics(yte, pred)
                    row = {"case": case, "latent_dim": z, "seed": seed,
                           "epochs": epochs, "val_mse": val_mse,
                           "test_mse": mse, "test_mae": mae,
                           "test_nmse": nmse, "test_r2": r2}
                    writer.writerow(row)
                    f.flush()
                    print(row, flush=True)
                    del model, pred
    rows = list(csv.DictReader(output.open(newline="")))
    summary = []
    for case in args.cases:
        for z in args.latents:
            scores = [float(r["test_r2"]) for r in rows if r["case"] == case and int(r["latent_dim"]) == z]
            if len(scores) == len(args.seeds):
                summary.append({"case": case, "latent_dim": z, "n_seeds": len(scores),
                                "mean_test_r2": float(np.mean(scores)),
                                "sd_test_r2": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0})
    with (args.output / "ae_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "latent_dim", "n_seeds", "mean_test_r2", "sd_test_r2"])
        writer.writeheader()
        writer.writerows(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True, help="Uploaded old data sets.zip")
    p.add_argument("--current", type=Path, required=True, help="NEW Datasets50k_eps21316.zip")
    p.add_argument("--primary-manifest", type=Path, required=True,
                   help="Manifest associated with the corrected current ZIP")
    p.add_argument("--repo", type=Path, required=True, help="Current paper's reproducibility repository")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=ALL_CASES, default=list(ALL_CASES))
    p.add_argument("--latents", nargs="+", type=int, default=[10, 20, 24, 30])
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--inspect-only", action="store_true", help="Run source QC and matched PCA, without PyTorch")
    args = p.parse_args()
    if any(z < 1 or z > 200 for z in args.latents):
        p.error("latent dimensions must be between 1 and 200")
    if len(set(args.latents)) != len(args.latents) or len(set(args.seeds)) != len(args.seeds):
        p.error("latent dimensions and seeds must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    primary_manifest = json.loads(args.primary_manifest.read_text())
    if primary_manifest.get("epsilon_r_SiO2") != 2.1316 or primary_manifest.get("small_zip_sha256") != digest(args.current):
        raise ValueError("--current is not the corrected 2.1316 primary ZIP")
    generator_path = args.repo / "data_generation/generate_legacy_multilayer.py"
    pipeline_path = args.repo / "code/paper1_intrinsic_dimensionality_pipeline_fixed.py"
    gen = import_file(generator_path, "legacy_gen")
    if gen.EPS_SIO2 != 2.1316:
        raise ValueError("Spectral generator must use epsilon_r(SiO2)=2.1316")
    cases, checks = load_cases(args, gen)
    metadata = {
        "old_zip_sha256": digest(args.old), "current_zip_sha256": digest(args.current),
        "primary_manifest_sha256": digest(args.primary_manifest),
        "generator_sha256": digest(generator_path), "pipeline_sha256": digest(pipeline_path),
        "frequency_grids_THz": {"narrow": [350, 750, 200], "wide": [250, 1050, 200]},
        "split": "sklearn train_test_split 80/10/10, random_state=20260816",
        "cases": args.cases, "latents": args.latents, "seeds": args.seeds,
        "epochs": args.epochs, "batch_size": 512, "lr": 1e-3, "patience": 15,
        "response": "epsilon=2.1316 field-transmission magnitude |t|; legacy last thickness inactive",
        "data_spot_checks": checks,
    }
    diagnostic = {"metadata": metadata, "pca": {name: pca_check(y) for name, y in cases.items()}}
    (args.output / "pca_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print("Wrote", args.output / "pca_diagnostic.json", flush=True)
    if args.inspect_only:
        return
    pipeline = import_file(pipeline_path, "paper_ae_pipeline")
    training(args, cases, pipeline, metadata)


if __name__ == "__main__":
    main()
