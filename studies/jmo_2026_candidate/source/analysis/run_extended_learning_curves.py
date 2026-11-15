#!/usr/bin/env python3
"""Extend corrected-epsilon 14/16/18-layer learning curves.

Retain the same seed-specific split recipe on the NEW corrected 50k labels.
At sizes above 40,000, append the relabeled extra designs. Replay the NEW
corrected 40,000-sample means before extension; never compare to 1.46 values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

LAYERS = (14, 16, 18)
WIDTHS = {14: 128, 16: 256, 18: 448}
MEMBER = "Datasets/SiO2_TiO2_{layer}_layer_50K_200_points_{kind}.csv"
FIELDS = ["layers", "n_train", "seed", "width", "n_parameters", "epochs_trained",
          "best_val_standardized_mse", "final_lr", "train_r2", "val_r2", "test_r2",
          "test_rmse", "test_mae"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_learning_curve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original learning-curve module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_device(torch, requested: str):
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if mps else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not mps:
        raise RuntimeError("Apple MPS requested but unavailable")
    return torch.device(requested)


def summary(rows: list[dict], output: Path, seeds: list[int]):
    groups = {}
    for r in rows:
        groups.setdefault((int(r["layers"]), int(r["n_train"])), []).append(r)
    out = []
    for (layer, n), group in sorted(groups.items()):
        if {int(r["seed"]) for r in group} != set(seeds):
            continue
        values = np.array([float(r["test_r2"]) for r in group], dtype=np.float64)
        out.append({"layers": layer, "n_train": n, "n_seeds": len(values),
                    "mean_test_r2": float(values.mean()),
                    "sd_test_r2": float(values.std(ddof=1)),
                    "seeds_r2_ge_0999": int((values >= 0.999).sum()),
                    "mean_r2_ge_0999": bool(values.mean() >= 0.999)})
    with (output / "extended_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layers", "n_train", "n_seeds", "mean_test_r2",
                                        "sd_test_r2", "seeds_r2_ge_0999", "mean_r2_ge_0999"])
        w.writeheader()
        w.writerows(out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--primary-manifest", type=Path, required=True)
    p.add_argument("--extra-dir", type=Path, required=True)
    p.add_argument("--pipeline", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True,
                   help="NEW cross_structure_learning_curve_summary.csv produced by corrected rerun")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    p.add_argument("--sizes", type=int, nargs="+", default=[40_000, 50_000, 60_000, 80_000,
                                                           100_000, 120_000, 160_000, 200_000])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--baseline-tolerance", type=float, default=2e-4)
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()
    if any(l not in LAYERS for l in args.layers) or len(set(args.layers)) != len(args.layers):
        p.error("Choose unique layers from 14, 16, 18")
    if sorted(set(args.sizes)) != args.sizes or args.sizes[0] != 40_000:
        p.error("Sizes must be unique, increasing, and start at 40000")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        p.error("Seeds must be distinct and nonempty")
    if args.baseline_tolerance <= 0:
        p.error("Baseline tolerance must be positive")

    try:
        import torch
    except ImportError as e:
        p.error("PyTorch is required for the original model/training protocol")
    device = check_device(torch, args.device)
    pipeline = load_pipeline(args.pipeline)
    args.out.mkdir(parents=True, exist_ok=True)
    primary_manifest = json.loads(args.primary_manifest.read_text())
    if primary_manifest.get("epsilon_r_SiO2") != 2.1316 or sha256(args.primary) != primary_manifest.get("small_zip_sha256"):
        raise ValueError("Wrong or unverified corrected primary dataset")
    manifest = json.loads((args.extra_dir / "generation_manifest.json").read_text())
    if sha256(args.primary) != manifest["primary_archive_sha256"]:
        raise ValueError("Primary archive differs from generation manifest")
    if sha256(args.primary_manifest) != manifest.get("primary_manifest_sha256") or manifest.get("epsilon_r_SiO2") != 2.1316:
        raise ValueError("Augmented cohort manifest is not aligned with corrected primary manifest")
    for layer in args.layers:
        info = manifest["cases"].get(str(layer))
        if info is None or sha256(args.extra_dir / info["file"]) != info["file_sha256"]:
            raise ValueError(f"Generated cohort missing or checksum mismatch for L{layer}")
        if max(args.sizes) > 40_000 + info["rows"]:
            raise ValueError(f"L{layer} has insufficient generated training pairs")
    corrected_reference = pd.read_csv(args.reference)
    if args.reference.name != "cross_structure_learning_curve_summary.csv":
        raise ValueError("The baseline reference must be the NEW cross-structure summary, not frozen release data")
    meta = {
        "primary_sha256": sha256(args.primary),
        "primary_manifest_sha256": sha256(args.primary_manifest),
        "generation_manifest_sha256": sha256(args.extra_dir / "generation_manifest.json"),
        "pipeline_sha256": sha256(args.pipeline),
        "reference_sha256": sha256(args.reference),
        "layers": args.layers, "sizes": args.sizes, "seeds": args.seeds,
        "widths": {str(l): WIDTHS[l] for l in args.layers}, "depth": 4,
        "train_size_original": 40_000, "val_size_original": 5_000,
        "test_size_original": 5_000,
        "batch_size": 1024, "max_epochs": 1200, "patience": 100, "lr": 1e-3,
        "baseline_tolerance": args.baseline_tolerance,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "device": str(device),
        "source_note": "epsilon=2.1316 primary and relabeled extra designs, legacy N-1 thickness geometry",
    }
    meta_file = args.out / "experiment.json"
    if meta_file.exists():
        if json.loads(meta_file.read_text()) != meta:
            raise ValueError("Output directory belongs to a different experiment; choose a fresh --out")
    else:
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
    result_file = args.out / "extended_fit_records.csv"
    rows = list(csv.DictReader(result_file.open(newline=""))) if result_file.exists() else []
    completed = {(int(r["layers"]), int(r["n_train"]), int(r["seed"])) for r in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate fit records in output")

    with ZipFile(args.primary) as z:
        for layer in args.layers:
            names = {k: MEMBER.format(layer=layer, kind=k) for k in ("DS", "RS")}
            with z.open(names["DS"]) as f:
                ds = pd.read_csv(f, header=None).to_numpy(dtype=np.float32)
            with z.open(names["RS"]) as f:
                rs = pd.read_csv(f, header=None).to_numpy(dtype=np.float32)
            X, Y = pipeline.orient_pair(ds, rs, layer)
            with np.load(args.extra_dir / manifest["cases"][str(layer)]["file"]) as npz:
                Xextra = np.asarray(npz["designs_m"], dtype=np.float32)
                Yextra = np.asarray(npz["response_abs_t"], dtype=np.float32)
            assert X.shape == (50_000, layer) and Y.shape == (50_000, 200)
            assert Xextra.shape == (manifest["cases"][str(layer)]["rows"], layer)
            assert Yextra.shape == (len(Xextra), 200)
            print(f"L{layer}: {len(Xextra):,} extra training pairs; device={device}", flush=True)
            seed_data = {}
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(50_000)
                test, val, pool = perm[:5_000], perm[5_000:10_000], perm[10_000:]
                extra_perm = rng.permutation(len(Xextra))
                seed_data[seed] = (test, val, pool, extra_perm)

            def run_size(n: int):
                nonlocal rows, completed
                for seed in args.seeds:
                    key = (layer, n, seed)
                    if key in completed:
                        continue
                    test, val, pool, extra_perm = seed_data[seed]
                    if n == 40_000:
                        train_x, train_y = X[pool], Y[pool]
                    else:
                        n_extra = n - 40_000
                        indices = extra_perm[:n_extra]
                        train_x = np.concatenate((X[pool], Xextra[indices]), axis=0)
                        train_y = np.concatenate((Y[pool], Yextra[indices]), axis=0)
                    print(f"L{layer} n={n:,} seed={seed}: training", flush=True)
                    result = pipeline.train_one(
                        train_x, train_y, X[val], Y[val], X[test], Y[test],
                        layer=layer, width=WIDTHS[layer], depth=4,
                        seed=seed, device=device, batch_size=1024, lr=1e-3,
                        max_epochs=1200, patience=100,
                    )
                    history = result.pop("history")
                    hdir = args.out / "histories"
                    hdir.mkdir(exist_ok=True)
                    np.savez_compressed(hdir / f"L{layer}_n{n}_seed{seed}.npz",
                                        epoch_train_val_lr=np.asarray(history, dtype=np.float64))
                    row = {"layers": layer, "n_train": n, "seed": seed,
                           "width": WIDTHS[layer], **result}
                    with result_file.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow({k: row[k] for k in FIELDS})
                        f.flush()
                    rows.append(row)
                    completed.add(key)
                    summary(rows, args.out, args.seeds)
                    print(f"L{layer} n={n:,} seed={seed}: test R²={result['test_r2']:.8f}", flush=True)

            run_size(40_000)
            sr = summary(rows, args.out, args.seeds)
            mean = next(r["mean_test_r2"] for r in sr if r["layers"] == layer and r["n_train"] == 40_000)
            refrow = corrected_reference[(corrected_reference.layers == layer) & (corrected_reference.n_train == 40_000)]
            if len(refrow) != 1:
                raise ValueError(f"Missing corrected 40k baseline for L{layer}")
            reference = float(refrow.test_r2_mean.iloc[0])
            difference = abs(mean - reference)
            print(f"L{layer} corrected 40k replay: new={mean:.8f}, reference={reference:.8f}, |Δ|={difference:.8g}", flush=True)
            if difference > args.baseline_tolerance:
                raise RuntimeError(f"L{layer} baseline replay exceeded {args.baseline_tolerance}; investigate before extension")
            if not args.baseline_only:
                for n in args.sizes[1:]:
                    run_size(n)
            del X, Y, Xextra, Yextra

    print("All requested fits complete. Inspect extended_summary.csv and experiment.json.", flush=True)


if __name__ == "__main__":
    main()
