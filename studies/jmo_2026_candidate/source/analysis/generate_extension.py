#!/usr/bin/env python3
"""Relabel exact augmented designs with epsilon_r(SiO2)=2.1316.

Use the corrected primary 50k archive for seed-specific test/validation sets.
Provide the original Table 2 data-and-run ZIP to retain the identical extra
designs, or regenerate them from the recorded seeds if it is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

LAYERS = (14, 16, 18)
CHECK_INDICES = [0, 1, 49, 499, 4_999, 24_999, 49_999]
SOURCE_MEMBERS = "Datasets/SiO2_TiO2_{layer}_layer_50K_200_points_{kind}.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def member_sha(z: ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with z.open(name) as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--primary-manifest", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-augmented-zip", type=Path,
                   help="Reconstructed ORIGINAL Table2_Extended_Data_and_Run_Package.zip (designs only)")
    p.add_argument("--extra-per-layer", type=int, default=160_000)
    p.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    p.add_argument("--seed-base", type=int, default=2_026_102_500)
    p.add_argument("--chunk-size", type=int, default=2_048)
    args = p.parse_args()
    if any(l not in LAYERS for l in args.layers) or len(set(args.layers)) != len(args.layers):
        p.error("Choose unique layers from 14, 16, 18")
    if args.extra_per_layer < 1 or args.chunk_size < 1:
        p.error("extra-per-layer and chunk-size must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Use a fresh empty --out; previous augmented results must remain untouched")
    args.out.mkdir(parents=True, exist_ok=True)
    gen = load_generator(args.generator)
    primary_manifest = json.loads(args.primary_manifest.read_text())
    if gen.EPS_SIO2 != 2.1316 or primary_manifest.get("epsilon_r_SiO2") != 2.1316:
        raise ValueError("Generator or primary manifest is not the corrected epsilon=2.1316 version")
    if sha256(args.primary) != primary_manifest.get("small_zip_sha256"):
        raise ValueError("Primary ZIP hash does not match corrected material-model manifest")
    if args.source_augmented_zip and not args.source_augmented_zip.is_file():
        raise FileNotFoundError(args.source_augmented_zip)
    manifest = {
        "primary_archive_sha256": sha256(args.primary),
        "primary_manifest_sha256": sha256(args.primary_manifest),
        "source_augmented_archive_sha256": sha256(args.source_augmented_zip) if args.source_augmented_zip else None,
        "generator_sha256": sha256(args.generator),
        "generation_script_sha256": sha256(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "generator_status": "epsilon-only rerun of reconstruction; original Python source unknown",
        "epsilon_r_SiO2": gen.EPS_SIO2,
        "epsilon_r_TiO2": gen.EPS_TIO2,
        "response": "field-transmission magnitude |t|, last thickness inactive",
        "frequency_grid_THz": [250.0, 1050.0, 200],
        "design_sampling": "independent numpy.default_rng uniform 30-70 nm",
        "extra_per_layer": args.extra_per_layer,
        "cases": {},
    }
    with ZipFile(args.primary) as z:
        for layer in args.layers:
            names = {k: SOURCE_MEMBERS.format(layer=layer, kind=k) for k in ("DS", "RS")}
            with z.open(names["DS"]) as f:
                old_x = np.loadtxt(f, delimiter=",", dtype=np.float64)
            with z.open(names["RS"]) as f:
                old_y = np.loadtxt(f, delimiter=",", dtype=np.float32)
            if old_x.shape != (layer, 50_000) or old_y.shape != (200, 50_000):
                raise ValueError(f"Unexpected primary shapes for L{layer}: {old_x.shape}, {old_y.shape}")
            spot = gen.legacy_response(old_x[:, CHECK_INDICES], chunk_size=args.chunk_size)
            delta = np.abs(spot - old_y[:, CHECK_INDICES])
            mae, max_error = float(delta.mean()), float(delta.max())
            if mae >= 2e-5 or max_error >= 1e-3:
                raise ValueError(f"Archived-label check failed for L{layer}: MAE={mae}, max={max_error}")
            print(f"L{layer} corrected-primary QC: mean abs error {mae:.8g}, max {max_error:.8g}", flush=True)
            seed = args.seed_base + layer
            if args.source_augmented_zip:
                member = f"data/extra_L{layer}_{args.extra_per_layer}.npz"
                with ZipFile(args.source_augmented_zip) as source:
                    with np.load(io.BytesIO(source.read(member))) as arrays:
                        new_x = np.asarray(arrays["designs_m"], dtype=np.float64).T
                if new_x.shape != (layer, args.extra_per_layer):
                    raise ValueError(f"Original augmented design shape mismatch for L{layer}")
                design_origin = f"original augmented archive member: {member}"
            else:
                new_x = gen.generate_designs(layer, args.extra_per_layer, seed)
                design_origin = "regenerated with recorded seed (verify against original archive if available)"
            new_y = gen.legacy_response(new_x, chunk_size=args.chunk_size).T.astype(np.float32)
            if not np.isfinite(new_x).all() or not np.isfinite(new_y).all():
                raise ValueError(f"Nonfinite generated values for L{layer}")
            if not (new_x.min() >= 30e-9 and new_x.max() <= 70e-9):
                raise ValueError(f"Generated thickness outside 30-70 nm for L{layer}")
            out = args.out / f"extra_L{layer}_{args.extra_per_layer}.npz"
            np.savez_compressed(out, designs_m=new_x.T, response_abs_t=new_y)
            manifest["cases"][str(layer)] = {
                "seed": seed,
                "file": out.name,
                "file_sha256": sha256(out),
                "rows": args.extra_per_layer,
                "response_points": 200,
                "design_origin": design_origin,
                "corrected_primary_spotcheck_mae": mae,
                "corrected_primary_spotcheck_max_error": max_error,
                "source_member_sha256": {k: member_sha(z, n) for k, n in names.items()},
            }
            (args.out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Wrote {out} ({out.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
