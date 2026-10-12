#!/usr/bin/env python3
"""Recalculate approximate paper-format CSVs from stored design matrices.

Exact archived design and response CSVs used in the paper are release assets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import zipfile
import numpy as np

from generate_legacy_multilayer import legacy_response, write_legacy_csv

LAYERS = [8, 10, 12, 14, 16, 18, 20]


def zip_tree(src_dir: Path, zip_path: Path, arc_root: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path(arc_root) / p.relative_to(src_dir)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-npz", type=Path,
                    default=Path("data/exact_designs/paper1_exact_designs.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("generated"))
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--extended-only", action="store_true")
    args = ap.parse_args()

    if args.small_only and args.extended_only:
        ap.error("Choose at most one of --small-only and --extended-only")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.design_npz) as z:
        if not args.extended_only:
            small = args.out_dir / "Datasets"
            if small.exists():
                shutil.rmtree(small)
            small.mkdir()
            for L in LAYERS:
                key = f"L{L}_50k"
                designs = np.asarray(z[key], dtype=np.float64)
                responses = legacy_response(designs, chunk_size=args.chunk_size)
                prefix = f"SiO2_TiO2_{L}_layer_50K_200_points"
                write_legacy_csv(small / f"{prefix}_DS.csv", designs)
                write_legacy_csv(small / f"{prefix}_RS.csv", responses)
                print(f"completed {key}")
            zip_tree(small, args.out_dir / "Datasets50k_regenerated.zip", "Datasets")

        if not args.small_only:
            ext = args.out_dir / "Datasets20"
            if ext.exists():
                shutil.rmtree(ext)
            ext.mkdir()
            designs = np.asarray(z["L20_200k"], dtype=np.float64)
            responses = legacy_response(designs, chunk_size=args.chunk_size)
            prefix = "SiO2_TiO2_20_layer_200K_200_points"
            write_legacy_csv(ext / f"{prefix}_DS.csv", designs)
            write_legacy_csv(ext / f"{prefix}_RS.csv", responses)
            print("completed L20_200k")
            zip_tree(ext, args.out_dir / "Datasets20_regenerated.zip", "Datasets20")


if __name__ == "__main__":
    main()
