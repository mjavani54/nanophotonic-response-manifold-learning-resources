#!/usr/bin/env python3
"""Generate NEW epsilon-corrected paper-format CSVs from exact stored designs.

Do not mix these newly calculated labels with historical 1.46 labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

from generate_legacy_multilayer import (EPS_SIO2, EPS_TIO2, FREQ_THz,
                                        legacy_response, write_legacy_csv)

LAYERS = [8, 10, 12, 14, 16, 18, 20]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite {path}; choose a fresh --out-dir")


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

    if EPS_SIO2 != 2.1316:
        raise RuntimeError("Wrong SiO2 permittivity; aborting corrected rerun")
    if not args.design_npz.is_file():
        raise FileNotFoundError(args.design_npz)
    small = args.out_dir / "Datasets"
    ext = args.out_dir / "Datasets20"
    small_zip = args.out_dir / "Datasets50k_eps21316.zip"
    ext_zip = args.out_dir / "Datasets20_eps21316.zip"
    for path in ([small, small_zip] if args.small_only else
                 [ext, ext_zip] if args.extended_only else
                 [small, ext, small_zip, ext_zip]):
        require_new(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "new epsilon-corrected rerun; NOT historical archived responses",
        "design_npz_sha256": sha256(args.design_npz),
        "generator_sha256": sha256(Path(__file__).with_name("generate_legacy_multilayer.py")),
        "epsilon_r_SiO2": EPS_SIO2, "epsilon_r_TiO2": EPS_TIO2,
        "frequency_grid_THz": [float(FREQ_THz[0]), float(FREQ_THz[-1]), len(FREQ_THz)],
        "geometry": "historical N-1 propagated thicknesses; Nth material exit medium",
        "response": "field transmission magnitude |t|, not power transmittance",
        "csv_format": "%.17g", "cohorts": {},
    }
    with np.load(args.design_npz) as z:
        if not args.extended_only:
            small.mkdir()
            for L in LAYERS:
                key = f"L{L}_50k"
                designs = np.asarray(z[key], dtype=np.float64)
                if designs.shape != (L, 50_000):
                    raise ValueError(f"Unexpected {key} shape: {designs.shape}")
                responses = legacy_response(designs, chunk_size=args.chunk_size)
                prefix = f"SiO2_TiO2_{L}_layer_50K_200_points"
                write_legacy_csv(small / f"{prefix}_DS.csv", designs)
                write_legacy_csv(small / f"{prefix}_RS.csv", responses)
                manifest["cohorts"][key] = {"designs": list(designs.shape),
                                           "responses": list(responses.shape)}
                print(f"completed {key}")
            zip_tree(small, small_zip, "Datasets")
            manifest["small_zip_sha256"] = sha256(small_zip)

        if not args.small_only:
            ext.mkdir()
            designs = np.asarray(z["L20_200k"], dtype=np.float64)
            if designs.shape != (20, 200_000):
                raise ValueError(f"Unexpected L20_200k shape: {designs.shape}")
            responses = legacy_response(designs, chunk_size=args.chunk_size)
            prefix = "SiO2_TiO2_20_layer_200K_200_points"
            write_legacy_csv(ext / f"{prefix}_DS.csv", designs)
            write_legacy_csv(ext / f"{prefix}_RS.csv", responses)
            manifest["cohorts"]["L20_200k"] = {"designs": list(designs.shape),
                                                "responses": list(responses.shape)}
            print("completed L20_200k")
            zip_tree(ext, ext_zip, "Datasets20")
            manifest["extended_zip_sha256"] = sha256(ext_zip)
    (args.out_dir / "material_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
