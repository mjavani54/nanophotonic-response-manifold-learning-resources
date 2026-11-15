#!/usr/bin/env python3
"""Reject old 1.46 response archives before a corrected analysis starts."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--small", type=Path)
    p.add_argument("--extended", type=Path)
    args = p.parse_args()
    if not args.small and not args.extended:
        p.error("Pass --small and/or --extended")
    info = json.loads(args.manifest.read_text())
    if info.get("epsilon_r_SiO2") != 2.1316 or info.get("epsilon_r_TiO2") != 5.913:
        raise ValueError("Manifest does not describe the corrected material parameters")
    for name, path, key, layers in (
        ("small", args.small, "small_zip_sha256", [8, 10, 12, 14, 16, 18, 20]),
        ("extended", args.extended, "extended_zip_sha256", [20]),
    ):
        if path is None:
            continue
        if sha256(path) != info.get(key):
            raise ValueError(f"{name} archive does not match the corrected manifest: {path}")
        with ZipFile(path) as z:
            names = set(z.namelist())
        for layer in layers:
            size = "50K" if name == "small" else "200K"
            root = "Datasets" if name == "small" else "Datasets20"
            for kind in ("DS", "RS"):
                member = f"{root}/SiO2_TiO2_{layer}_layer_{size}_200_points_{kind}.csv"
                if member not in names:
                    raise ValueError(f"Missing {member} in {path}")
        print(f"PASS {name}: epsilon=2.1316, SHA-256={info[key]}, expected CSV names present")


if __name__ == "__main__":
    main()
