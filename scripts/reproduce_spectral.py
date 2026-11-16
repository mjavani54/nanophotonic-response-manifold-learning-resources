#!/usr/bin/env python3
"""Recreate four spectral response arrays from the two design ensembles."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    if len(sys.argv)!=3:raise SystemExit("Usage: python scripts/reproduce_spectral.py ASSET_DIR NEW_OUTPUT_DIR")
    assets=Path(sys.argv[1]);out=Path(sys.argv[2])
    if out.exists():raise SystemExit("Choose a fresh output directory")
    out.mkdir(parents=True)
    module=ROOT/"source/data_generation/generate_legacy_multilayer.py"
    spec=importlib.util.spec_from_file_location("generator",module)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    a=np.load(assets/"source_designs/ensemble_A_designs.npy").T
    with np.load(assets/"source_designs/paper1_exact_designs.npz") as z:b=z["L20_50k"]
    expected=json.loads((assets/"spectral/reconstruction_manifest.json").read_text())
    for name,designs,bounds in (
        ("A_narrow",a,(350,750)),("A_wide",a,(250,1050)),
        ("B_narrow",b,(350,750)),("B_wide",b,(250,1050))):
        gen.FREQ_THz=np.linspace(bounds[0],bounds[1],200,dtype=np.float64)
        y=gen.legacy_response(designs,chunk_size=2048).T.astype(np.float32)
        file=out/f"{name}.npz"
        np.savez_compressed(file,response_abs_t=y)
        actual=sha256(file.read_bytes()).hexdigest()
        if actual!=expected["cases"][name]["npz_sha256"]:
            raise ValueError(f"{name}: reconstructed asset differs")
        print(f"{name}: reconstructed asset SHA-256 {actual}")
if __name__=="__main__":main()
