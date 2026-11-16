#!/usr/bin/env python3
"""Recreate byte-verified augmented data from compact source designs."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    if len(sys.argv)!=3:raise SystemExit("Usage: python scripts/reproduce_augmentation.py ASSET_DIR NEW_OUTPUT_DIR")
    assets=Path(sys.argv[1]);out=Path(sys.argv[2])
    if out.exists():raise SystemExit("Choose a fresh output directory")
    out.mkdir(parents=True)
    module=ROOT/"source/data_generation/generate_legacy_multilayer.py"
    spec=importlib.util.spec_from_file_location("generator",module)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    recorded=json.loads((ROOT/"records/extension_data/generation_manifest.json").read_text())
    with np.load(assets/"source_designs/augmented_source_designs.npz") as z:
        for layer in (14,16,18):
            designs=z[f"L{layer}_160k"]
            response=gen.legacy_response(designs.T,chunk_size=2048).T.astype(np.float32)
            file=out/f"extra_L{layer}_160000.npz"
            # The recorded file serializes designs in Fortran order; preserve
            # array layout as well as numerical values for exact NPZ identity.
            np.savez_compressed(file,designs_m=np.asfortranarray(designs),response_abs_t=response)
            actual=sha256(file.read_bytes()).hexdigest()
            expected=recorded["cases"][str(layer)]["file_sha256"]
            if actual!=expected:raise ValueError(f"L{layer} differs: {actual} vs {expected}")
            print(f"L{layer}: exact recorded SHA-256 {actual}")
if __name__=="__main__":main()
