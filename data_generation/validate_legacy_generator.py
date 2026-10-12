#!/usr/bin/env python3
"""Validate the Python translation against archived paper-label fixtures."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from generate_legacy_multilayer import legacy_response

fixture_path = ROOT / "data" / "reference_fixture" / "legacy_generator_reference.npz"
with np.load(fixture_path) as z:
    tests = [
        ("L8", z["L8_designs"], z["L8_responses"]),
        ("L14", z["L14_designs"], z["L14_responses"]),
        ("L20_200k", z["L20_200k_designs"], z["L20_200k_responses"]),
    ]

worst = 0.0
for name, designs, expected in tests:
    predicted = legacy_response(designs, chunk_size=16)
    err = float(np.max(np.abs(predicted - expected)))
    rmse = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    worst = max(worst, err)
    print(f"{name}: max_abs_error={err:.3e}, rmse={rmse:.3e}")

# Stored historical CSVs were rounded to about five significant digits, while
# the public exact-design archive contains those stored (rounded) thicknesses.
# A 4e-4 tolerance captures the worst observed round-trip difference in the archived fixture, which occurs near a spectrally sensitive point in the 20-layer set.
if worst > 4e-4:
    raise SystemExit(f"FAILED: worst error {worst:.3e} exceeds 4e-4")
print("PASS: Python reconstruction agrees with selected archived fixtures within tolerance; full-archive identity and original source provenance are unverified.")
