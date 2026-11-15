#!/usr/bin/env python3
"""Small numerical and provenance checks before an expensive rerun."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data_generation/generate_legacy_multilayer.py"
spec = importlib.util.spec_from_file_location("corrected_generator", SOURCE)
assert spec is not None and spec.loader is not None
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def main() -> None:
    if gen.EPS_SIO2 != 2.1316 or gen.EPS_TIO2 != 5.913:
        raise AssertionError("Unexpected material constants")
    if not np.isclose(np.sqrt(gen.EPS_SIO2), 1.46, atol=1e-14):
        raise AssertionError("epsilon=2.1316 does not imply n=1.46")
    original_grid = gen.FREQ_THz.copy()
    try:
        gen.FREQ_THz = np.array([350.0, 550.0, 750.0])
        rng = np.random.default_rng(87)
        designs = rng.uniform(30e-9, 70e-9, size=(20, 4))
        corrected = gen.legacy_response(designs, chunk_size=2)
        if corrected.shape != (3, 4) or not np.isfinite(corrected).all():
            raise AssertionError("Corrected response shape or finiteness failed")
        # Explicitly audit the retained legacy model limitation.
        modified = designs.copy()
        modified[-1] += 10e-9
        if not np.allclose(corrected, gen.legacy_response(modified, chunk_size=2), atol=1e-13):
            raise AssertionError("Unexpected change in legacy last-thickness convention")
        gen.EPS_SIO2 = 1.46
        old_parameter = gen.legacy_response(designs, chunk_size=2)
        gen.EPS_SIO2 = 2.1316
        difference = float(np.mean(np.abs(corrected - old_parameter)))
        if difference <= 1e-3:
            raise AssertionError("Changing epsilon failed to change the labels materially")
        # At zero propagated thickness and two media, the interfaces telescope
        # to the air/TiO2 field coefficient; this is independent of SiO2.
        zero = np.zeros((2, 1))
        actual = gen.legacy_response(zero, chunk_size=1)
        expected = 2.0 / (1.0 + np.sqrt(gen.EPS_TIO2))
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError("Zero-thickness interface calculation failed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "designs.csv"
            gen.write_legacy_csv(path, designs)
            if not np.allclose(np.loadtxt(path, delimiter=","), designs, atol=1e-22):
                raise AssertionError("Saved design precision is insufficient")
        print(json.dumps({"status": "PASS", "epsilon_r_SiO2": 2.1316,
                          "refractive_index_SiO2_constant": 1.46,
                          "mean_abs_change_vs_old_parameter": difference,
                          "legacy_last_thickness_inactive": True}, indent=2))
    finally:
        gen.EPS_SIO2 = 2.1316
        gen.FREQ_THz = original_grid


if __name__ == "__main__":
    main()
