#!/usr/bin/env python3
"""Reproduce the historical multilayer labels used in this paper.

This is a Python translation of ``multlayer_metamat_generation_old.m``.
It intentionally preserves the historical response definition and propagation
logic so that the published datasets can be regenerated from the archived
paper-design matrices.

Historical generator definition
--------------------------------
* frequency grid: 250--1050 THz, 200 uniformly spaced frequency points
* incidence: normal (theta = 0 deg)
* polarization flag: p = 1 (parallel/TM in the legacy code)
* alternating material parameters:
    epsilon_SiO2 = 1.46
    epsilon_TiO2 = 5.913
    mu = 1, sigma = 0
* thickness entries: 30--70 nm
* stored response: abs(t_legacy), the field-transmission magnitude
* legacy geometry: only the first N-1 thicknesses are propagated; the Nth
  material is used as the exit medium, so the final sampled thickness does
  not affect the stored response.

The script can either regenerate labels for the exact compressed design
matrices shipped with the repository or generate a new iid uniform design
ensemble with an explicit seed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

EPS0 = 8.85e-12
MU0 = 12.5664e-7
C0 = 1.0 / np.sqrt(EPS0 * MU0)
EPS_SIO2 = 1.46
EPS_TIO2 = 5.913
MU_LAYER = 1.0
SIGMA_LAYER = 0.0
FREQ_THz = np.linspace(250.0, 1050.0, 200, dtype=np.float64)


def legacy_response(designs_m: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Return historical |t| labels for designs shaped (N, samples).

    The implementation follows the archived MATLAB code, including the
    final-thickness omission. Output shape is (200, samples).
    """
    designs_m = np.asarray(designs_m, dtype=np.float64)
    if designs_m.ndim != 2:
        raise ValueError("designs_m must have shape (layers, samples)")
    n_layers, n_samples = designs_m.shape
    if n_layers < 2:
        raise ValueError("At least two material entries are required")

    eps = np.array(
        [EPS_SIO2 if i % 2 == 0 else EPS_TIO2 for i in range(n_layers)],
        dtype=np.float64,
    )
    mu = np.full(n_layers, MU_LAYER, dtype=np.float64)
    sigma = np.full(n_layers, SIGMA_LAYER, dtype=np.float64)

    # theta = 0 and p = 1 in the historical code.
    st = 0.0
    ct = 1.0
    # sigma = 0, so Er does not depend on frequency.
    er = eps.astype(np.complex128)
    z = np.sqrt(er * mu - st**2) / er / ct

    rho = np.empty(n_layers, dtype=np.complex128)
    tau = np.empty(n_layers, dtype=np.complex128)
    rho[0] = (z[0] - 1.0) / (z[0] + 1.0)
    tau[0] = rho[0] + 1.0
    for i in range(1, n_layers):
        rho[i] = (z[i] - z[i - 1]) / (z[i] + z[i - 1])
        tau[i] = rho[i] + 1.0

    responses = np.empty((FREQ_THz.size, n_samples), dtype=np.float64)

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        d = designs_m[:, start:stop]
        m = stop - start

        # Matrix elements of one 2x2 total matrix per design in the chunk.
        t00 = np.ones(m, dtype=np.complex128)
        t01 = np.zeros(m, dtype=np.complex128)
        t10 = np.zeros(m, dtype=np.complex128)
        t11 = np.ones(m, dtype=np.complex128)

        # Reinitialize the total matrix for each frequency, exactly as MATLAB.
        for k, f_thz in enumerate(FREQ_THz):
            t00.fill(1.0); t01.fill(0.0); t10.fill(0.0); t11.fill(1.0)
            lam = C0 / (f_thz * 1e12)
            beta1 = 2.0 * np.pi / lam

            # Historical code: for i = 1:(N-1). Final thickness is unused.
            for i in range(n_layers - 1):
                a = np.exp(1j * beta1 * d[i] * np.sqrt(er[i] - st**2))
                ac = np.exp(-1j * beta1 * d[i] * np.sqrt(er[i] - st**2))
                inv_tau = 1.0 / tau[i]
                b00 = inv_tau * a
                b01 = inv_tau * rho[i] * ac
                b10 = inv_tau * rho[i] * a
                b11 = inv_tau * ac

                n00 = t00 * b00 + t01 * b10
                n01 = t00 * b01 + t01 * b11
                n10 = t10 * b00 + t11 * b10
                n11 = t10 * b01 + t11 * b11
                t00, t01, t10, t11 = n00, n01, n10, n11

            # Historical final interface into the Nth material.
            inv_tau_n = 1.0 / tau[-1]
            b00 = inv_tau_n
            b01 = inv_tau_n * rho[-1]
            b10 = inv_tau_n * rho[-1]
            b11 = inv_tau_n
            n00 = t00 * b00 + t01 * b10
            # Only total(1,1) is needed for taot = 1/total(1,1).
            taot = 1.0 / n00
            responses[k, start:stop] = np.abs(taot)

    return responses


def load_designs(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path) as z:
        if key not in z:
            raise KeyError(f"{key!r} not found. Available keys: {sorted(z.files)}")
        return np.asarray(z[key], dtype=np.float64)


def generate_designs(layers: int, samples: int, seed: int) -> np.ndarray:
    """Generate a new reproducible iid uniform 30--70 nm ensemble.

    This produces a new ensemble. It is not claimed to reconstruct the
    historical MATLAB RNG state, which was not recorded in the source file.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(30e-9, 70e-9, size=(layers, samples)).astype(np.float64)


def write_legacy_csv(path: Path, array: np.ndarray) -> None:
    # MATLAB dlmwrite default precision is effectively five significant digits
    # for these historical files. This format reproduces their stored precision.
    np.savetxt(path, array, delimiter=",", fmt="%.5g")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--design-npz", type=Path,
                     help="Compressed exact design archive included with repository")
    src.add_argument("--new-designs", action="store_true",
                     help="Generate a new seeded iid uniform design ensemble")
    ap.add_argument("--key", help="NPZ key, e.g. L14_50k or L20_200k")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out-dir", type=Path, default=Path("generated_data"))
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()

    if args.design_npz:
        if not args.key:
            ap.error("--key is required with --design-npz")
        designs = load_designs(args.design_npz, args.key)
        layers, samples = designs.shape
        suffix = args.key
    else:
        if args.layers is None or args.samples is None:
            ap.error("--layers and --samples are required with --new-designs")
        designs = generate_designs(args.layers, args.samples, args.seed)
        layers, samples = designs.shape
        suffix = f"L{layers}_{samples}"

    responses = legacy_response(designs, chunk_size=args.chunk_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"SiO2_TiO2_{layers}_layer_{samples}_200_points"
    ds_path = args.out_dir / f"{prefix}_DS.csv"
    rs_path = args.out_dir / f"{prefix}_RS.csv"
    write_legacy_csv(ds_path, designs)
    write_legacy_csv(rs_path, responses)

    print(f"Wrote {ds_path} shape={designs.shape}")
    print(f"Wrote {rs_path} shape={responses.shape}")
    print("Historical label definition: |t_legacy| on 250--1050 THz (200 points)")
    print("Note: final sampled thickness is unused by the historical response calculation.")


if __name__ == "__main__":
    main()
