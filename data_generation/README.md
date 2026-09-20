# Data generation

This directory contains the generator that produced the label definition analyzed in the manuscript.

## Files

- `multlayer_metamat_generation_old.m` — archived historical MATLAB source, included without changing its numerical definition.
- `generate_legacy_multilayer.py` — Python translation of the historical response calculation.
- `validate_legacy_generator.py` — checks the Python translation against stored response fixtures from the analyzed datasets.

## Historical numerical definition

The source uses 200 uniformly spaced **frequency** points from 250 to 1050 THz. It alternates `epsilon=1.46` and `epsilon=5.913`, with `mu=1`, zero conductivity, normal incidence, and the parallel-polarization flag. It stores `abs(taot)`.

The MATLAB propagation loop is `for i=1:(N-1)`. The final sampled thickness is therefore not propagated; the Nth material is used as the exit medium. This behavior is retained here because changing it would generate a different dataset.

## Reproducing paper labels

Use the included stored design matrices:

```bash
python data_generation/generate_legacy_multilayer.py   --design-npz data/exact_designs/paper1_exact_designs.npz   --key L20_200k   --out-dir generated/L20_200k
```

Available keys are `L8_50k`, `L10_50k`, `L12_50k`, `L14_50k`, `L16_50k`, `L18_50k`, `L20_50k`, and `L20_200k`.

## Precision note

The historical design CSVs were stored with limited decimal precision. The included design archive preserves the values that were actually available to the downstream analysis. Recomputing labels from those rounded designs can differ very slightly from archived response CSV values near highly sensitive spectral features. The validation script quantifies this round-trip effect.
