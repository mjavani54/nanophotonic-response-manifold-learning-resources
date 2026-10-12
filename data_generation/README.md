# Data generation

This directory contains an older MATLAB reference and a later Python reconstruction of the response convention analyzed in the manuscript. The author reports that the original data were generated with Python; that specific Python source has not yet been identified in the public repository.

## Files

- `multlayer_metamat_generation_old.m` — archived MATLAB reference, not established as the executed data-generation program.
- `generate_legacy_multilayer.py` — Python reconstruction of the documented response calculation.
- `validate_legacy_generator.py` — checks the Python reconstruction against selected response fixtures from the analyzed datasets.

## Historical numerical definition

The reconstructed convention uses 200 uniformly spaced **frequency** points from 250 to 1050 THz. It alternates `epsilon=1.46` and `epsilon=5.913`, with `mu=1`, zero conductivity, normal incidence, and the parallel-polarization flag. It stores `abs(taot)`.

The MATLAB propagation loop is `for i=1:(N-1)`. The final sampled thickness is therefore not propagated; the Nth material is used as the exit medium. This behavior is retained here because changing it would generate a different dataset.

## Reproducing paper labels

Use the included stored design matrices:

```bash
python data_generation/generate_legacy_multilayer.py   --design-npz data/exact_designs/paper1_exact_designs.npz   --key L20_200k   --out-dir generated/L20_200k
```

Available keys are `L8_50k`, `L10_50k`, `L12_50k`, `L14_50k`, `L16_50k`, `L18_50k`, `L20_50k`, and `L20_200k`.

## Precision note

The historical design CSVs were stored with limited decimal precision. The included design archive preserves the values that were actually available to the downstream analysis. Recomputing labels from those rounded designs can differ from archived response CSV values near highly sensitive spectral features. The validation script quantifies the error for selected fixtures only. The exact archived response CSVs used in the reported analyses are not yet public.
