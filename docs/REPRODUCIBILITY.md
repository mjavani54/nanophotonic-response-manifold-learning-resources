# Reproducibility workflow

## 0. Verify the dataset generator

```bash
python data_generation/validate_legacy_generator.py
```

The original MATLAB source and a Python translation are included under `data_generation/`.

## 1. Rebuild the historical response labels

The exact stored design matrices used in the study are included under `data/exact_designs/`.

Example:

```bash
python data_generation/generate_legacy_multilayer.py   --design-npz data/exact_designs/paper1_exact_designs.npz   --key L8_50k   --out-dir generated/L8
```

Repeat with the keys listed in `data/README.md`, or use `data_generation/build_paper_datasets.py` to construct the full 50k and extended 20-layer archives.

## 2. Intrinsic dimensionality

```bash
python code/paper1_intrinsic_dimensionality_pipeline_fixed.py   --zip generated/Datasets50k_regenerated.zip   --mode full
```

## 3. Targeted high-capacity diagnostic

```bash
python code/paper1_high_capacity_diagnostic.py   --zip generated/Datasets50k_regenerated.zip   --out paper1_capacity_diagnostic
```

## 4. 20-layer optimization diagnostic

```bash
python code/paper1_20layer_optimization_diagnostic.py   --zip generated/Datasets50k_regenerated.zip   --out paper1_20layer_optimization
```

## 5. Extended 20-layer 200k learning curve

```bash
python code/paper1_20layer_200k_learning_curve.py   --zip generated/Datasets20_regenerated.zip   --out paper1_20layer_200k_learning_curve
```

## 6. Controlled cross-structure learning curves

```bash
python code/paper1_cross_structure_learning_curves_fixed.py   --zip-small generated/Datasets50k_regenerated.zip   --zip-20 generated/Datasets20_regenerated.zip   --out paper1_cross_structure_learning_curves
```

## 7. Low-sample refinement

```bash
python code/paper1_low_sample_refinement.py   --zip-small generated/Datasets50k_regenerated.zip   --out paper1_low_sample_refinement
```

## 8. Frozen final outputs

The paper-level consolidated CSV files are under `results/`. These are the exact frozen statistics reported in the manuscript.

## Reproducibility conventions

- Three training seeds are used in the final neural-learning experiments.
- Validation and test sets remain held out from model fitting.
- Standardization statistics are estimated from the current training subset only in the controlled learning-curve experiments.
- Training subsets are nested within seed-specific permutations.
- Resource thresholds are discrete-grid **minimum tested** values.
- 8L and 10L sample thresholds are left-censored at 1,000 samples.
- 20L fixed-50k capacity threshold is right-censored and excluded from the capacity-correlation fit.
