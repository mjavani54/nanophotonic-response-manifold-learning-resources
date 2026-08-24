# Reproducibility workflow

The final manuscript combines several experiments. Reproduce them in the following order.

## 1. Intrinsic dimensionality

Run the corrected intrinsic-dimensionality pipeline on the 50k multilayer datasets.

```bash
python code/paper1_intrinsic_dimensionality_pipeline_fixed.py \
  --zip "YOUR_50K_DATASET.zip" \
  --mode full
```

This analysis provides participation ratio, PCA dimensions, nonlinear AE reconstruction curves, and initial capacity/learning-curve diagnostics.

## 2. Targeted high-capacity diagnostic

Use the 50k datasets to test whether the larger structures are limited by model capacity.

```bash
python code/paper1_high_capacity_diagnostic.py \
  --zip "YOUR_50K_DATASET.zip" \
  --out paper1_capacity_diagnostic
```

## 3. 20-layer optimization diagnostic

Hold the best 20-layer architecture fixed and test whether optimizer/training choices resolve the 50k plateau.

```bash
python code/paper1_20layer_optimization_diagnostic.py \
  --zip "YOUR_50K_DATASET.zip" \
  --out paper1_20layer_optimization
```

## 4. Extended 20-layer 200k learning curve

```bash
python code/paper1_20layer_200k_learning_curve.py \
  --zip "YOUR_20L_200K_DATASET.zip" \
  --out paper1_20layer_200k_learning_curve
```

The primary extended training sizes are 10k, 20k, 30k, 50k, 75k, 100k, 150k, and 180k, with validation/test holdouts.

## 5. Controlled cross-structure learning curves

```bash
python code/paper1_cross_structure_learning_curves_fixed.py \
  --zip-small "YOUR_50K_DATASET.zip" \
  --zip-20 "YOUR_20L_200K_DATASET.zip" \
  --out paper1_cross_structure_learning_curves
```

This is the main controlled sample-complexity experiment. Fixed capacity-appropriate architectures are used for each layer count.

## 6. Low-sample refinement

```bash
python code/paper1_low_sample_refinement.py \
  --zip-small "YOUR_50K_DATASET.zip" \
  --out paper1_low_sample_refinement
```

This resolves the low-sample regime for 8–14 layers using 1k–5k training samples.

## 7. Final frozen outputs

The paper-level consolidated CSV files are provided under `results/`. They include:

- `final_master_summary.csv`
- `final_correlation_statistics.csv`
- `final_learning_curves.csv`
- `left_censoring_sensitivity.csv`
- `Table1_final_complexity_resource_summary.csv`

The figures in `figures/` were generated from these final outputs.

## Reproducibility conventions

- Three seeds are used in the final neural-learning experiments.
- Validation and test sets remain held out from model fitting.
- Standardization statistics are estimated from the current training subset only in the controlled learning-curve experiments.
- Training subsets are nested within seed-specific permutations.
- Resource thresholds are discrete-grid **minimum tested** values.
- 8L and 10L sample thresholds are left-censored at 1,000 samples.
- 20L fixed-50k capacity threshold is right-censored and excluded from the capacity-correlation fit.
