# Results provenance

This repository separates exploratory/diagnostic analyses from the frozen results used in the manuscript.

## Manuscript-level results

The files in `results/` are the final consolidated outputs used for the manuscript figures and numerical statements.

### Primary sample-complexity thresholds at mean test R² ≥ 0.99

| Layers | Minimum tested training samples |
|---:|---:|
| 8 | ≤1,000 |
| 10 | ≤1,000 |
| 12 | 2,000 |
| 14 | 4,000 |
| 16 | 10,000 |
| 18 | 15,000 |
| 20 | 30,000 |

### Nonlinear AE dimensions at reconstruction R² ≥ 0.99

5, 6, 8, 10, 14, 20, and 30 for 8–20 layers, respectively.

### Resolved model-capacity thresholds

The 8–18 layer systems have resolved fixed-50k capacity thresholds; the 20-layer point remains right-censored and is excluded from the capacity-correlation statistic.

## Why diagnostic scripts are retained

The high-capacity and 20-layer optimization scripts document how alternative explanations (under-capacity and training/optimization limitation) were tested before the final controlled learning-curve analysis.
