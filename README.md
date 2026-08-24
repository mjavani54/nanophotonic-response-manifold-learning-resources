# Intrinsic Response-Manifold Dimensionality Predicts Learning Resources for Nanophotonic Surrogate Models

Reproducibility repository for the ACS Photonics manuscript:

> **Intrinsic Response-Manifold Dimensionality Predicts Learning Resources for Nanophotonic Surrogate Models**

## Scientific question

How much training data and neural-network capacity are needed to learn a nanophotonic forward model, and can those resource requirements be anticipated before full-scale training?

This study tests the empirical chain

**structural complexity → response-manifold dimensionality → learning-resource demand**

for alternating SiO2/TiO2 multilayer structures with 8–20 variable layers.

## Main findings

The final analysis reports:

- PCA dimension at 99% explained variance increases from **9 to 53** from 8 to 20 layers.
- Nonlinear autoencoder latent dimension at reconstruction R² ≥ 0.99 increases from **5 to 30**.
- Minimum tested training-set size for mean test R² ≥ 0.99 increases from **≤1,000 to 30,000** samples.
- AE dimension vs. minimum tested training-set threshold: **Pearson r = 0.992, p = 1.1 × 10⁻5**.
- AE dimension vs. resolved minimum tested model-capacity threshold: **Pearson r = 0.955, p = 0.00295**.
- For the extended 20-layer study, the first tested sizes crossing R² = 0.99, 0.995, and 0.999 are **30k, 50k, and 150k**, respectively.

The wording **minimum tested** is intentional. Resource thresholds were evaluated on discrete grids. The 8- and 10-layer sample thresholds are left-censored at 1,000 samples, and the 20-layer fixed-50k capacity threshold is right-censored.

## Repository structure

```text
.
├── code/              Analysis and diagnostic scripts
├── data/              Data manifest and external-data instructions
├── docs/              Reproducibility, provenance, and release notes
├── figures/           Final manuscript figures and ACS TOC graphic
├── manuscript/        Latest human-revised ACS Photonics manuscript
├── results/           Frozen CSV tables used for final statistics
├── scripts/           Repository verification helper
├── supplementary/     Supporting-information notes
├── CITATION.cff
├── LICENSE
├── environment.yml
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

### Conda

```bash
conda env create -f environment.yml
conda activate nanophotonic-manifold-paper
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is strongly recommended for the neural-network sweeps.

## Reproducing the study

Place the raw data under `data/raw/` following `data/dataset_manifest.csv` and `data/README.md`. Then run the analyses in the order documented in `docs/REPRODUCIBILITY.md`.

The core scripts are:

1. `code/paper1_intrinsic_dimensionality_pipeline_fixed.py`
2. `code/paper1_high_capacity_diagnostic.py`
3. `code/paper1_20layer_optimization_diagnostic.py`
4. `code/paper1_20layer_200k_learning_curve.py`
5. `code/paper1_cross_structure_learning_curves_fixed.py`
6. `code/paper1_low_sample_refinement.py`

Frozen final outputs are in `results/` so readers can inspect the reported statistics without rerunning all GPU experiments.

## Data availability

The raw datasets are not committed here because they are too large for a normal GitHub repository. Deposit them in a DOI-bearing archive such as Zenodo and replace the placeholders in this README and `CITATION.cff` with the permanent record.

**Data DOI:** `REPLACE_WITH_DATA_DOI`

## Important methods note before public release

The thesis documents alternating TiO2/SiO2 multilayers, independent layer thicknesses in the 30–70 nm range, transmission responses, and a custom transfer-matrix calculation over 300–750 nm. The current manuscript/repository must be reconciled against the actual dataset-generation code before public release if a different spectral range or convention is stated. See `docs/METHODS_SOURCE_NOTE.md` and `docs/OPEN_ITEMS_BEFORE_PUBLIC_RELEASE.md`.

## Citation

Please use `CITATION.cff` after replacing the repository and DOI placeholders.

## License

Code in this repository is released under the MIT License. Publication figures, manuscript text, and third-party material may be subject to separate publisher/copyright terms after publication.
