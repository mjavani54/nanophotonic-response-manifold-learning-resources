# Intrinsic Response-Manifold Dimensionality Predicts Learning Resources for Nanophotonic Surrogate Models

Reproducibility repository for the ACS Photonics manuscript:

> **Intrinsic Response-Manifold Dimensionality Predicts Learning Resources for Nanophotonic Surrogate Models**

## Scientific question

How much training data and neural-network capacity are needed to learn a nanophotonic forward model, and can those resource requirements be anticipated before full-scale training?

This study analyzes the empirical chain

**structural complexity → response-manifold dimensionality → learning-resource demand**

for alternating SiO2/TiO2 multilayer datasets with nominal design-vector lengths of 8–20.

## Main findings

The frozen analysis reports:

- PCA dimension at 99% explained variance: **9 → 53**.
- Nonlinear autoencoder latent dimension at reconstruction R² ≥ 0.99: **5 → 30**.
- Minimum tested training-set size for mean test R² ≥ 0.99: **≤1,000 → 30,000** samples.
- AE dimension vs. minimum tested training-set threshold: **Pearson r = 0.992, p = 1.1 × 10⁻5**.
- AE dimension vs. resolved minimum tested model-capacity threshold: **Pearson r = 0.955, p = 0.00295**.
- For the extended 20-layer study, the first tested sizes crossing R² = 0.99, 0.995, and 0.999 are **30k, 50k, and 150k**, respectively.

The wording **minimum tested** is intentional. The resource thresholds were evaluated on discrete grids.

## Repository structure

```text
.
├── code/                 Neural-network and dimensionality analyses
├── data_generation/      Historical MATLAB generator + validated Python translation
├── data/
│   ├── exact_designs/    Compressed design matrices used by the paper
│   └── reference_fixture/Small label fixtures for generator verification
├── docs/                 Reproducibility and provenance documentation
├── figures/              Final figures and ACS TOC graphic
├── manuscript/           Latest manuscript version
├── results/              Frozen CSV tables used for final statistics
├── scripts/              Repository verification helper
├── supplementary/        Supporting-information notes
├── CITATION.cff
├── LICENSE
├── environment.yml
└── requirements.txt
```

## Historical dataset generator used by the analyzed data

The repository now includes the original MATLAB source, `data_generation/multlayer_metamat_generation_old.m`, together with a Python translation that reproduces its stored response labels.

The analyzed historical labels were generated with:

- 200 **uniform frequency** samples from **250 to 1050 THz**;
- normal incidence (`theta = 0°`);
- the legacy parallel-polarization flag (`p = 1`);
- alternating material parameters `epsilon_SiO2 = 1.46` and `epsilon_TiO2 = 5.913`, with `mu = 1` and `sigma = 0`;
- sampled thickness entries from **30 to 70 nm**;
- stored response `abs(t_legacy)`, i.e. the magnitude of the historical field-transmission coefficient.

The historical implementation propagates the first `N-1` thicknesses and then applies an interface into the `N`th material. Consequently, the final sampled thickness is present in the design vector but does not affect the stored historical response. This behavior is preserved deliberately in the reproduction code because it is the label definition analyzed in the paper.

This is **not** a corrected finite-stack power-transmittance solver. The repository documents the historical label definition rather than silently replacing it with a different physical model.

## Exact design matrices and data regeneration

`data/exact_designs/paper1_exact_designs.npz` contains the design matrices used in the reported analyses:

- 8, 10, 12, 14, 16, 18, and 20 entries × 50,000 designs;
- the separate 20-entry × 200,000-design ensemble used for the extended learning curve.

The archived design CSVs were historically stored at limited decimal precision. The Python translation therefore reproduces the archived response CSVs to the precision expected from those stored design values; `data_generation/validate_legacy_generator.py` verifies this behavior against selected archived label fixtures.

Example:

```bash
python data_generation/validate_legacy_generator.py

python data_generation/generate_legacy_multilayer.py   --design-npz data/exact_designs/paper1_exact_designs.npz   --key L14_50k   --out-dir generated/L14
```

A new statistically equivalent ensemble can also be generated with an explicit seed:

```bash
python data_generation/generate_legacy_multilayer.py   --new-designs --layers 14 --samples 50000 --seed 20260918   --out-dir generated/L14_new
```

The new-design option does not claim to reconstruct the unrecorded historical MATLAB random-number-generator state.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is strongly recommended for the neural-network sweeps.

## Reproducing the analysis

Generate the historical response labels from the included design archive, or use already generated CSVs locally, and then run the analysis scripts in the order documented in `docs/REPRODUCIBILITY.md`.

The core scripts are:

1. `code/paper1_intrinsic_dimensionality_pipeline_fixed.py`
2. `code/paper1_high_capacity_diagnostic.py`
3. `code/paper1_20layer_optimization_diagnostic.py`
4. `code/paper1_20layer_200k_learning_curve.py`
5. `code/paper1_cross_structure_learning_curves_fixed.py`
6. `code/paper1_low_sample_refinement.py`

Frozen final outputs are included under `results/` so readers can inspect the reported statistics without rerunning the full GPU workflow.

## Data availability

A separate raw-data DOI is not required for this repository version because the exact stored design matrices used in the study are included directly in the repository and the response-label generator is provided. The repository also contains frozen final analysis outputs and validation fixtures.

For archival transparency, users should cite the tagged GitHub release or its software archive DOI if one is created.

## Important interpretation note

The paper's numerical learning results are results for the **historical label definition described above**. Any manuscript statement that instead describes 350–750 nm power transmittance of a fully finite N-layer stack would not match the analyzed datasets. See `docs/DATA_GENERATION_PROVENANCE.md`.

## Citation

See `CITATION.cff`. Replace the repository URL/DOI fields after the public release is created.

## License

Code in this repository is released under the MIT License. Publication figures, manuscript text, and third-party material may be subject to separate publisher/copyright terms after publication.
