# Response Complexity Tracks Learning Resources in Multilayer Nanophotonic Surrogate Models

Reproducibility repository for the ACS Photonics manuscript:

> **Response Complexity Tracks Learning Resources in Multilayer Nanophotonic Surrogate Models**

## Scientific question

How do operational measures of response complexity relate to the tested training-set sizes and neural-network capacities needed to learn a multilayer forward model?

This study compares response descriptors with learning-resource thresholds within one controlled family of alternating SiO2/TiO2 multilayers with nominal design-vector lengths of 8–20. Layer count and response complexity vary together, so the correlations do not establish independent prediction beyond layer count or transfer to unrelated structures. The autoencoder bottleneck is an operational reconstruction threshold, not a mathematical intrinsic dimension.

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
├── data_generation/      Archived MATLAB reference + Python reconstruction
├── data/
│   ├── exact_designs/    Compressed design matrices used by the paper
│   └── reference_fixture/Small label fixtures for generator verification
├── docs/                 Reproducibility and provenance documentation
├── figures/              Final figures and ACS TOC graphic
├── results/              Frozen CSV tables used for final statistics
├── scripts/              Repository verification helper
├── supplementary/        Supporting-information notes
├── CHECKSUMS.sha256
├── CITATION.cff
├── LICENSE
├── environment.yml
└── requirements.txt
```

## Response convention and source provenance

The author reports that the analyzed datasets were generated using Python. The repository contains an older MATLAB reference file, `data_generation/multlayer_metamat_generation_old.m`, and a later Python reconstruction of the documented numerical convention. The specific original Python program executed to generate the archived response CSVs has not been identified. Agreement with selected response fixtures supports the documented label convention but does not establish source-code identity or full-archive equality.

The reconstructed historical label convention uses:

- 200 **uniform-frequency** samples from **250 to 1050 THz**;
- normal incidence (`theta = 0°`);
- the legacy parallel-polarization flag (`p = 1`);
- alternating material parameters `epsilon_SiO2 = 1.46` and `epsilon_TiO2 = 5.913`, with `mu = 1` and `sigma = 0`;
- sampled thickness entries from **30 to 70 nm**; and
- stored response `abs(t_legacy)`, the magnitude of the historical field-transmission coefficient.

The reconstructed implementation propagates the first `N-1` thicknesses and then applies an interface into the `N`th material. Consequently, the final sampled thickness is present in the design vector but does not affect the response under this documented convention. This behavior is preserved in the reproduction code.

This is **not** a corrected finite-stack power-transmittance solver. The repository documents the historical label definition rather than silently replacing it with a different physical model.

## Exact design matrices and data regeneration

`data/exact_designs/paper1_exact_designs.npz` contains the design matrices used in the reported analyses:

- 8, 10, 12, 14, 16, 18, and 20 entries × 50,000 designs; and
- the separate 20-entry × 200,000-design ensemble used for the extended learning curve.

The archived design CSVs were stored at limited decimal precision. Recomputed responses can therefore differ from the exact archived response CSVs. `data_generation/validate_legacy_generator.py` checks selected archived label fixtures; it does not verify every archived response or establish identity with the original data-generation source.

Example:

```bash
python data_generation/validate_legacy_generator.py

python data_generation/generate_legacy_multilayer.py \
  --design-npz data/exact_designs/paper1_exact_designs.npz \
  --key L14_50k \
  --out-dir generated/L14
```

A new statistically comparable ensemble can also be generated with an explicit seed:

```bash
python data_generation/generate_legacy_multilayer.py \
  --new-designs \
  --layers 14 \
  --samples 50000 \
  --seed 20260918 \
  --out-dir generated/L14_new
```

The new-design option does not reconstruct the random-number-generator state used for the archived datasets.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is strongly recommended for the neural-network sweeps.

## Reproducing the analysis

For exact numerical reproduction, download the archived design and response CSVs from [release v1.1.0](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources/releases/tag/v1.1.0) and run the analysis scripts in the order documented in `docs/REPRODUCIBILITY.md`. Generated responses from the stored rounded designs are suitable for checking the documented convention, subject to the numerical differences described above.

The core scripts are:

1. `code/paper1_intrinsic_dimensionality_pipeline_fixed.py`
2. `code/paper1_high_capacity_diagnostic.py`
3. `code/paper1_20layer_optimization_diagnostic.py`
4. `code/paper1_20layer_200k_learning_curve.py`
5. `code/paper1_cross_structure_learning_curves_fixed.py`
6. `code/paper1_low_sample_refinement.py`

Frozen final outputs are included under `results/` so readers can inspect the reported statistics without rerunning the full GPU workflow.

## Public data release

The exact archived design and response CSVs used for the reported analyses are public as assets of [release v1.1.0](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources/releases/tag/v1.1.0):

| Asset | Contents | SHA-256 |
|---|---|---|
| `Paper1_50k_RawData.zip` | Exact 8–20 layer, 50,000-sample design and response CSVs used in the primary analyses | `8a4406d8d470d8c56f424baf835097606d2742eea10e6a1ba7b7ef15671986cb` |
| `Paper1_20L_200k_RawData.zip` | Exact 20-layer, 200,000-sample design and response CSVs used in the extended learning curve | `6aeef791606c60706d4f0f88b72e5a925177c7f4e7eac3039fe2abd550f65a01` |

Both public downloads were checked against the listed SHA-256 values. The large archives are distributed as release assets rather than committed to ordinary Git history.

## Data availability

The repository contains the stored design matrices, Python response reconstruction, analysis scripts, frozen summary outputs, figures, and selected validation fixtures. Use the v1.1.0 release assets for exact numerical reproduction of the reported analyses.

## Important interpretation note

The paper's numerical learning results apply to the **historical label definition described above**. A statement describing 350–750 nm power transmittance of a fully finite N-layer stack would not match the analyzed datasets. See `docs/DATA_GENERATION_PROVENANCE.md`.

## Citation

See `CITATION.cff` and cite release v1.1.0. Add an archive DOI if one is created later.

## License

Code in this repository is released under the MIT License. Publication figures, manuscript text, and third-party material may be subject to separate publisher or copyright terms after publication.
