# Nanophotonic response manifold learning resources

This repository contains two **distinct numerical studies** of multilayer optical surrogates. The archived [v1.1.0 release](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources/releases/tag/v1.1.0) remains an exact record of its earlier data and results. The [Journal of Modern Optics candidate](studies/jmo_2026_candidate/README.md) is a separate working analysis. It is **not yet a complete public reproduction deposit** or a published version of the paper.

| Scope | Tagged v1.1.0 archive | Journal of Modern Optics candidate |
|---|---|---|
| SiO₂ relative permittivity in response calculation | 1.46 | 2.1316 |
| TiO₂ relative permittivity | 5.913 | 5.913 |
| Shared source design matrices | `data/exact_designs/paper1_exact_designs.npz` | The same primary design vectors; additional design-source cohorts are identified in the new metadata |
| Numerical results | Root `results/` and `figures/`; exact response CSVs are v1.1.0 release assets | `studies/jmo_2026_candidate/records/` and its separate figures |
| Exact complete study response arrays | Archived arrays are public in v1.1.0 | **Not deposited here yet** |

Do not combine tables, response arrays, or code paths from the two columns. In particular, the old `results/` CSVs, root `figures/`, root `code/`, and `data_generation/generate_legacy_multilayer.py` describe the archived model and **do not reproduce the candidate paper's numerical values**. The folder `studies/jmo_2026_candidate/source/` contains the code snapshot used for the candidate analysis and documents the distinct material parameter.

The candidate paper studies seven nominal layer counts (8–20) and reports first **tested** resource thresholds under specified architectures and R² criteria. Its initial autoencoder bottleneck at mean held-out R² ≥ 0.99 runs from 6 to 30; first primary-grid training sizes at mean forward test R² ≥ 0.99 range from ≤1,000 to 25,000. Its 20-layer capacity threshold on the fixed 50,000-design cohort is unresolved through 5,202,120 parameters. Two source-design ensembles test 350–750 versus 250–1050 THz spectral windows. Layer count and compression descriptors have identical class ranks, so the associations do not establish an independent predictive effect of compression. These are findings for the implemented model, not a universal physical sample-complexity law.

Both calculations use a 200-point frequency grid at normal incidence, constant real material parameters, and the **field-transmission magnitude** `|t|` as the label. The last sampled thickness is inactive because the implemented transfer calculation propagates N−1 thicknesses and uses material N as the exit medium. Neither model is a closed finite stack into air or flux-normalized power transmittance. Full methods and remaining data gaps are in the [candidate-study README](studies/jmo_2026_candidate/README.md).

## Integrity and status

`CHECKSUMS.sha256` covers the tracked checkout and the added candidate-study files (except the manifest itself). From the repository root:

```bash
sha256sum -c CHECKSUMS.sha256
python scripts/verify_repository.py
```

The verifier checks file integrity, selected v1.1.0 fixtures, and candidate-study **summary consistency**. It cannot certify full regeneration of the current analyses without the missing response matrices and source-design archives.

## Citing the work

Cite the tagged [v1.1.0 release](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources/releases/tag/v1.1.0) for archived inputs and frozen outputs. The candidate paper and its complete present-study deposit need their own verified public identifiers before a data-availability statement can identify them as complete. `CITATION.cff` currently points to the archived release.

Code is under the MIT license. The manuscript and Supporting Information are separate working documents and are not presented here as accepted or published articles.
