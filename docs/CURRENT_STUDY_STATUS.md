# Current-study deposit status

The files on `main` are working resources for *Response Compressibility and Accuracy-Dependent Learning Requirements for Multilayer Optical Surrogates*, a Journal of Modern Optics candidate. The earlier paper's exact inputs and outputs remain citable as [release v1.1.0](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources/releases/tag/v1.1.0). These are distinct numerical response calculations; do not mix their labels or summaries.

| Resource | Location on `main` | Status for the candidate study |
|---|---|---|
| Primary stored design vectors | `data/exact_designs/paper1_exact_designs.npz` | Included; generator checks its SHA-256 against the material-model manifest. |
| Source and analysis code | `studies/jmo_2026_candidate/source/`, `tools/` | Included source snapshot and matched low-sample command; scripts alone do not establish raw-input availability. |
| Summary tables, selected per-seed scores, metadata | `studies/jmo_2026_candidate/records/` | Included; `verify_summary.py` checks selected numerical consistency and source hashes. |
| Recorded fits | `records/run_records_review.zip`, `records/matched_low_sample_records.zip` | Included as fit records; the old batch-512 small-sample component in the review ZIP is not used for the candidate paper tables. |
| Candidate figures | `studies/jmo_2026_candidate/figures/` | Included; root `figures/` contains archived earlier-study figures. |
| Exact 50,000-pair and separate 200,000-pair response archives | Expected hashes in `records/data/material_model_manifest.json` | **Absent.** The v1.1.0 response archives are from a different numerical definition. |
| Additional 14-, 16-, and 18-layer matrices and source designs | Expected provenance in `records/extension_data/generation_manifest.json` | **Absent.** |
| A-ensemble source designs and the crossed spectral-window response arrays | Source hashes and experiment metadata in `records/spectral/` and the [candidate README](../studies/jmo_2026_candidate/README.md) | **Absent.** The 48 AE fit results are present, but the four response matrices are not. |

The candidate's numerical definition uses constant εᵣ(SiO₂) = 2.1316 and εᵣ(TiO₂) = 5.913, 200 sampled frequencies, and field-transmission magnitude `|t|`. It propagates only N−1 of N stored thicknesses and takes the Nth material as the exit medium. The paper's inference is limited to that idealized model; it is not finite-stack power transmittance into air. See the [candidate-study README](../studies/jmo_2026_candidate/README.md) for the exact grids and protocol.

From a clean checkout, use `python studies/jmo_2026_candidate/verify_summary.py` to check available summaries and known input hashes. Use `python scripts/verify_repository.py` to check repository checksums and selected archived fixtures (requires NumPy). On macOS, `shasum -a 256 -c CHECKSUMS.sha256` checks the file manifest without installing `sha256sum`.

**Publication gate:** Deposit and independently verify the missing inputs and full output provenance under a distinct public version. Until then, do not state that this repository completely reproduces the candidate article or replace the manuscript's bracketed data-availability fields with a claim of complete public availability. The manuscript and Supporting Information in the author's separate working package are not submission-ready simply because this repository snapshot is uploadable.
