# Open items before public release/submission

Resolve these items before making the repository public or citing it in the ACS Photonics manuscript.

1. **Verify the exact multilayer simulation convention from the dataset-generation code.**
   - stored response quantity (transmission is documented in the thesis);
   - exact spectral range and sampling grid;
   - TiO2/SiO2 refractive-index or dispersion model;
   - polarization;
   - incident and exit media/boundary assumptions.

2. **Resolve the spectral-range discrepancy.**
   - Dissertation: 300–750 nm.
   - Some working manuscript text: 350–750 THz.
   - Use only the value verified from the actual code/data record.

3. **Replace the extended 20-layer 200k filenames** in `data/dataset_manifest.csv` with the final archived filenames.

4. **Deposit raw datasets externally** and add the permanent DOI.

5. **Replace placeholders** in `README.md` and `CITATION.cff` with the GitHub URL and DOI.

6. **Finalize authorship/affiliations** in the manuscript before public release.

7. **Verify bibliographic metadata** in the manuscript against publisher/Crossref records.
