# Commands after restoring the JMO data bundle

Use Python 3 with `source/requirements.txt` in an environment that supports NumPy, SciPy, scikit-learn, pandas, matplotlib, and PyTorch. The source snapshot preserves the actual scripts, split recipes, fit grids, and metadata. Its original README discusses how the current response model differs from earlier archived calculations; this repository's root README is the current index.

```bash
python scripts/restore_data.py "/path/to/downloaded_parts" "/path/to/data_output"
python scripts/verify_package.py --data "/path/to/data_output/JMO_Data_Assets"
python verify_summary.py "/path/to/data_output/JMO_Data_Assets"
bash source/run.sh verify
```

The original `source/run.sh` can rerun intrinsic, cross-structure, capacity, optimization, and extended 20-layer fits against the *reconstructed bundle* by using `primary/Datasets50k_eps21316.zip`, `extended/Datasets20_eps21316.zip`, and the root `material_model_manifest.json` inside `JMO_Data_Assets/`. It verifies these present ZIP hashes. Long neural fits may not reproduce bit-for-bit on a different GPU/software stack; report validation with the per-seed CSVs.

The three supplied augmented NPZs exactly match the recorded hashes. To regenerate them from only the compact source designs, run `python scripts/reproduce_augmentation.py ASSET_DIR OUTPUT_DIR`; it fails if any output hash differs. To regenerate the four spectral input arrays from the two supplied design ensembles, run `python scripts/reproduce_spectral.py ASSET_DIR OUTPUT_DIR`; it verifies current bundle hashes. The original `source/analysis/compare_ae_bands.py` expects a historical design ZIP container that was not supplied in exactly the recorded form. Use the included design-source and generated spectral-array checks instead of substituting old response labels.

`python source/data_generation/build_paper_datasets.py --design-npz ASSET_DIR/source_designs/paper1_exact_designs.npz --out-dir YOUR_FRESH_OUTPUT_DIR` reconstructs the primary and 200k CSV ZIPs. Their ZIP-container hashes may differ because the original container bytes are unavailable. Verify known CSV-member SHA values and documented QC before comparing results. Do not overwrite the recorded metadata files under `records/`.
