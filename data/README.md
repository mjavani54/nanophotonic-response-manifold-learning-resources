# Raw data

Raw simulation CSV files are intentionally excluded from GitHub because of their size.

## Primary 50k multilayer datasets

The analysis expects design (`DS`) and response (`RS`) matrices for 8, 10, 12, 14, 16, 18, and 20 layers. The original file naming convention is documented in `dataset_manifest.csv`.

For the 50k datasets:

- `DS`: one design variable per layer; 50,000 samples.
- `RS`: 200 spectral response values; 50,000 samples.

The scripts accept either samples-by-features or features-by-samples orientation where implemented and orient arrays internally.

## Extended 20-layer dataset

The extended 20-layer study uses 200,000 total design-response samples. The raw 200k CSV files should also be archived externally.

## Recommended public release

1. Deposit the raw CSV files in Zenodo or another DOI-bearing repository.
2. Include file-level checksums in the data record.
3. Add the DOI to the root `README.md` and `CITATION.cff`.
4. Do not add the raw CSVs to Git history.

## Local layout

```text
data/raw/
  SiO2_TiO2_8_layer_50K_200_points_DS.csv
  SiO2_TiO2_8_layer_50K_200_points_RS.csv
  ...
  SiO2_TiO2_20_layer_50K_200_points_DS.csv
  SiO2_TiO2_20_layer_50K_200_points_RS.csv
  # plus the externally archived 20-layer 200k DS/RS pair
```
