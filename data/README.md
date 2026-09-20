# Data and regeneration assets

The repository includes the exact **stored design matrices** used by the paper in compressed form:

`data/exact_designs/paper1_exact_designs.npz`

Keys:

- `L8_50k`
- `L10_50k`
- `L12_50k`
- `L14_50k`
- `L16_50k`
- `L18_50k`
- `L20_50k`
- `L20_200k`

Each array is stored in the same orientation used by the historical CSV files: `(design variables, samples)`, with thickness values in meters.

## Response labels

The historical response definition can be regenerated with:

```bash
python data_generation/generate_legacy_multilayer.py   --design-npz data/exact_designs/paper1_exact_designs.npz   --key L8_50k   --out-dir generated/L8
```

See `data_generation/README.md` for the exact historical TMM convention.

## Exact archived response CSVs

For byte-level reproduction of the original downstream analysis, the original rounded response CSVs should be attached to a tagged GitHub Release rather than committed to Git history. The repository's generator reproduces the same historical numerical definition, but small differences can occur when recomputing from the stored rounded design values because the original unrounded random thickness values were not archived separately.

A separate DOI-bearing raw-data repository is therefore optional if the exact raw CSV ZIP files are distributed as GitHub Release assets together with a tagged code release.
