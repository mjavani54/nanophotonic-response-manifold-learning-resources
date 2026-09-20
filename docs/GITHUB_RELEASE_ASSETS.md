# GitHub Release assets for exact raw-data reproduction

For a no-separate-DOI release strategy, upload the following files as assets of the tagged GitHub release.

| Asset | Purpose | SHA-256 |
|---|---|---|
| `Paper1_50k_RawData.zip` | Exact 8–20 layer 50k DS/RS CSVs used in the primary analyses | `8a4406d8d470d8c56f424baf835097606d2742eea10e6a1ba7b7ef15671986cb` |
| `Paper1_20L_200k_RawData.zip` | Exact 20-layer 200k DS/RS CSVs used in the extended learning curve | `6aeef791606c60706d4f0f88b72e5a925177c7f4e7eac3039fe2abd550f65a01` |

These files should not be committed to ordinary Git history because of their size. Attach them to the same tagged release as repository version 1.1.0.

Once uploaded, the manuscript may cite the public GitHub repository/release URL rather than a separate raw-data DOI.
