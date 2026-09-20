# Verified methods source note

The multilayer dataset-generation convention is now verified from the archived MATLAB source file `data_generation/multlayer_metamat_generation_old.m` and by numerical comparison with the analyzed response CSVs.

## Verified historical generator

- alternating design entries assigned `epsilon_SiO2 = 1.46` and `epsilon_TiO2 = 5.913`;
- `mu = 1`, `sigma = 0`;
- thickness entries sampled between 30 and 70 nm;
- normal incidence (`theta = 0°`);
- legacy polarization flag `p = 1`;
- 200 uniformly spaced **frequency** points from **250 to 1050 THz**;
- stored label `abs(taot)`, the magnitude of the legacy field-transmission coefficient.

The historical MATLAB loop propagates only `1:(N-1)` thicknesses before applying an interface into material `N`. Consequently, the final sampled thickness does not affect the stored label.

The Python translation in `data_generation/generate_legacy_multilayer.py` reproduces the archived labels when supplied with the stored design matrices, within the small round-trip differences caused by the historical CSV precision.

## Manuscript consequence

This verified source overrides earlier thesis prose/table summaries that referred to 300–750 nm or 350–750 nm. Those ranges do not describe the actual response CSVs analyzed in this paper.
