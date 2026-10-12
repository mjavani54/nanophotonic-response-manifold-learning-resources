# Verified methods source note

The author reports that Python generated the analyzed multilayer datasets. The archived MATLAB file `data_generation/multlayer_metamat_generation_old.m` and selected response fixtures support the numerical convention below, but the specific original Python script has not been identified in the public repository.

## Documented response convention

- alternating design entries assigned `epsilon_SiO2 = 1.46` and `epsilon_TiO2 = 5.913`;
- `mu = 1`, `sigma = 0`;
- thickness entries sampled between 30 and 70 nm;
- normal incidence (`theta = 0°`);
- legacy polarization flag `p = 1`;
- 200 uniformly spaced **frequency** points from **250 to 1050 THz**;
- stored label `abs(taot)`, the magnitude of the legacy field-transmission coefficient.

The historical MATLAB loop propagates only `1:(N-1)` thicknesses before applying an interface into material `N`. Consequently, the final sampled thickness does not affect the stored label.

The Python reconstruction in `data_generation/generate_legacy_multilayer.py` agrees with selected archived fixtures within the stated numerical tolerance when supplied with the stored rounded designs. This check does not establish exact reproduction of every archived label.

## Manuscript consequence

The checked response fixtures support the 250–1050 THz convention. Earlier thesis prose/table summaries that referred to 300–750 nm or 350–750 nm do not describe this checked convention.
