# Data-generation provenance

The final repository was audited against the archived historical MATLAB generator used for the multilayer labels.

## Source facts

The original source file is `data_generation/multlayer_metamat_generation_old.m`. It specifies:

- `NSamp = 200`;
- `fmin = 250e3` GHz and `fmax = 1050e3` GHz;
- `p = 1` and `theta = 0`;
- `epsilon_Sio2 = 1.46`;
- `epsilon_Tio2 = 5.913`;
- layer thickness range 30–70 nm;
- alternating material assignment beginning with the SiO2 parameter;
- stored output `response = abs(taot)`.

The propagation loop contains only `1:(N-1)` and the final step applies an interface into material N. The final sampled thickness is therefore absent from the response calculation.

## Empirical cross-check

The Python translation was compared against archived label columns from the paper datasets. Using the stored design values, it reproduces the archived responses to small numerical differences consistent with the historical CSV rounding. This verifies that the 250–1050 THz legacy definition, rather than a 350–750 nm power-transmittance definition, generated the analyzed response files.

## Consequence for manuscript wording

The manuscript must describe the analyzed labels as the historical field-transmission-magnitude response on the 250–1050 THz frequency grid. It should not describe these specific datasets as 350–750 nm finite-stack power transmittance.

A physics-corrected finite-stack power-transmittance generator is a different model and would require regenerated datasets and a new analysis.
