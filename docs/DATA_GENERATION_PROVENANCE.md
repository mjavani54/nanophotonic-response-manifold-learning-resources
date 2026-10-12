# Data-generation provenance

The author reports that Python generated the analyzed multilayer labels. This repository includes an older MATLAB implementation and a later Python reconstruction, but does not identify the specific Python script originally executed. The numerical response convention below is supported by comparison with selected archived label fixtures.

## Source facts

The archived MATLAB reference file is `data_generation/multlayer_metamat_generation_old.m`. It specifies:

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

The Python reconstruction was compared against selected archived label columns. With stored rounded design values, the checked fixtures agree within the tolerance reported by `validate_legacy_generator.py`. This supports the documented 250–1050 THz response convention for the checked fixtures. It does not establish that the MATLAB program was run, identify the original Python generator, or show bitwise agreement for the full archives.

## Consequence for manuscript wording

The manuscript must describe the analyzed labels as the historical field-transmission-magnitude response on the 250–1050 THz frequency grid. It should not describe these specific datasets as 350–750 nm finite-stack power transmittance.

A physics-corrected finite-stack power-transmittance generator is a different model and would require regenerated datasets and a new analysis.
