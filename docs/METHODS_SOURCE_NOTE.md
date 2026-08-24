# Methods source note from the dissertation

The dissertation provides the following source-supported description of the multilayer simulations used as the basis for the multilayer datasets:

> The multilayer systems are one-dimensional N-layer thin-film filters composed of alternating titania (TiO2) and silica (SiO2) layers. The design variables are the individual layer thicknesses, each varied over 30–70 nm. The optical response is the transmission spectrum, calculated using a custom Python implementation of the transfer-matrix method (TMM). The dissertation describes the calculation over a wavelength interval of 300–750 nm and depicts illumination perpendicular to the layer planes with transmission evaluated on the opposite side of the stack.

Source: *Machine Learning Approaches for Knowledge Discovery in Nanophotonic Structures*, Chapter 2, Sec. 2.1 (multilayer thin films), especially dissertation pp. 14–16 / PDF pp. 31–33.

## What the dissertation does not document clearly enough

Do **not** infer these items from the thesis alone:

- numerical refractive-index values or an explicit TiO2/SiO2 dispersion formula;
- polarization used for the multilayer TMM calculation;
- exact refractive index/material of the incident and exit semi-infinite media;
- any alternative spectral range stated elsewhere in the current manuscript.

Those details should be copied from the exact dataset-generation code used for this paper.

## Spectral-range discrepancy to resolve

The dissertation states **300–750 nm** for the multilayer transmission calculation. Earlier working versions of this paper referred to **350–750 THz**. These are not equivalent. The final manuscript and public repository must use the range and sampling convention verified from the actual simulation code/dataset-generation record.
