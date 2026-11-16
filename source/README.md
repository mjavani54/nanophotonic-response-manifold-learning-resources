# SiO2 epsilon = 2.1316: controlled manuscript rerun

This is a **new experiment**, derived from repository commit
`1629afb7813dc0db57fdac7c0e600e0f95f1c002` of
[`nanophotonic-response-manifold-learning-resources`](https://github.com/mjavani54/nanophotonic-response-manifold-learning-resources).
The generator and affected analysis scripts are bundled here, so the old
repository and frozen releases can remain untouched. The baseline numerical
parameter `epsilon_r(SiO2) = 1.46` was being used as **permittivity**; this
package instead uses `epsilon_r(SiO2) = 2.1316 = 1.46²`. It keeps
`epsilon_r(TiO2) = 5.913`, the same frequency grids, design cohorts, model
architectures, splits, accuracy definitions, and *legacy transfer convention*.

**Scientific limit:** This is an epsilon-only sensitivity/rerun, **not** a
fully physical multilayer simulation. The inherited transfer convention
propagates only the first N−1 sampled thicknesses and ends in the Nth material
instead of applying an exit interface into air. It outputs field coefficient
`|t|`, not power transmittance. Both materials are constant and nondispersive
over 250–1050 THz (and 350–750 THz in the secondary window). Correcting one
constant does not fix those other issues or validate a fabricated stack.
Describe the results accordingly; a realistic finite-stack claim needs a
separately validated dispersive solver and a complete new training analysis.

## Inputs required (not included in this small code package)

1. `paper1_exact_designs.npz` from
   `data/exact_designs/` in the public repository. It has keys `L8_50k` through
   `L20_50k`, plus `L20_200k`. This preserves the exact primary design arrays.
2. The **original** `Table2_Extended_Data_and_Run_Package.zip`, reconstructed
   from the 14 original parts with their `restore_archive.py`, if rerunning
   the augmented cohort. The new generator reads its `designs_m` arrays and
   **never uses** its old response labels. If the archive is unavailable,
   pass `-` and the original recorded seeds will regenerate the extra designs;
   confirm the exact design arrays before calling that a paired rerun.
3. `historical_20L.zip` from the prior
   `IEEE_Photonics_20L_Window_Sensitivity_Resources.zip` for the spectral
   comparison. Its original response CSV is intentionally ignored: only its
   50,000 historical thickness vectors are loaded.

The new `Datasets50k_eps21316.zip` and `Datasets20_eps21316.zip` are generated
from input 1. Their internal filenames match the original analysis scripts,
but the ZIPs and manifest are named differently and the results must be saved
in new directories. The new CSVs keep 17 significant digits to avoid adding
the old rounding discrepancy between saved designs and recalculated labels.

## Ubuntu/WSL and macOS commands

Use Bash. Keep quoted paths if a folder contains spaces.

```bash
cd "/path/to/SiO2_Epsilon_21316_Rerun_v1.0.0"
bash run.sh setup
bash run.sh verify

DESIGNS="/path/to/paper1_exact_designs.npz"
HISTORICAL="/path/to/historical_20L.zip"
ORIGINAL_TABLE2="/path/to/Table2_Extended_Data_and_Run_Package.zip"
RUNS="$PWD/corrected_runs"
mkdir -p "$RUNS"

bash run.sh generate "$DESIGNS" "$RUNS/data"
SMALL="$RUNS/data/Datasets50k_eps21316.zip"
EXTENDED="$RUNS/data/Datasets20_eps21316.zip"
MODEL="$RUNS/data/material_model_manifest.json"

bash run.sh intrinsic "$SMALL" "$MODEL" "$RUNS/intrinsic"
bash run.sh primary_additional "$SMALL" "$MODEL" "$RUNS/intrinsic"
bash run.sh cross "$SMALL" "$EXTENDED" "$MODEL" "$RUNS/intrinsic" "$RUNS/cross"
bash run.sh extended_20 "$EXTENDED" "$MODEL" "$RUNS/extended_20"
bash run.sh low_sample "$SMALL" "$MODEL" "$RUNS/low_sample"
bash run.sh capacity "$SMALL" "$MODEL" "$RUNS/capacity"
bash run.sh optimization "$SMALL" "$MODEL" "$RUNS/optimization"

bash run.sh extension_data "$SMALL" "$MODEL" "$ORIGINAL_TABLE2" "$RUNS/extension_data"
bash run.sh extension_run "$SMALL" "$MODEL" "$RUNS/extension_data" "$RUNS/cross" "$RUNS/extension_run"

bash run.sh spectral_inspect "$HISTORICAL" "$SMALL" "$MODEL" "$RUNS/spectral"
bash run.sh spectral "$HISTORICAL" "$SMALL" "$MODEL" "$RUNS/spectral"

bash run.sh bundle "$RUNS" "$PWD/Corrected_SiO2_Results_for_Review.zip"
```

`setup` installs packages in `.venv`. On Ubuntu/WSL, install `python3-venv`
first if your distribution lacks it. CUDA PyTorch on Linux may require the
appropriate official build for your driver; the installation above uses the
default package index. Check `python -c 'import torch; print(torch.cuda.is_available())'`
in the environment before lengthy training. On Apple hardware, the spectral
and augmented scripts select MPS when available; some inherited primary
analysis scripts select CPU if CUDA is absent. Set aside substantial compute
time and free disk space: the full rerun includes hundreds of neural fits and
large CSVs. A single GPU does not imply that the code will use all four GPUs.

## Order, outputs, and safeguards

- `generate` creates seven 50k primary sets and the independent 200k 20-layer
  set on 250–1050 THz. A new `material_model_manifest.json` records the
  design-source checksum, generator checksum, numerical model, and output ZIP
  checksums. The command refuses an existing data output directory.
- `intrinsic` does PCA and the full three-seed AE grid. Its AE fits checkpoint
  to CSV and resume only when the dataset/protocol metadata match. It skips
  redundant forward/capacity sweeps; `primary_additional` then runs those
  original sweeps to update Table 1 capacity and supplementary analyses.
- `cross` reads **new** PCA and AE summary CSVs, rather than hardcoded old
  dimensions. It tests seven layer classes on the original controlled
  training grid. `low_sample`, `capacity`, `optimization`, and `extended_20`
  rerun the remaining primary/diagnostic resource claims. Use fresh output
  directories for these non-resumable inherited scripts.
- `extension_data` relabels the same additional 14/16/18-layer designs from
  the original Table 2 archive if supplied. `extension_run` replays each
  **corrected** 40k baseline against `cross`'s new summary, then extends the
  learning curves. Do not weaken its baseline gate to obtain a desired result.
  Completed augmented fits can resume from their recorded CSV/metadata.
- `spectral_inspect` computes the 2×2 corrected spectra and PCA diagnostics.
  `spectral` then trains the same 48 planned AEs. It ignores the old archive's
  ε=1.46 response CSV and checks its corrected-current labels against the new
  generator before training. Completed fits can resume when metadata match.
- `bundle` includes CSVs, metadata, logs, and figures, while excluding the
  potentially huge response archives and extracted input data.

The material change can alter all numerical endpoints, including whether a
target is first reached, not reached, or left censored. Old tables, figures,
summary statistics, numerical claims, and spectral-window interpretations
cannot simply be copied into a manuscript with `2.1316` substituted. Review
new curves and optimization convergence, then revise the manuscript and SI
from **these new records**. Recheck whether the original fixed width schedule
removes capacity limitations under the changed spectra. If it does not,
predeclare and run an expanded capacity analysis before drawing a resource
conclusion.

## What to send back for manuscript revision

Upload `Corrected_SiO2_Results_for_Review.zip` after all planned commands
complete, together with `material_model_manifest.json` if you bundled a
different output tree. Include any stderr/terminal log of a failed command.
Keep the big raw ZIPs locally unless an audit of particular spectra is needed.
The review ZIP does not assert that the manuscript is already ready to submit.

The supplied source scripts were adapted from the public repository and
earlier submitted Table 2/spectral resource packages. The paper repository's
old datasets and releases should stay frozen; publish the new datasets and
analysis under a clearly distinct corrected-model version only after review.
