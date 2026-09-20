#!/usr/bin/env python3
"""Lightweight integrity check for the public reproducibility package."""
from pathlib import Path
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "environment.yml",
    "manuscript/ACS_Photonics_Manuscript_DataCode_Aligned.docx",
    "data_generation/multlayer_metamat_generation_old.m",
    "data_generation/generate_legacy_multilayer.py",
    "data_generation/validate_legacy_generator.py",
    "data_generation/build_paper_datasets.py",
    "data/exact_designs/paper1_exact_designs.npz",
    "data/reference_fixture/legacy_generator_reference.npz",
    "results/final_master_summary.csv",
    "results/final_correlation_statistics.csv",
    "results/final_learning_curves.csv",
    "results/left_censoring_sensitivity.csv",
    "figures/Figure1_intrinsic_dimension_vs_layers.png",
    "figures/Figure2_controlled_learning_curves.png",
    "figures/Figure3_ae_dimension_vs_sample_requirement.png",
    "figures/Figure5_ae_dimension_vs_model_capacity.png",
    "figures/Figure6_20layer_extended_learning_curve.png",
]

missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    print("Missing required files:")
    for p in missing:
        print(" -", p)
    sys.exit(1)

with (ROOT / "results/final_master_summary.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 7:
    raise SystemExit(f"Expected 7 rows in final_master_summary.csv; found {len(rows)}")
layers = [int(float(r["layers"])) for r in rows]
if layers != [8, 10, 12, 14, 16, 18, 20]:
    raise SystemExit(f"Unexpected layer ordering: {layers}")

subprocess.run(
    [sys.executable, str(ROOT / "data_generation/validate_legacy_generator.py")],
    check=True,
    cwd=ROOT,
)

print("Repository verification passed.")
print("Layer counts:", layers)
print("Historical data generator and archived-label fixture validation passed.")
