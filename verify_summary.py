#!/usr/bin/env python3
"""Validate candidate-study summaries; never claim full raw-data validation."""
from pathlib import Path
from hashlib import sha256
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
REC = HERE / "records"

def read_csv(name):
    with (REC / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

manifest = json.loads((REC / "data/material_model_manifest.json").read_text())
generator = HERE / "source/data_generation/generate_legacy_multilayer.py"
designs = (Path(sys.argv[1]) / "source_designs/paper1_exact_designs.npz") if len(sys.argv) > 1 else None
assert sha256(generator.read_bytes()).hexdigest() == manifest["generator_sha256"]
if designs is not None:
    assert sha256(designs.read_bytes()).hexdigest() == manifest["design_npz_sha256"]

rows = read_csv("primary_learning_curves.csv")
assert len(rows) == 72
layers = [8, 10, 12, 14, 16, 18, 20]
expected = {
    0.99: [1000, 1000, 2000, 4000, 10000, 15000, 25000],
    0.995: [1000, 2000, 3000, 10000, 10000, 20000, 35000],
    0.999: [3000, 10000, 15000, None, 30000, None, None],
}
threshold_rows = {(int(x["layers"]), float(x["target_r2"])): x
                  for x in read_csv("primary_thresholds.csv")}
assert len(threshold_rows) == 21
for threshold, numbers in expected.items():
    for layer, expected_size in zip(layers, numbers):
        series = sorted((x for x in rows if int(x["layers"]) == layer),
                        key=lambda x: int(x["n_train"]))
        assert len(series) == (12 if layer in (8, 10, 12, 14) else 8)
        reached = [int(x["n_train"]) for x in series
                   if float(x["test_r2_mean"]) >= threshold]
        first = min(reached) if reached else None
        assert first == expected_size, (layer, threshold, first)
        recorded = threshold_rows[layer, threshold]
        assert (int(recorded["first_tested_samples"]) if recorded["first_tested_samples"] else None) == first

master = read_csv("intrinsic/paper1_master_summary.csv")
assert ([(int(x["layers"]), int(x["ae_dim_r2_99"])) for x in master]
        == list(zip(layers, [6, 7, 9, 12, 14, 20, 30])))
assert [int(float(x["params_for_r2_99"])) for x in master[:3]] == [
    17736, 60232, 60488]
capacity = {int(x["layers"]): x
            for x in read_csv("capacity/capacity_diagnostic_decision.csv")}
assert [int(float(capacity[n]["min_parameters_reaching_target"]))
        for n in (14, 16, 18)] == [77256, 253128, 527816]
capacity20 = capacity[20]
assert not capacity20["min_parameters_reaching_target"]
assert int(capacity20["best_parameters"]) == 5202120
paired = read_csv("matched_low_sample/paired_5k_check.csv")
assert len(paired) == 12 and all(float(x["difference"]) == 0 for x in paired)

spectral = {(x["case"], int(x["latent_dim"])): float(x["mean_test_r2"])
            for x in read_csv("spectral/ae_summary.csv")}
assert len(spectral) == 16
assert len(read_csv("spectral/ae_results.csv")) == 48
for cohort in ("old", "current"):
    assert spectral[cohort + "_narrow", 10] >= 0.99
    assert spectral[cohort + "_wide", 10] < 0.99
    assert spectral[cohort + "_wide", 30] >= 0.99

aug = read_csv("extension_run/extended_summary.csv")
assert len(aug) == 24
assert len(read_csv("extension_run/extended_fit_records.csv")) == 72
for layer in (14, 18):
    passing = [int(x["n_train"]) for x in aug
               if int(x["layers"]) == layer and float(x["mean_test_r2"]) >= .999]
    assert min(passing) == 50000

separate20 = read_csv("extended_20/learning_curve_summary.csv")
for target, first_expected in ((.99, 30000), (.995, 50000), (.999, 75000)):
    first = min(int(x["n_train"]) for x in separate20
                if float(x["test_r2_mean"]) >= target)
    assert first == first_expected


table1 = list(csv.DictReader((ROOT / "tables/article_table_1.csv").open(newline="", encoding="utf-8")))
table2 = list(csv.DictReader((ROOT / "tables/article_table_2.csv").open(newline="", encoding="utf-8")))
assert len(table1) == len(table2) == 7
for i, layer in enumerate(layers):
    one, two, raw = table1[i], table2[i], master[i]
    assert int(one["nominal_layers"]) == int(two["nominal_layers"]) == layer
    assert int(one["pca99_components"]) == int(raw["pca_dim_99"])
    assert int(one["ae99_threshold"]) == int(raw["ae_dim_r2_99"])
    assert round(float(raw["participation_ratio"]), 2) == float(one["participation_ratio"])
    first = expected[0.99][i]
    assert int(one["first_tested_samples_r2_ge_099"].replace("≤", "").replace(",", "")) == first
    assert int(two["primary_r2_ge_099"].replace("≤", "").replace(",", "")) == first
    for target, field in ((0.995, "primary_r2_ge_0995"), (0.999, "primary_r2_ge_0999")):
        value = two[field]
        observed = None if value == "Not reached" else int(value.replace("≤", "").replace(",", ""))
        assert observed == expected[target][i]
    if layer < 20:
        param = int(one["first_tested_parameters_r2_ge_099"].replace(",", ""))
        observed = int(float(raw["params_for_r2_99"])) if layer < 14 else int(float(capacity[layer]["min_parameters_reaching_target"]))
        assert param == observed
    else:
        assert one["first_tested_parameters_r2_ge_099"] == "Not reached"
assert [two["augmented_extension_r2_ge_0999"] for two in table2] == [
    "Not tested", "Not tested", "Not tested", "50,000",
    "40,000 baseline", "50,000", "Not tested"]

print("New-run summaries and source-generator SHA passed; source-design SHA " + ("passed." if designs is not None else "requires an extracted data bundle."))
print("Full response arrays and complete regeneration were not checked.")
