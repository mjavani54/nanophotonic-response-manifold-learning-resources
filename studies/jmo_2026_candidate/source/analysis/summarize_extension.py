#!/usr/bin/env python3
"""Create audit-ready candidate endpoints only after every planned fit exists."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.out / "experiment.json").read_text())
    with (args.out / "extended_fit_records.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    keyed = {}
    for r in records:
        key = (int(r["layers"]), int(r["n_train"]), int(r["seed"]))
        if key in keyed:
            raise ValueError(f"Duplicate fit: {key}")
        keyed[key] = float(r["test_r2"])
    expected = {(layer, n, seed) for layer in meta["layers"]
                for n in meta["sizes"] for seed in meta["seeds"]}
    missing, unexpected = expected - keyed.keys(), keyed.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Incomplete or unexpected records: {len(missing)} missing, {len(unexpected)} unexpected")
    table = []
    for layer in meta["layers"]:
        grid = []
        for n in meta["sizes"]:
            scores = [keyed[layer, n, seed] for seed in meta["seeds"]]
            grid.append({"n": n, "mean": mean(scores), "sd": stdev(scores),
                         "passing_seeds": sum(v >= .999 for v in scores)})
        first = next((g for g in grid if g["mean"] >= .999), None)
        table.append({
            "layers": layer,
            "r2_target": .999,
            "first_tested_train_size": first["n"] if first else "Not reached",
            "status": "resolved_on_generated_extension_grid" if first else "right_censored_on_generated_extension_grid",
            "max_tested_train_size": grid[-1]["n"],
            "mean_test_r2_at_first_or_max": first["mean"] if first else grid[-1]["mean"],
            "sd_test_r2_at_first_or_max": first["sd"] if first else grid[-1]["sd"],
            "seeds_passing_at_first_or_max": first["passing_seeds"] if first else grid[-1]["passing_seeds"],
            "source": "40k corrected-primary training pool plus relabeled augmented designs; corrected-primary holdouts",
        })
    output = args.out / "table2_extension_candidates.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(row)
    print(f"Wrote {output}. Confirm baseline replay and review all seed records before manuscript changes.")


if __name__ == "__main__":
    main()
