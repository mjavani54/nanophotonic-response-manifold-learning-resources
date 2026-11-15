#!/usr/bin/env bash
set -euo pipefail

# Run this beside the completed epsilon=2.1316 experiment. The first argument
# is the extracted SiO2_Epsilon_21316_Rerun_v1.0.0 project directory.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash run_low_sample_matched.sh "/path/to/SiO2_Epsilon_21316_Rerun_v1.0.0" ["/path/to/corrected_runs"]' >&2
  exit 2
fi

PROJECT="$(cd -- "$1" && pwd)"
RUNS="${2:-$PROJECT/corrected_runs}"
SMALL="$RUNS/data/Datasets50k_eps21316.zip"
MODEL="$RUNS/data/material_model_manifest.json"
CROSS="$RUNS/cross/cross_structure_learning_curve_results.csv"
OUT="$RUNS/matched_low_sample_1024"
BUNDLE="$RUNS/Matched_Low_Sample_1024_for_Review.zip"
PY="${PYTHON:-$PROJECT/.venv/bin/python}"

[[ -x "$PY" ]] || { echo "Python environment not found: $PY" >&2; exit 2; }
for input in "$SMALL" "$MODEL" "$CROSS" "$PROJECT/verify_inputs.py" "$PROJECT/code/paper1_low_sample_refinement.py"; do
  [[ -f "$input" ]] || { echo "Missing input: $input" >&2; exit 2; }
done
if [[ -e "$OUT" || -e "$BUNDLE" ]]; then
  echo "Output already exists: $OUT or $BUNDLE. Choose a fresh runs directory or archive the old output first." >&2
  exit 2
fi

export MPLBACKEND=Agg
"$PY" "$PROJECT/verify_inputs.py" --small "$SMALL" --manifest "$MODEL"
"$PY" "$PROJECT/code/paper1_low_sample_refinement.py" \
  --zip-small "$SMALL" --out "$OUT" \
  --batch-size 1024 --train-sizes 1000 2000 3000 4000 5000

"$PY" - "$OUT" "$CROSS" "$SMALL" "$MODEL" "$PROJECT/code/paper1_low_sample_refinement.py" "$BUNDLE" <<'PY'
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

out, cross_path, small, model, script, bundle = map(Path, sys.argv[1:])

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

with (out / 'low_sample_refinement_results.csv').open(newline='') as f:
    new = list(csv.DictReader(f))
with cross_path.open(newline='') as f:
    old = list(csv.DictReader(f))
old_5k = {(int(r['layers']), int(r['seed'])): r for r in old
          if int(r['n_train']) == 5000 and int(r['layers']) in (8, 10, 12, 14)}
paired = []
for r in new:
    if int(r['n_train']) != 5000:
        continue
    key = (int(r['layers']), int(r['seed']))
    assert key in old_5k, f'Missing main-grid comparison: {key}'
    original = old_5k[key]
    assert r['n_parameters'] == original['n_parameters'], key
    delta = float(r['test_r2']) - float(original['test_r2'])
    paired.append({'layers': key[0], 'seed': key[1],
                   'matched_low_5k_r2': r['test_r2'],
                   'main_5k_r2': original['test_r2'],
                   'difference': format(delta, '.17g')})
assert len(paired) == 12, len(paired)
with (out / 'paired_5k_check.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0]))
    w.writeheader()
    w.writerows(paired)
provenance = {
    'purpose': 'Match the existing main-grid batch size and compare the 5k overlap.',
    'batch_size': 1024,
    'training_sizes': [1000, 2000, 3000, 4000, 5000],
    'epsilon_r_SiO2': 2.1316,
    'small_zip_sha256': digest(small),
    'material_manifest_sha256': digest(model),
    'cross_results_sha256': digest(cross_path),
    'low_sample_script_sha256': digest(script),
    'max_absolute_5k_r2_difference': max(abs(float(x['difference'])) for x in paired),
    'note': 'Review the per-seed differences before consolidating thresholds. GPU nondeterminism can prevent bit-for-bit equality.'
}
(out / 'matched_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
files = sorted(p for p in out.rglob('*') if p.is_file())
manifest = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}
(out / 'FILE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():
            z.write(p, arcname='matched_low_sample_1024/' + str(p.relative_to(out)))
print('Matched low-sample run:', out)
print('Max |difference| at 5k:', provenance['max_absolute_5k_r2_difference'])
print('Upload this review bundle:', bundle)
PY
