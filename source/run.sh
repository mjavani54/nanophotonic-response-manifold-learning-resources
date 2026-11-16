#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="${PYTHON:-python3}"; fi
export MPLBACKEND=Agg

usage() {
  cat <<'EOF'
Usage (all paths may contain spaces; quote them):
  bash run.sh setup
  bash run.sh verify
  bash run.sh generate EXACT_DESIGNS_NPZ OUTPUT_DATA_DIR
  bash run.sh intrinsic SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh primary_additional SMALL_ZIP MATERIAL_MANIFEST INTRINSIC_OUTPUT_DIR
  bash run.sh cross SMALL_ZIP EXTENDED_ZIP MATERIAL_MANIFEST INTRINSIC_OUTPUT_DIR OUTPUT_DIR
  bash run.sh extended_20 EXTENDED_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh low_sample SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh capacity SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh optimization SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh extension_data SMALL_ZIP MATERIAL_MANIFEST ORIGINAL_TABLE2_ZIP_OR_DASH OUTPUT_DIR
  bash run.sh extension_run SMALL_ZIP MATERIAL_MANIFEST EXTENSION_DATA_DIR CROSS_OUTPUT_DIR OUTPUT_DIR
  bash run.sh spectral_inspect HISTORICAL_20L_ZIP SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh spectral HISTORICAL_20L_ZIP SMALL_ZIP MATERIAL_MANIFEST OUTPUT_DIR
  bash run.sh bundle RUNS_ROOT OUTPUT_ZIP
EOF
}
fresh() {
  if [[ -e "$1" ]]; then
    printf 'Refusing to reuse %s. Choose a fresh output directory.\n' "$1" >&2
    exit 2
  fi
}
check_small() { "$PY" "$ROOT/verify_inputs.py" --small "$1" --manifest "$2"; }
check_extended() { "$PY" "$ROOT/verify_inputs.py" --extended "$1" --manifest "$2"; }

cmd="${1:-}"
if [[ -z "$cmd" ]]; then usage; exit 2; fi
shift
case "$cmd" in
  setup)
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/python" -m pip install --upgrade pip
    "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
    ;;
  verify)
    "$PY" "$ROOT/verify_corrected_model.py"
    ;;
  generate)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    fresh "$2"
    "$PY" "$ROOT/data_generation/build_paper_datasets.py" --design-npz "$1" --out-dir "$2"
    check_small "$2/Datasets50k_eps21316.zip" "$2/material_model_manifest.json"
    check_extended "$2/Datasets20_eps21316.zip" "$2/material_model_manifest.json"
    ;;
  intrinsic)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_small "$1" "$2"
    "$PY" "$ROOT/code/paper1_intrinsic_dimensionality_pipeline_fixed.py" --zip "$1" --source-manifest "$2" --out "$3" --mode full --skip-learning-curves --skip-capacity
    ;;
  primary_additional)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_small "$1" "$2"
    [[ -f "$3/ae_dimension_summary.csv" ]] || { echo 'Run intrinsic first.' >&2; exit 2; }
    "$PY" "$ROOT/code/paper1_intrinsic_dimensionality_pipeline_fixed.py" --zip "$1" --source-manifest "$2" --out "$3" --mode full --skip-ae
    ;;
  cross)
    [[ $# -eq 5 ]] || { usage; exit 2; }
    check_small "$1" "$3"; check_extended "$2" "$3"; fresh "$5"
    "$PY" "$ROOT/code/paper1_cross_structure_learning_curves_fixed.py" --zip-small "$1" --zip-20 "$2" --primary-manifest "$3" --intrinsic-dir "$4" --out "$5"
    ;;
  extended_20)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_extended "$1" "$2"; fresh "$3"
    "$PY" "$ROOT/code/paper1_20layer_200k_learning_curve.py" --zip "$1" --out "$3"
    ;;
  low_sample)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_small "$1" "$2"; fresh "$3"
    "$PY" "$ROOT/code/paper1_low_sample_refinement.py" --zip-small "$1" --out "$3"
    ;;
  capacity)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_small "$1" "$2"; fresh "$3"
    "$PY" "$ROOT/code/paper1_high_capacity_diagnostic.py" --zip "$1" --out "$3"
    ;;
  optimization)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    check_small "$1" "$2"; fresh "$3"
    "$PY" "$ROOT/code/paper1_20layer_optimization_diagnostic.py" --zip "$1" --out "$3"
    ;;
  extension_data)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    check_small "$1" "$2"; fresh "$4"
    extra=()
    if [[ "$3" != '-' ]]; then extra=(--source-augmented-zip "$3"); fi
    "$PY" "$ROOT/analysis/generate_extension.py" --primary "$1" --primary-manifest "$2" --generator "$ROOT/data_generation/generate_legacy_multilayer.py" --out "$4" "${extra[@]}"
    ;;
  extension_run)
    [[ $# -eq 5 ]] || { usage; exit 2; }
    check_small "$1" "$2"
    "$PY" "$ROOT/analysis/run_extended_learning_curves.py" --primary "$1" --primary-manifest "$2" --extra-dir "$3" --pipeline "$ROOT/code/paper1_cross_structure_learning_curves_fixed.py" --reference "$4/cross_structure_learning_curve_summary.csv" --out "$5"
    "$PY" "$ROOT/analysis/summarize_extension.py" --out "$5"
    ;;
  spectral_inspect|spectral)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    check_small "$2" "$3"
    option=()
    if [[ "$cmd" == 'spectral_inspect' ]]; then option=(--inspect-only); fi
    "$PY" "$ROOT/analysis/compare_ae_bands.py" --old "$1" --current "$2" --primary-manifest "$3" --repo "$ROOT" --output "$4" "${option[@]}"
    ;;
  bundle)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    "$PY" "$ROOT/bundle_results.py" --runs-root "$1" --out "$2"
    ;;
  *) usage; exit 2 ;;
esac
