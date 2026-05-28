#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
THREADS="${THREADS:-16}"
MODEL="${MODEL:-google/gemma-3-1b-it}"
MAX_PAIRS="${MAX_PAIRS:-40}"
LAYERS="${LAYERS:-all}"
NULL_TRIALS="${NULL_TRIALS:-1000}"
BOOTSTRAP_TRIALS="${BOOTSTRAP_TRIALS:-1000}"
MAX_STEERING="${MAX_STEERING:-24}"
MAX_PATCHING="${MAX_PATCHING:-12}"
RUN_SAE="${RUN_SAE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python)"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"

echo "== Hardware =="
"$PYTHON_BIN" scripts/check_hardware.py

echo
echo "== Hugging Face access =="
"$PYTHON_BIN" scripts/check_hf_access.py --model-name "$MODEL"

echo
echo "== Prompts =="
"$PYTHON_BIN" scripts/generate_prompts.py \
  --icd-csv data/icd_diagnosis_ccs.csv \
  --output outputs/concept_prompts.csv \
  --max-pairs-per-axis "$MAX_PAIRS"

echo
echo "== Axis sweep =="
"$PYTHON_BIN" scripts/fit_axes.py \
  --prompts outputs/concept_prompts.csv \
  --model-name "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --threads "$THREADS" \
  --layers "$LAYERS" \
  --null-trials "$NULL_TRIALS" \
  --bootstrap-trials "$BOOTSTRAP_TRIALS" \
  --output-dir outputs/axis \
  --figure-dir figures

echo
echo "== Steering =="
"$PYTHON_BIN" scripts/run_steering.py \
  --prompts outputs/concept_prompts.csv \
  --axis-dir outputs/axis \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --threads "$THREADS" \
  --max-prompts-per-axis "$MAX_STEERING" \
  --output-dir outputs/steering \
  --figure-dir figures

echo
echo "== Patching =="
"$PYTHON_BIN" scripts/run_patching.py \
  --prompts outputs/concept_prompts.csv \
  --axis-dir outputs/axis \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --threads "$THREADS" \
  --max-pairs-per-axis "$MAX_PATCHING" \
  --output-dir outputs/patching \
  --figure-dir figures

if [[ "$RUN_SAE" == "1" ]]; then
  echo
  echo "== SAE tracing =="
  "$PYTHON_BIN" scripts/trace_sae_features.py \
    --prompts outputs/concept_prompts.csv \
    --axis-dir outputs/axis \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --threads "$THREADS" \
    --output-dir outputs/sae \
    --figure-dir figures
fi

echo
echo "== Report figures =="
"$PYTHON_BIN" scripts/draw_concept_graph.py \
  --axis-summary outputs/axis/axis_summary.csv \
  --sae-features outputs/sae/sae_features.csv \
  --output figures/medical_concept_graph.png

"$PYTHON_BIN" scripts/draw_circuit_diagram.py \
  --axis-summary outputs/axis/axis_summary.csv \
  --steering-results outputs/steering/steering_results.csv \
  --sae-features outputs/sae/sae_features.csv \
  --output figures/mechanistic_circuit_diagram.png

echo
echo "== Report =="
"$PYTHON_BIN" scripts/build_report.py \
  --output-dir outputs \
  --report report.md

echo
echo "Done."
echo "Report: report.md"
