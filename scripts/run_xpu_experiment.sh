#!/usr/bin/env bash
# Clean end-to-end pipeline for the diabetes concept-axis experiment.
#
# Hardware target: Intel Lunar Lake XPU / ~15 GB unified memory.
# Main model: google/gemma-3-1b-it
# SAE: Gemma Scope 2 residual all-layer small SAE
#
# Every stage is a separate Python process so XPU memory is released between
# model / SAE loads.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN"
  echo "Create the local venv and install requirements first."
  exit 1
fi

PRESET="${PRESET:-gemma3-1b-it-res}"
DEVICE="${DEVICE:-xpu}"
DTYPE="${DTYPE:-bfloat16}"

LAYER_SWEEP="${LAYER_SWEEP:-all}"
FOLDS="${FOLDS:-1}"
NULL_TRIALS="${NULL_TRIALS:-2000}"
BOOTSTRAP_TRIALS="${BOOTSTRAP_TRIALS:-1000}"

SAE_LAYERS="${SAE_LAYERS:-auto}"
SAE_SPLIT="${SAE_SPLIT:-train}"
SAE_COMPLICATIONS="${SAE_COMPLICATIONS:-all}"

STEERING_ALPHAS="${STEERING_ALPHAS:--3,-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2,3}"
STEERING_POSITIONS="${STEERING_POSITIONS:-all}"
STEERING_KEYWORD="${STEERING_KEYWORD:-diabetes}"
STEERING_COMP="${STEERING_COMP:-none}"
MAX_STEERING_PROMPTS="${MAX_STEERING_PROMPTS:-24}"

PATCHING="${PATCHING:-1}"
PATCHING_LAYERS="${PATCHING_LAYERS:-auto}"
PATCHING_COMPS="${PATCHING_COMPS:-kidney,neurological}"
PATCHING_POSITIONS="${PATCHING_POSITIONS:--1,-2,-3,-4}"
PATCHING_SPLIT="${PATCHING_SPLIT:-all}"

RUN_270M="${RUN_270M:-0}"
DEVICE_270M="${DEVICE_270M:-$DEVICE}"
DTYPE_270M="${DTYPE_270M:-$DTYPE}"
LAYER_SWEEP_270M="${LAYER_SWEEP_270M:-all}"
NULL_TRIALS_270M="${NULL_TRIALS_270M:-$NULL_TRIALS}"

CIRCUIT_DIAGRAM="${CIRCUIT_DIAGRAM:-1}"
REPORT="${REPORT:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$PRESET" in
  gemma3-270m-it-res) SMOKE_PRESET="${SMOKE_PRESET:-gemma3-270m-it-res9}" ;;
  gemma3-1b-it-res) SMOKE_PRESET="${SMOKE_PRESET:-gemma3-1b-it-res13}" ;;
  *)
    echo "Unsupported PRESET=$PRESET"
    echo "Supported: gemma3-1b-it-res, gemma3-270m-it-res"
    exit 1
    ;;
esac

echo "== 0. Hardware check =="
"$PYTHON_BIN" scripts/check_hardware.py

if [[ "$DEVICE" == "xpu" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

ok = hasattr(torch, "xpu") and torch.xpu.is_available()
if ok:
    print(f"XPU ready: {torch.xpu.get_device_name(0)}")
    sys.exit(0)
print("XPU is not available to PyTorch. Use DEVICE=cpu DTYPE=float32 for CPU smoke tests.")
sys.exit(1)
PY
fi

echo
echo "== 1. Generate prompts =="
"$PYTHON_BIN" scripts/step2_generate_prompts.py

echo
echo "== 2. Model + SAE smoke test =="
"$PYTHON_BIN" scripts/step1_mvp.py \
  --preset "$SMOKE_PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

echo
echo "== 3. All-layer concept-axis sweep =="
"$PYTHON_BIN" scripts/step3_concept_axis.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --layers "$LAYER_SWEEP" \
  --folds "$FOLDS" \
  --null-trials "$NULL_TRIALS" \
  --bootstrap-trials "$BOOTSTRAP_TRIALS" \
  --output-dir outputs/axis

echo
echo "== 4. Axis-aligned Gemma Scope SAE tracing =="
SAE_EXTRA=()
if [[ -n "${MAX_AXIS_SAE_PAIRS:-}" ]]; then
  SAE_EXTRA+=(--max-pairs "$MAX_AXIS_SAE_PAIRS")
fi
"$PYTHON_BIN" scripts/step4_axis_sae.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --layers "$SAE_LAYERS" \
  --best-layers-json outputs/axis/best_layers.json \
  --axes-path outputs/axis/concept_axes_all_layers.pt \
  --axis-path outputs/axis/concept_axis.pt \
  --use-split "$SAE_SPLIT" \
  --complications "$SAE_COMPLICATIONS" \
  --bootstrap-trials "$BOOTSTRAP_TRIALS" \
  --output-dir outputs/circuit_axis \
  "${SAE_EXTRA[@]}"

echo
echo "== 5. Causal steering =="
"$PYTHON_BIN" scripts/step5_steering.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --axis-path outputs/axis/concept_axis.pt \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --use-split test \
  --filter-complication "$STEERING_COMP" \
  --max-prompts "$MAX_STEERING_PROMPTS" \
  --alphas="$STEERING_ALPHAS" \
  --positions "$STEERING_POSITIONS" \
  --keyword "$STEERING_KEYWORD" \
  --bootstrap-trials "$BOOTSTRAP_TRIALS" \
  --output-dir outputs/steering

if [[ "$PATCHING" == "1" ]]; then
  echo
  echo "== 6. Activation patching =="
  PATCHING_EXTRA=()
  if [[ -n "${MAX_PATCHING_PAIRS:-}" ]]; then
    PATCHING_EXTRA+=(--max-pairs-per-complication "$MAX_PATCHING_PAIRS")
  fi
  "$PYTHON_BIN" scripts/step6_patching.py \
    --prompts data/diabetes_contrastive_prompts.csv \
    --best-layers-json outputs/axis/best_layers.json \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --layers "$PATCHING_LAYERS" \
    --complications "$PATCHING_COMPS" \
    --positions="$PATCHING_POSITIONS" \
    --use-split "$PATCHING_SPLIT" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --output-dir outputs/patching \
    "${PATCHING_EXTRA[@]}"
fi

if [[ "$RUN_270M" == "1" ]]; then
  echo
  echo "== 7. 270M replay: axis sweep =="
  "$PYTHON_BIN" scripts/step3_concept_axis.py \
    --prompts data/diabetes_contrastive_prompts.csv \
    --preset gemma3-270m-it-res \
    --device "$DEVICE_270M" \
    --dtype "$DTYPE_270M" \
    --layers "$LAYER_SWEEP_270M" \
    --folds "$FOLDS" \
    --null-trials "$NULL_TRIALS_270M" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --output-dir outputs/270m/axis

  echo
  echo "== 8. 270M replay: steering =="
  "$PYTHON_BIN" scripts/step5_steering.py \
    --prompts data/diabetes_contrastive_prompts.csv \
    --axis-path outputs/270m/axis/concept_axis.pt \
    --preset gemma3-270m-it-res \
    --device "$DEVICE_270M" \
    --dtype "$DTYPE_270M" \
    --use-split test \
    --filter-complication "$STEERING_COMP" \
    --max-prompts "$MAX_STEERING_PROMPTS" \
    --alphas="$STEERING_ALPHAS" \
    --positions "$STEERING_POSITIONS" \
    --keyword "$STEERING_KEYWORD" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --output-dir outputs/270m/steering
fi

if [[ "$CIRCUIT_DIAGRAM" == "1" ]]; then
  echo
  echo "== 9. Circuit diagram =="
  "$PYTHON_BIN" scripts/step8_circuit_diagram.py \
    --features-csv outputs/circuit_axis/axis_sae_features.csv \
    --layer-summary-csv outputs/circuit_axis/axis_sae_layer_summary.csv \
    --axis-summary outputs/axis/axis_summary.txt \
    --patching-summary outputs/patching/patching_summary.txt \
    --output-dir outputs/circuit_diagram
fi

if [[ "$REPORT" == "1" ]]; then
  echo
  echo "== 10. Aggregate report =="
  "$PYTHON_BIN" scripts/step7_report.py \
    --output-root outputs \
    --report-md outputs/report.md \
    --report-html outputs/report.html
fi

echo
echo "== Done =="
echo "Axis:             outputs/axis/axis_summary.txt"
echo "Axis SAE:         outputs/circuit_axis/axis_sae_summary.txt"
echo "Steering:         outputs/steering/steering_summary.txt"
if [[ "$PATCHING" == "1" ]]; then
  echo "Patching:         outputs/patching/patching_summary.txt"
fi
if [[ "$CIRCUIT_DIAGRAM" == "1" ]]; then
  echo "Circuit diagram:  outputs/circuit_diagram/medical_circuit_diagram.png"
fi
if [[ "$REPORT" == "1" ]]; then
  echo "Report:           outputs/report.html"
fi
