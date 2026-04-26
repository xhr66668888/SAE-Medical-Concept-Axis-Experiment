#!/usr/bin/env bash
# End-to-end pipeline for the medical SAE concept-axis experiment on Intel XPU.
#
# Each stage is a separate Python process so XPU memory is released between
# model / SAE loads.
#
# Common overrides:
#   DEVICE=cpu DTYPE=float32 PRESET=gemma3-270m-it-res bash scripts/run_xpu_experiment.sh
#   LAYER_SWEEP=all FOLDS=5 PATCHING=1 RUN_270M=1 bash scripts/run_xpu_experiment.sh
#   SKIP_STEERING=1 PATCHING=0 bash scripts/run_xpu_experiment.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN"
  echo "Please create/install the local venv first. See 使用说明.md."
  exit 1
fi

PRESET="${PRESET:-gemma3-1b-it-res}"
DEVICE="${DEVICE:-xpu}"
DTYPE="${DTYPE:-bfloat16}"

NULL_TRIALS="${NULL_TRIALS:-2000}"
FOLDS="${FOLDS:-1}"
BOOTSTRAP_TRIALS="${BOOTSTRAP_TRIALS:-1000}"
LAYER_SWEEP="${LAYER_SWEEP:-all}"

TARGET_COMP="${TARGET_COMP:-kidney}"
BASELINE_COMP="${BASELINE_COMP:-neurological}"
STEP4_LAYERS="${STEP4_LAYERS:-auto}"
STEP4_ALL_PAIRS="${STEP4_ALL_PAIRS:-0}"
STEP4_DRAW_GRAPH="${STEP4_DRAW_GRAPH:-0}"
RUN_65K="${RUN_65K:-0}"
SAE_65K_ID_FORMAT="${SAE_65K_ID_FORMAT:-layer_{layer}_width_65k_l0_small}"

SKIP_STEERING="${SKIP_STEERING:-0}"
STEERING_ALPHAS="${STEERING_ALPHAS:--3,-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2,3}"
STEERING_POSITIONS="${STEERING_POSITIONS:-all}"
STEERING_KEYWORD="${STEERING_KEYWORD:-diabetes}"
STEERING_COMP="${STEERING_COMP:-none}"
MAX_STEERING_PROMPTS="${MAX_STEERING_PROMPTS:-24}"

PATCHING="${PATCHING:-1}"
PATCHING_LAYERS="${PATCHING_LAYERS:-auto}"
PATCHING_COMPS="${PATCHING_COMPS:-$TARGET_COMP,$BASELINE_COMP}"
PATCHING_POSITIONS="${PATCHING_POSITIONS:--1,-2,-3,-4}"
PATCHING_SPLIT="${PATCHING_SPLIT:-all}"

RUN_270M="${RUN_270M:-0}"
DEVICE_270M="${DEVICE_270M:-$DEVICE}"
DTYPE_270M="${DTYPE_270M:-$DTYPE}"
NULL_TRIALS_270M="${NULL_TRIALS_270M:-$NULL_TRIALS}"
LAYER_SWEEP_270M="${LAYER_SWEEP_270M:-all}"
REPORT="${REPORT:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$PRESET" in
  gemma3-270m-it-res) DEFAULT_SMOKE_PRESET="gemma3-270m-it-res9" ;;
  gemma3-1b-it-res) DEFAULT_SMOKE_PRESET="gemma3-1b-it-res13" ;;
  gemma3-4b-it-res) DEFAULT_SMOKE_PRESET="gemma3-4b-it-res17" ;;
  gemma2-2b-res) DEFAULT_SMOKE_PRESET="gemma2-2b-res12" ;;
  *) DEFAULT_SMOKE_PRESET="gemma3-1b-it-res13" ;;
esac
SMOKE_PRESET="${SMOKE_PRESET:-$DEFAULT_SMOKE_PRESET}"

echo "== 0. Hardware check =="
"$PYTHON_BIN" scripts/check_hardware.py

if [[ "$DEVICE" == "xpu" ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
import sys
import torch

ok = hasattr(torch, "xpu") and torch.xpu.is_available()
if ok:
    print(f"XPU ready: {torch.xpu.get_device_name(0)}")
    sys.exit(0)
print("XPU is not available to PyTorch yet.")
print("Set DEVICE=cpu DTYPE=float32 bash scripts/run_xpu_experiment.sh to fall back to CPU.")
sys.exit(1)
PY
  then
    echo
    echo "Stopped before loading the model because PyTorch cannot see an XPU device."
    echo "Suggested CPU smoke test:"
    echo "  DEVICE=cpu DTYPE=float32 PRESET=gemma3-270m-it-res bash scripts/run_xpu_experiment.sh"
    exit 1
  fi
fi

echo
echo "== 1. Generate prompts =="
"$PYTHON_BIN" scripts/step2_generate_prompts.py

echo
echo "== 2. Step 1 MVP smoke test =="
"$PYTHON_BIN" scripts/step1_mvp.py \
  --preset "$SMOKE_PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

echo
echo "== 3. Step 3 all-layer axis sweep + DLA + logit lens =="
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
echo "== 4. Step 4 SAE feature candidates =="
STEP4_EXTRA=()
if [[ "$STEP4_ALL_PAIRS" == "1" ]]; then
  STEP4_EXTRA+=(--all-pairs)
fi
if [[ "$STEP4_DRAW_GRAPH" == "1" ]]; then
  STEP4_EXTRA+=(--draw-graph)
fi
"$PYTHON_BIN" scripts/step4_trace_circuit.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --layers "$STEP4_LAYERS" \
  --best-layers-json outputs/axis/best_layers.json \
  --target-complication "$TARGET_COMP" \
  --baseline-complication "$BASELINE_COMP" \
  --use-split train \
  --output-dir outputs/circuit \
  "${STEP4_EXTRA[@]}"

if [[ "$RUN_65K" == "1" ]]; then
  echo
  echo "== 4b. Optional 65k-width SAE check at the best layer =="
  BEST_LAYER="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

blob = json.loads(Path("outputs/axis/best_layers.json").read_text(encoding="utf-8"))
print(blob["best_layer"])
PY
)"
  "$PYTHON_BIN" scripts/step4_trace_circuit.py \
    --prompts data/diabetes_contrastive_prompts.csv \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --layers "$BEST_LAYER" \
    --sae-id-format "$SAE_65K_ID_FORMAT" \
    --target-complication "$TARGET_COMP" \
    --baseline-complication "$BASELINE_COMP" \
    --use-split train \
    --output-dir outputs/circuit_65k
fi

if [[ "$SKIP_STEERING" != "1" ]]; then
  echo
  echo "== 5. Step 5 causal steering with bootstrap CI =="
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
fi

if [[ "$PATCHING" == "1" ]]; then
  echo
  echo "== 6. Step 6 activation patching =="
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
  echo "== 6b. Optional 270M replay: Step 3 axis sweep =="
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

  if [[ "$SKIP_STEERING" != "1" ]]; then
    echo
    echo "== 6c. Optional 270M replay: Step 5 steering =="
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
fi

if [[ "$REPORT" == "1" ]]; then
  echo
  echo "== 7. Aggregate report =="
  "$PYTHON_BIN" scripts/step7_report.py \
    --output-root outputs \
    --report-md outputs/report.md \
    --report-html outputs/report.html
fi

echo
echo "== Done =="
echo "Axis outputs:      outputs/axis/"
echo "Circuit outputs:   outputs/circuit/"
if [[ "$SKIP_STEERING" != "1" ]]; then
  echo "Steering outputs:  outputs/steering/"
fi
if [[ "$PATCHING" == "1" ]]; then
  echo "Patching outputs:  outputs/patching/"
fi
if [[ "$RUN_270M" == "1" ]]; then
  echo "270M outputs:      outputs/270m/"
fi
if [[ "$REPORT" == "1" ]]; then
  echo "Report:            outputs/report.html"
fi
echo
echo "Quick reads:"
echo "  cat outputs/axis/axis_summary.txt"
if [[ "$STEP4_ALL_PAIRS" == "1" ]]; then
  echo "  ls outputs/circuit/circuit_route*.txt"
else
  echo "  cat outputs/circuit/circuit_route.txt"
fi
if [[ "$SKIP_STEERING" != "1" ]]; then
  echo "  cat outputs/steering/steering_summary.txt"
fi
if [[ "$PATCHING" == "1" ]]; then
  echo "  cat outputs/patching/patching_summary.txt"
fi
