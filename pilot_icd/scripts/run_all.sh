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
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python)"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/experiment.yaml}"
cfg() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$1" "$2" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

try:
    import yaml
except ModuleNotFoundError:
    print(default)
    raise SystemExit(0)

if not path.exists():
    print(default)
    raise SystemExit(0)

data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
value = data
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        print(default)
        raise SystemExit(0)
    value = value[part]

if value is None:
    print(default)
elif isinstance(value, bool):
    print("1" if value else "0")
elif isinstance(value, list):
    print(",".join(str(item) for item in value))
else:
    print(value)
PY
}

FIXED_MODEL="$(cfg model google/gemma-3-4b-it)"
if [[ -n "${MODEL:-}" && "${MODEL}" != "$FIXED_MODEL" ]]; then
  echo "This experiment is configured for $FIXED_MODEL." >&2
  exit 2
fi
MODEL="$FIXED_MODEL"

RUN_NAME="${RUN_NAME:-$(cfg run.name gemma3_4b_ccs_icd9_full)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/outputs}"
FIGURE_DIR="${FIGURE_DIR:-$RUN_DIR/figures}"
REPORT_PATH="${REPORT_PATH:-$RUN_DIR/report.md}"
PROMPTS_PATH="${PROMPTS_PATH:-$OUTPUT_DIR/concept_prompts.csv}"

DEVICE="${DEVICE:-$(cfg runtime.device cpu)}"
DTYPE="${DTYPE:-$(cfg runtime.dtype float32)}"
THREADS="${THREADS:-$(cfg runtime.threads 16)}"

ICD_CSV="${ICD_CSV:-$(cfg data.icd_csv data/icd_diagnosis_ccs.csv)}"
CCS_APPENDIX="${CCS_APPENDIX:-$(cfg data.ccs_appendix AppendixASingleDX.txt)}"
MAX_PAIRS="${MAX_PAIRS:-$(cfg data.max_pairs_per_axis 120)}"
MIN_PRIMARY_PAIRS="${MIN_PRIMARY_PAIRS:-$(cfg data.min_primary_pairs_per_side 30)}"
HELDOUT_PAIR_FRACTION="${HELDOUT_PAIR_FRACTION:-$(cfg data.heldout_pair_fraction 0.25)}"
HELDOUT_TEMPLATE_IDS="${HELDOUT_TEMPLATE_IDS:-$(cfg data.heldout_template_ids 8,9,10)}"
LAYERS="${LAYERS:-$(cfg axis.layers all)}"
FIT_SPLITS="${FIT_SPLITS:-$(cfg axis.fit_splits train,test)}"
AXIS_CACHE_DIR="${AXIS_CACHE_DIR:-$RUN_DIR/$(cfg axis.activation_cache cache/axis_activations)}"
NULL_TRIALS="${NULL_TRIALS:-$(cfg axis.null_trials 5000)}"
BOOTSTRAP_TRIALS="${BOOTSTRAP_TRIALS:-$(cfg axis.bootstrap_trials 5000)}"

DEFAULT_STEERING_ALPHAS="-6,-4,-2,-1,0,1,2,4,6"
STEERING_ALPHAS="${STEERING_ALPHAS:-$(cfg steering.alphas "$DEFAULT_STEERING_ALPHAS")}"
if [[ -z "${STEERING_ALPHAS//[[:space:]]/}" ]]; then
  STEERING_ALPHAS="$DEFAULT_STEERING_ALPHAS"
fi
STEERING_POSITIONS="${STEERING_POSITIONS:-$(cfg steering.positions prompt_all)}"
MAX_STEERING="${MAX_STEERING:-$(cfg steering.max_prompts_per_axis 120)}"

PATCH_POSITIONS="${PATCH_POSITIONS:-$(cfg patching.positions -1,-2,-3,-4)}"
PATCH_LAYERS="${PATCH_LAYERS:-$(cfg patching.layers window:2)}"
MAX_PATCHING="${MAX_PATCHING:-$(cfg patching.max_pairs_per_axis 60)}"

RUN_SAE="${RUN_SAE:-1}"
SAE_RELEASE="${SAE_RELEASE:-$(cfg sae.release gemma-scope-2-4b-it-res-all)}"
SAE_ID_FORMAT="${SAE_ID_FORMAT:-$(cfg sae.sae_id_format layer_{layer}_width_16k_l0_small)}"
SAE_TOP_K="${SAE_TOP_K:-$(cfg sae.top_k 30)}"

RUNTIME_ARGS=()
if [[ -n "$DEVICE" && "$DEVICE" != "auto" ]]; then
  RUNTIME_ARGS+=(--device "$DEVICE")
fi
if [[ -n "$DTYPE" && "$DTYPE" != "auto" ]]; then
  RUNTIME_ARGS+=(--dtype "$DTYPE")
fi
if [[ -n "$THREADS" && "$THREADS" != "auto" ]]; then
  RUNTIME_ARGS+=(--threads "$THREADS")
fi
STEERING_ALPHA_ARGS=("--alphas=$STEERING_ALPHAS")
PATCH_POSITION_ARGS=("--positions=$PATCH_POSITIONS")

ARCHIVE_INTERRUPTED="${ARCHIVE_INTERRUPTED:-$(cfg run.archive_interrupted_default_run 1)}"
OLD_DEFAULT_RUN="$ROOT_DIR/runs/gemma3_4b"
if [[ "$ARCHIVE_INTERRUPTED" == "1" && "$RUN_NAME" != "gemma3_4b" && -d "$OLD_DEFAULT_RUN" ]]; then
  ARCHIVE_DIR="$ROOT_DIR/runs/archive/gemma3_4b_interrupted_20260529"
  if [[ ! -e "$ARCHIVE_DIR" ]]; then
    mkdir -p "$(dirname "$ARCHIVE_DIR")"
    mv "$OLD_DEFAULT_RUN" "$ARCHIVE_DIR"
    echo "Archived interrupted run: $ARCHIVE_DIR"
  fi
fi

mkdir -p "$OUTPUT_DIR" "$FIGURE_DIR"

START_AT="${START_AT:-preflight}"
stage_index() {
  case "$1" in
    preflight) echo 0 ;;
    hf | huggingface) echo 1 ;;
    prompts) echo 2 ;;
    lexical) echo 3 ;;
    axis) echo 4 ;;
    steering) echo 5 ;;
    patching) echo 6 ;;
    sae) echo 7 ;;
    figures | report_figures) echo 8 ;;
    report) echo 9 ;;
    *) return 1 ;;
  esac
}
if ! START_INDEX="$(stage_index "$START_AT")"; then
  echo "Unknown START_AT stage: $START_AT" >&2
  echo "Valid stages: preflight, hf, prompts, lexical, axis, steering, patching, sae, figures, report" >&2
  exit 2
fi
should_run_stage() {
  local stage="$1"
  local index
  index="$(stage_index "$stage")"
  [[ "$index" -ge "$START_INDEX" ]]
}

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -n "$THREADS" && "$THREADS" != "auto" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
fi

echo "== Medical concept axis experiment =="
echo "run dir: $RUN_DIR"
echo "model  : $MODEL"
echo "config : $CONFIG_PATH"
echo "device : ${DEVICE:-auto}"
echo "dtype  : ${DTYPE:-auto}"
echo "threads: ${THREADS:-auto}"
echo "source : ICD-9-CM CCS Appendix A"
echo "axis cache: $AXIS_CACHE_DIR"
echo "start at: $START_AT"

if should_run_stage preflight; then
  echo
  echo "== Local preflight =="
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

try:
    import torch
except Exception as exc:
    print(f"torch unavailable: {exc}")
else:
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.cuda.is_available()}")
    print(f"xpu : {getattr(torch, 'xpu', None) is not None and torch.xpu.is_available()}")
    print(f"threads: {torch.get_num_threads()}")

home = Path.home()
hf = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface"))
print(f"HF cache: {hf}")
usage = shutil.disk_usage(Path.cwd())
print(f"cwd free GiB: {usage.free / 1024**3:.1f}")
PY
fi

if should_run_stage hf; then
  echo
  echo "== Hugging Face access =="
  "$PYTHON_BIN" scripts/check_hf_access.py --model-name "$MODEL"
fi

if should_run_stage prompts; then
  echo
  echo "== Prompts =="
  "$PYTHON_BIN" scripts/generate_prompts.py \
    --icd-csv "$ICD_CSV" \
    --ccs-appendix "$CCS_APPENDIX" \
    --output "$PROMPTS_PATH" \
    --max-pairs-per-axis "$MAX_PAIRS" \
    --min-primary-pairs-per-side "$MIN_PRIMARY_PAIRS" \
    --heldout-template-ids "$HELDOUT_TEMPLATE_IDS" \
    --heldout-pair-fraction "$HELDOUT_PAIR_FRACTION"
fi

if should_run_stage lexical; then
  echo
  echo "== Lexical baseline =="
  "$PYTHON_BIN" scripts/run_lexical_baseline.py \
    --prompts "$PROMPTS_PATH" \
    --output "$OUTPUT_DIR/lexical_baseline.csv"
fi

if should_run_stage axis; then
  echo
  echo "== Axis sweep =="
  "$PYTHON_BIN" scripts/fit_axes.py \
    --prompts "$PROMPTS_PATH" \
    --model-name "$MODEL" \
    "${RUNTIME_ARGS[@]}" \
    --layers "$LAYERS" \
    --null-trials "$NULL_TRIALS" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --fit-splits "$FIT_SPLITS" \
    --activation-cache-dir "$AXIS_CACHE_DIR" \
    --output-dir "$OUTPUT_DIR/axis" \
    --figure-dir "$FIGURE_DIR"
fi

if should_run_stage steering; then
  echo
  echo "== Steering =="
  "$PYTHON_BIN" scripts/run_steering.py \
    --prompts "$PROMPTS_PATH" \
    --axis-dir "$OUTPUT_DIR/axis" \
    --model-name "$MODEL" \
    "${RUNTIME_ARGS[@]}" \
    "${STEERING_ALPHA_ARGS[@]}" \
    --positions "$STEERING_POSITIONS" \
    --max-prompts-per-axis "$MAX_STEERING" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --output-dir "$OUTPUT_DIR/steering" \
    --figure-dir "$FIGURE_DIR"
fi

if should_run_stage patching; then
  echo
  echo "== Patching =="
  "$PYTHON_BIN" scripts/run_patching.py \
    --prompts "$PROMPTS_PATH" \
    --axis-dir "$OUTPUT_DIR/axis" \
    --model-name "$MODEL" \
    "${RUNTIME_ARGS[@]}" \
    "${PATCH_POSITION_ARGS[@]}" \
    --layers "$PATCH_LAYERS" \
    --max-pairs-per-axis "$MAX_PATCHING" \
    --bootstrap-trials "$BOOTSTRAP_TRIALS" \
    --output-dir "$OUTPUT_DIR/patching" \
    --figure-dir "$FIGURE_DIR"
fi

if should_run_stage sae && [[ "$RUN_SAE" == "1" ]]; then
  echo
  echo "== SAE tracing =="
  "$PYTHON_BIN" scripts/trace_sae_features.py \
    --prompts "$PROMPTS_PATH" \
    --axis-dir "$OUTPUT_DIR/axis" \
    --model-name "$MODEL" \
    --sae-release "$SAE_RELEASE" \
    --sae-id-format "$SAE_ID_FORMAT" \
    "${RUNTIME_ARGS[@]}" \
    --max-pairs-per-axis "$MAX_PATCHING" \
    --top-k "$SAE_TOP_K" \
    --output-dir "$OUTPUT_DIR/sae" \
    --figure-dir "$FIGURE_DIR"
fi

if should_run_stage figures; then
  echo
  echo "== Report figures =="
  "$PYTHON_BIN" scripts/draw_concept_graph.py \
    --axis-summary "$OUTPUT_DIR/axis/axis_summary.csv" \
    --sae-features "$OUTPUT_DIR/sae/sae_features.csv" \
    --output "$FIGURE_DIR/medical_concept_graph.png"

  "$PYTHON_BIN" scripts/draw_circuit_diagram.py \
    --axis-summary "$OUTPUT_DIR/axis/axis_summary.csv" \
    --steering-results "$OUTPUT_DIR/steering/steering_results.csv" \
    --sae-features "$OUTPUT_DIR/sae/sae_features.csv" \
    --output "$FIGURE_DIR/mechanistic_circuit_diagram.png"
fi

if should_run_stage report; then
  echo
  echo "== Report =="
  "$PYTHON_BIN" scripts/build_report.py \
    --output-dir "$OUTPUT_DIR" \
    --figure-dir "$FIGURE_DIR" \
    --report "$REPORT_PATH"
fi

echo
echo "Done."
echo "Run directory: $RUN_DIR"
echo "Report: $REPORT_PATH"
