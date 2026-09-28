#!/usr/bin/env bash
# Generality check: is the over-attribution finding specific to Gemma-3?
# Behaviour + external only. No SAE / causal arms: Gemma Scope has no Qwen SAEs, and nothing here
# is tuned for Qwen -- it reuses the identical prompt set, sign convention and scoring code, so
# there are no free parameters to select.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.hf_env; set +a
PY=./venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
O=runs/bridge2026/stage5_generality
DEV=${DEV:-cuda:0}

echo "===== Qwen dev readout ====="
$PY scripts/e1_model_readout.py --split dev --model "$M" --device "$DEV" \
    --batch-size 4 --out "$O/e1_model_readout_dev_qwen.csv"
echo "===== Qwen test readout ====="
$PY scripts/e1_model_readout.py --split test --model "$M" --device "$DEV" \
    --batch-size 4 --i-am-running-the-final-test --out "$O/e1_model_readout_test_qwen.csv"
echo "===== Qwen external (CCPE-M) ====="
$PY scripts/e4_external.py --model "$M" --device "$DEV" --batch-size 4 \
    --out "$O/e4_external_qwen.csv"
echo "===== Qwen context contrast ====="
$PY scripts/context_contrast.py --per-item "$O/e1_model_readout_test_qwen_per_item.csv" \
    --out "$O/context_contrast_test_qwen.csv"
echo "===== DONE ====="
