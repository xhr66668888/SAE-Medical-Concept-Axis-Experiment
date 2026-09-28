#!/usr/bin/env bash
# Stage 4 - the ONE-SHOT test pass. Everything here reads the frozen configs and the frozen split.
# Nothing in this script selects a layer, a feature, a dose, a span or a threshold.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.hf_env; set +a
PY=./venv/bin/python
S4=runs/bridge2026/stage4
DEV=${DEV:-cuda:0}
mkdir -p "$S4" runs/bridge2026/logs

echo "=============== 1/9  text + surface baselines (test) ==============="
$PY scripts/e1_baselines.py --eval-split test --out "$S4/e1_baselines_test.csv"

echo "=============== 2/9  4B behavioural readout (test) ==============="
$PY scripts/e1_model_readout.py --split test --device "$DEV" \
    --i-am-running-the-final-test --out "$S4/e1_model_readout_test.csv"

echo "=============== 3/9  12B behavioural readout (test) ==============="
$PY scripts/e1_model_readout.py --split test --model google/gemma-3-12b-it --device "$DEV" \
    --batch-size 4 --i-am-running-the-final-test --out "$S4/e1_model_readout_test_12b.csv"

echo "=============== 4/9  4B representation (test) ==============="
$PY scripts/e2_representation.py --acts runs/bridge2026/cache/acts_4b.npz \
    --eval-split test --device "$DEV" --boot 2000 --null-dirs 150 --outdir "$S4"
$PY scripts/e2_incremental.py --acts runs/bridge2026/cache/acts_4b.npz \
    --eval-split test --out "$S4/e2_incremental_test.csv"

echo "=============== 5/9  12B representation (test) ==============="
$PY scripts/e2_representation.py --acts runs/bridge2026/cache/acts_12b.npz \
    --model google/gemma-3-12b-it --sae-layers 12,24,31,41 --eval-split test --device "$DEV" \
    --boot 2000 --null-dirs 100 --outdir "$S4/e2_12b"
$PY scripts/e2_incremental.py --acts runs/bridge2026/cache/acts_12b.npz \
    --eval-split test --out "$S4/e2_incremental_test_12b.csv"

echo "=============== 6/9  4B causal (test) ==============="
$PY scripts/e3_causal.py --config runs/bridge2026/frozen_config_4b.json --split test \
    --device "$DEV" --boot 2000 --i-am-running-the-final-test --out "$S4/e3_causal_test.csv"

echo "=============== 7/9  12B causal (test) ==============="
$PY scripts/e3_causal.py --config runs/bridge2026/frozen_config_12b.json --split test \
    --model google/gemma-3-12b-it --device "$DEV" --batch-size 4 --boot 2000 \
    --directions runs/bridge2026/stage3/e2_12b/e2_directions.npz \
    --i-am-running-the-final-test --out "$S4/e3_causal_test_12b.csv"

echo "=============== 8/9  external test (CCPE-M, 4B + 12B) ==============="
$PY scripts/e4_external.py --model google/gemma-3-4b-it --device "$DEV" \
    --config runs/bridge2026/frozen_config_4b.json \
    --directions runs/bridge2026/stage2/e2_directions.npz --out "$S4/e4_external_4b.csv"
$PY scripts/e4_external.py --model google/gemma-3-12b-it --device "$DEV" --batch-size 4 \
    --config runs/bridge2026/frozen_config_12b.json \
    --directions runs/bridge2026/stage3/e2_12b/e2_directions.npz --out "$S4/e4_external_12b.csv"

echo "=============== 9/9  free-form responses (test) ==============="
$PY scripts/e4_freeform.py --split test --config runs/bridge2026/frozen_config_4b.json \
    --device "$DEV" --limit 100 --i-am-running-the-final-test --out "$S4/e4_freeform_test.csv"

echo "=============== DONE ==============="
