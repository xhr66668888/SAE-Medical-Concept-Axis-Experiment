# Medical SAE Concept-Axis Experiments

This project explores whether open language models represent simple medical
concepts in linearly separable and traceable ways. It uses Gemma 3 with Gemma
Scope 2 sparse autoencoders (SAEs), ICD diagnosis codes, and ATC drug codes to
study Type 1 vs Type 2 diabetes prompts and complication-specific contexts such
as kidney or neurological complications.

The goal is mechanistic interpretability, not clinical decision support. The
scripts inspect model activations, decompose them through SAEs, and visualize
candidate feature routes. See [实验介绍.md](实验介绍.md) for the Chinese experiment
background and research motivation.

## What The Pipeline Does

1. **Step 1: MVP smoke test**  
   Load one Gemma model and one residual-stream SAE, capture one residual
   activation, encode it with the SAE, and print the top active feature IDs.

2. **Step 2: Prompt generation**  
   Generate contrastive diabetes prompts from `data/icd_diagnosis_ccs.csv` and
   `data/atc_drug_hierarchy.csv`.

3. **Step 3: Concept axis**  
   Compute a Type 1 minus Type 2 diabetes activation direction and draw a PCA
   plot of the captured residual activations.

4. **Step 4: Feature route graph**  
   Compare target vs baseline complication prompts, extract differential SAE
   features across layers, and draw a heuristic networkx feature-route graph.

## Repository Layout

```text
scripts/
  check_hardware.py          Hardware and PyTorch device diagnostics
  step1_mvp.py               Model + single SAE smoke test
  step2_generate_prompts.py  ICD/ATC contrastive prompt generation
  step3_concept_axis.py      Type 1 vs Type 2 activation axis + PCA
  step4_trace_circuit.py     Cross-layer differential SAE feature graph
sae_med/
  data_utils.py
  model_utils.py
data/
  atc_drug_hierarchy.csv
  diabetes_contrastive_prompts.csv
  icd_diagnosis_ccs.csv
requirements.txt            Generic CPU/CUDA/MPS dependency list
requirements-intel.txt      Non-torch dependencies for Intel XPU installs
使用说明.md                  Local runbook for this specific Intel machine
实验介绍.md                  Chinese experiment introduction
```

## Model Presets

The default preset is:

```text
google/gemma-3-1b-it + gemma-scope-2-1b-it-res
```

Available presets:

- `gemma3-270m-it-res`: smallest Gemma Scope 2 route, useful for CPU smoke tests.
- `gemma3-1b-it-res`: default and recommended starting point.
- `gemma3-4b-it-res`: larger local experiment; reduce prompts/layers first.
- `gemma2-2b-res`: legacy Gemma 2 2B fallback, not Gemma Scope 2.

Gemma checkpoints usually require accepting the Hugging Face license and
authenticating with `huggingface-cli login` or `HF_TOKEN`.

## Installation On Another Machine

Use Python 3.10-3.12. Python 3.11 is a safe default.

### Generic CPU/CUDA/MPS Setup

```bash
git clone <this-repo>
cd sae
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
huggingface-cli login
```

For CUDA machines, install the PyTorch build appropriate for your CUDA version
first if the default `pip install torch` is not what you want. Then install the
remaining dependencies from `requirements.txt`.

### Intel GPU / XPU Setup

For Intel Arc / integrated Arc GPUs, install the XPU PyTorch wheel first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements-intel.txt
huggingface-cli login
```

The user must be able to access `/dev/dri/renderD*`. On many Linux systems:

```bash
sudo usermod -aG render,video $USER
reboot
```

Then verify:

```bash
python scripts/check_hardware.py
```

## Quick Start

Generate prompts:

```bash
python scripts/step2_generate_prompts.py
```

Run the MVP smoke test:

```bash
python scripts/step1_mvp.py
```

Run the concept-axis experiment:

```bash
python scripts/step3_concept_axis.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --output-dir outputs/axis
```

Run the cross-layer feature route experiment:

```bash
python scripts/step4_trace_circuit.py \
  --prompts data/diabetes_contrastive_prompts.csv \
  --output-dir outputs/circuit
```

For Intel XPU, add:

```bash
--device xpu --dtype bfloat16
```

For CPU smoke tests, use the small preset:

```bash
python scripts/step1_mvp.py \
  --preset gemma3-270m-it-res9 \
  --device cpu \
  --dtype float32
```

## Outputs

Step 3 writes:

- `outputs/axis/concept_axis.pt`
- `outputs/axis/axis_results.csv`
- `outputs/axis/concept_axis_pca.png`
- `outputs/axis/axis_summary.txt`

Step 4 writes:

- `outputs/circuit/circuit_route.txt`
- `outputs/circuit/circuit_features.csv`
- `outputs/circuit/circuit_edges.csv`
- `outputs/circuit/circuit_graph.png`
- `outputs/circuit/circuit_graph.graphml`

## Notes And Limits

SAE feature IDs are not automatically human-readable concepts. The route graph
uses decoder-direction cosine similarity as a heuristic edge score; it is not a
causal proof. For stronger claims, add activation patching, feature ablation, or
other causal interventions.

The generated prompts and model outputs are for interpretability research only
and must not be used as medical advice.
