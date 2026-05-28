# Medical Concept Axis Experiment

This repository studies whether small open language models encode medical concepts as measurable residual-stream directions. The design is inspired by *The Assistant Axis* but replaces persona contrasts with controlled ICD/CCS medical concept contrasts.

The experiment no longer treats a specific drug token as the primary endpoint. It first asks whether medical concepts have residual structure, then tests whether those directions causally affect concept-label readouts and whether Gemma Scope 2 SAEs expose candidate features aligned with the directions.

## Concept Contrasts

The default pipeline constructs five matched axes:

- `diabetes_subtype`: Type 1 diabetes vs Type 2 diabetes
- `complication_status`: diabetes with complications vs diabetes without complications
- `neoplasm_behavior`: malignant neoplasm vs benign neoplasm
- `infectious_etiology`: bacterial infection vs viral infection
- `disease_course`: acute condition vs chronic condition

Prompts are generated from ICD/CCS diagnosis text with held-out templates. Evaluation uses multi-token concept-label log-probability, not treatment-token prediction.

## Hardware Target

The local machine is CPU-first:

- Intel Core i9-14900
- 62 GiB system memory
- no CUDA/XPU device currently available to PyTorch

The main local model is `google/gemma-3-1b-it`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-torch-cpu.txt
.venv/bin/pip install -r requirements.txt
```

Gemma checkpoints may require an accepted Hugging Face license and `HF_TOKEN` in the environment.
For fine-grained Hugging Face tokens, enable read access to public gated repositories.

## Run

```bash
bash scripts/run_experiment.sh
```

Before the model-heavy stages, the runner checks whether the token can download `google/gemma-3-1b-it/config.json`.

The full run writes:

- `outputs/concept_prompts.csv`
- `outputs/axis/axis_summary.csv`
- `outputs/axis/layer_sweep.csv`
- `outputs/steering/steering_results.csv`
- `outputs/patching/patching_results.csv`
- `outputs/sae/sae_features.csv`
- `figures/*.png`
- `report.md`

## Pipeline

1. Generate matched concept prompts from ICD/CCS descriptions.
2. Capture hidden states and fit mean-difference residual axes for every concept and layer.
3. Evaluate axes with held-out templates, bootstrap intervals, random-direction nulls, and label-permutation nulls.
4. Steer activations along each fitted axis and measure concept-label log-probability shifts.
5. Patch matched negative-side residuals into positive-side prompts.
6. Trace Gemma Scope 2 SAE features aligned with the fitted axes.
7. Draw the concept-structure and mechanistic-circuit figures.
8. Build a Markdown report with figures.

## Notes

SAE feature IDs are candidate mechanistic evidence, not a complete circuit by themselves. A concept axis should be interpreted only when the held-out layer sweep, null tests, steering, patching, and SAE tracing agree.
