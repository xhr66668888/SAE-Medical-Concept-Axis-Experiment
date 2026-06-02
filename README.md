# Medical Concept Axis Experiment

This repository studies whether `google/gemma-3-4b-it` encodes ICD-9-CM CCS medical concepts as measurable residual-stream directions. The design is inspired by *The Assistant Axis* but replaces persona contrasts with controlled medical ontology contrasts.

The experiment no longer treats a specific drug token as the primary endpoint. It asks whether clinically meaningful CCS diagnosis groups have residual structure, then tests whether those directions causally affect concept-label readouts and whether Gemma Scope 2 SAEs expose candidate features aligned with the directions.

## Concept Contrasts

The primary pipeline uses AHRQ 2015 single-level CCS for ICD-9-CM diagnoses. `AppendixASingleDX.txt` defines the canonical CCS categories and ICD-9-CM code membership; `data/icd_diagnosis_ccs.csv` supplies diagnosis descriptions and hierarchy fields.

Default primary axes are code-system aware and must pass a minimum per-side diagnosis count. The diabetes axis is CCS-defined:

- `ccs_diabetes_complication`: CCS 50 diabetes with complications vs CCS 49 diabetes without complication
- Additional high-count ICD-9 CCS contrasts are generated from clinically meaningful CCS categories such as infection, neoplasm, and injury/poisoning families.
- Type 1 vs Type 2 diabetes is not treated as a CCS primary axis; if present, it is exploratory and labeled ICD-derived.

Prompts are generated from ICD-9 diagnosis text with held-out diagnosis rows and held-out templates. Evaluation uses multi-token concept-label log-probability, not treatment-token prediction.

## Model

The experiment is fixed to `google/gemma-3-4b-it`. The SAE tracing stage uses `gemma-scope-2-4b-it-res-all`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Gemma checkpoints may require an accepted Hugging Face license and `HF_TOKEN` in the environment.
For fine-grained Hugging Face tokens, enable read access to public gated repositories.

## Run

```bash
bash scripts/run_all.sh
```

The full run writes to `runs/gemma3_4b_ccs_icd9_full/` by default, uses description-level held-out diagnosis pairs, held-out templates, 5,000 bootstrap/null trials, 120 steering prompts per axis, prompt-token steering, and an activation-patching window around each best layer. If the old interrupted `runs/gemma3_4b/` directory is present, the runner archives it under `runs/archive/` before starting the new default run.

To continue after an interrupted stage without rerunning earlier stages, set `START_AT`, for example `START_AT=steering bash scripts/run_all.sh`.

The full run writes:

- `runs/gemma3_4b_ccs_icd9_full/outputs/concept_prompts.csv`
- `runs/gemma3_4b_ccs_icd9_full/outputs/axis/axis_summary.csv`
- `runs/gemma3_4b_ccs_icd9_full/outputs/axis/layer_sweep.csv`
- `runs/gemma3_4b_ccs_icd9_full/outputs/steering/steering_results.csv`
- `runs/gemma3_4b_ccs_icd9_full/outputs/patching/patching_results.csv`
- `runs/gemma3_4b_ccs_icd9_full/outputs/sae/sae_features.csv`
- `runs/gemma3_4b_ccs_icd9_full/figures/*.png`
- `runs/gemma3_4b_ccs_icd9_full/report.md`

## Pipeline

1. Parse AHRQ Appendix A and join canonical ICD-9-CM CCS categories to diagnosis descriptions.
2. Split by both diagnosis pair and prompt template, so held-out evaluation uses unseen descriptions and unseen prompt phrasings.
3. Report a lexical ontology baseline for keyword-visible contrasts.
4. Capture residual vectors with a resumable activation cache and fit mean-difference axes for every concept and layer.
5. Evaluate axes with held-out diagnosis/template combinations, bootstrap intervals, random-direction nulls, and label-permutation nulls.
6. Steer prompt-token activations along each fitted axis and measure concept-label log-probability shifts.
7. Patch matched negative-side prompt residuals into positive-side prompts across a best-layer window.
8. Trace Gemma Scope 2 SAE features aligned with the fitted axes.
9. Draw the concept-structure and mechanistic-circuit figures.
10. Build a Markdown report with figures.

## Validity Criteria

An axis is treated as primary evidence only when it is an ICD-9 CCS primary axis and held-out layer-sweep accuracy, permutation nulls, random-direction nulls, steering dose response, activation patching, and SAE tracing agree. The lexical baseline is reported separately to distinguish residual geometry from keyword visibility in the diagnosis text.
