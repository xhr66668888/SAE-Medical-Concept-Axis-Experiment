# Medical Concept Axis / 医疗概念轴实验

## 中文版

本项目做一个受 *The Assistant Axis* 方法启发的局部复现实验：

> 在 `google/gemma-3-1b-it` 中，是否存在一个可测量、可干预、可由 Gemma Scope SAE 候选特征解释的医疗概念轴：  
> **Type 1 diabetes -> insulin** vs **Type 2 diabetes -> metformin**？

结论先说清楚：这不是对原论文的完整复现。原论文研究的是 assistant persona axis；这里研究的是一个受控医疗概念轴。当前结果支持一个较强的 residual-stream 层面发现，并给出一个可放入报告/论文的候选 SAE 电路示意图。SAE 部分应表述为 candidate mechanism，而不是完整机制证明。

### 当前最终结果

最终结果来自一次 clean run，而不是烟测输出。

主模型：

- Model: `google/gemma-3-1b-it`
- SAE: `gemma-scope-2-1b-it-res-all`
- Prompt set: 480 条 Type 1 / Type 2 diabetes matched prompts
- Type 1 readout token: `" insulin"` id `28933`
- Type 2 readout token: `" metformin"` id `188881`

Residual concept axis:

| Metric | Result |
|---|---:|
| Best layer | 21 |
| Candidate layers | 5, 19, 20, 21, 22, 23 |
| Held-out accuracy at layer 21 | 0.657 |
| 95% bootstrap CI | [0.58, 0.74] |
| Shuffled-label null mean | 0.528 |
| Null p-value | 0.011 |
| Direct logit attribution | +0.192 |
| Logit lens difference | +80.719 |

Causal steering:

- Hook: `blocks.21.hook_resid_post`
- Positions: all token positions
- Evaluation prompts: 24 held-out `none`-complication prompts
- Result: monotone increasing steering curve
- Mean delta logit difference moves from `-4.277` at alpha `-3` to `+4.621` at alpha `+3`

Activation patching:

- Matched Type 1 / Type 2 pairs: 96
- Complications: `kidney`, `neurological`
- Layers: 5, 19, 20, 21, 22, 23
- Positions: -1, -2, -3, -4
- Best cell: layer 21, position -1
- Normalized patching score: `0.9955`
- 95% bootstrap CI: `[0.9751, 1.0148]`

SAE candidate circuit:

- Best SAE layer by top feature contribution: 21
- Matched pairs used for SAE tracing: 122 train pairs across all complications
- Strong axis-aligned features appear around layers 20-23, with layer 21 aligned with the residual-axis and patching peak
- Top layer-21 robust candidates include features such as `F521` and `F3330`; exact IDs and scores are in `outputs/circuit_axis/axis_sae_summary.txt`

270M replay:

- Model: `google/gemma-3-270m-it`
- Best layer: 12
- Held-out accuracy: 0.707
- Null mean: 0.545
- Null p-value: 0.013
- Steering is also monotone increasing, from `-2.005` at alpha `-3` to `+2.005` at alpha `+3`

Interpretation:

- The residual-stream concept-axis result is strong.
- Steering and activation patching both support a causal role for the layer-21 axis in the 1B model.
- The SAE tracing gives readable candidate features, but feature ablation would be needed before claiming a complete SAE-level mechanism.
- This is enough for a single-concept medical-axis case study; it is not enough to claim that all medical concepts share one universal circuit.

### Pipeline

The runner keeps each model/SAE-heavy stage in a separate Python process so Intel XPU memory is released between stages.

```text
0. scripts/check_hardware.py
   Check CPU, memory, Intel GPU, and PyTorch XPU availability.

1. scripts/step2_generate_prompts.py
   Generate 480 matched diabetes prompts.

2. scripts/step1_mvp.py
   Smoke-test Gemma + one Gemma Scope SAE.

3. scripts/step3_concept_axis.py
   Sweep residual layers, fit the Type1-vs-Type2 concept axis, and report
   held-out accuracy, bootstrap CI, shuffled-label null, DLA, and logit lens.

4. scripts/step4_axis_sae.py
   Trace Gemma Scope SAE features aligned with the learned residual axis.

5. scripts/step5_steering.py
   Add alpha * axis during forward passes and measure
   logit(insulin) - logit(metformin).

6. scripts/step6_patching.py
   Patch real Type2 residual activations into matched Type1 prompts.

7. scripts/step8_circuit_diagram.py
   Draw a pure black-and-white candidate circuit schematic.

8. scripts/step7_report.py
   Build `outputs/report.md` and `outputs/report.html`.
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repository is tuned for the local Intel Lunar Lake XPU environment. The default runner uses:

```bash
DEVICE=xpu
DTYPE=bfloat16
```

### Run

Full clean run with 1B main model and 270M replay:

```bash
rm -rf outputs/
LAYER_SWEEP=all \
PATCHING=1 \
RUN_270M=1 \
bash scripts/run_xpu_experiment.sh
```

Main 1B run only:

```bash
rm -rf outputs/
LAYER_SWEEP=all \
PATCHING=1 \
RUN_270M=0 \
bash scripts/run_xpu_experiment.sh
```

Fast smoke run:

```bash
rm -rf outputs/
PRESET=gemma3-270m-it-res \
LAYER_SWEEP=4 \
SAE_LAYERS=4 \
PATCHING_LAYERS=4 \
MAX_AXIS_SAE_PAIRS=8 \
MAX_PATCHING_PAIRS=1 \
MAX_STEERING_PROMPTS=2 \
NULL_TRIALS=5 \
BOOTSTRAP_TRIALS=10 \
RUN_270M=0 \
bash scripts/run_xpu_experiment.sh
```

Important: smoke-run outputs are only for checking that the code works. Delete `outputs/` before a real run.

### Main Outputs

Read these summaries first:

```bash
cat outputs/axis/axis_summary.txt
cat outputs/circuit_axis/axis_sae_summary.txt
cat outputs/steering/steering_summary.txt
cat outputs/patching/patching_summary.txt
```

Report:

```text
outputs/report.md
outputs/report.html
```

Figures:

```text
outputs/axis/accuracy_by_layer.png
outputs/axis/dla_and_logit_lens.png
outputs/circuit_axis/axis_sae_contributions_by_layer.png
outputs/circuit_axis/axis_sae_top_features.png
outputs/steering/steering_curve.png
outputs/patching/patching_heatmap.png
outputs/circuit_diagram/medical_circuit_diagram.png
outputs/circuit_diagram/medical_circuit_diagram.svg
```

### How To Phrase The Claim

Good:

- "Gemma 3 1B contains a measurable Type1/Type2 diabetes residual-stream direction around layer 21."
- "The direction causally affects the insulin-vs-metformin readout under steering and activation patching."
- "Gemma Scope SAE features provide candidate components for a readable circuit."

Too strong:

- "This fully reproduces The Assistant Axis."
- "The SAE features are the complete mechanism."
- "All medical concepts share this circuit."

---

## English Version

This repository runs a focused interpretability experiment inspired by the method pattern of *The Assistant Axis*:

> Does `google/gemma-3-1b-it` contain a measurable, steerable, and partially SAE-readable medical concept axis for  
> **Type 1 diabetes -> insulin** versus **Type 2 diabetes -> metformin**?

This is not a full reproduction of the original paper. The paper studies an assistant-persona axis; this project studies one controlled medical concept axis. The current run supports a strong residual-stream finding and a readable candidate SAE circuit. The SAE part should be described as a candidate mechanism, not as a complete mechanistic proof.

### Final Run Results

These numbers come from a clean full run, not from smoke-test outputs.

Main model:

- Model: `google/gemma-3-1b-it`
- SAE: `gemma-scope-2-1b-it-res-all`
- Prompt set: 480 matched Type 1 / Type 2 diabetes prompts
- Type 1 readout token: `" insulin"` id `28933`
- Type 2 readout token: `" metformin"` id `188881`

Residual concept axis:

| Metric | Result |
|---|---:|
| Best layer | 21 |
| Candidate layers | 5, 19, 20, 21, 22, 23 |
| Held-out accuracy at layer 21 | 0.657 |
| 95% bootstrap CI | [0.58, 0.74] |
| Shuffled-label null mean | 0.528 |
| Null p-value | 0.011 |
| Direct logit attribution | +0.192 |
| Logit-lens difference | +80.719 |

Causal steering:

- Hook: `blocks.21.hook_resid_post`
- Positions: all token positions
- Evaluation prompts: 24 held-out prompts with no complication
- Result: monotone increasing steering curve
- Mean delta logit difference moves from `-4.277` at alpha `-3` to `+4.621` at alpha `+3`

Activation patching:

- Matched Type 1 / Type 2 pairs: 96
- Complications: `kidney`, `neurological`
- Layers: 5, 19, 20, 21, 22, 23
- Positions: -1, -2, -3, -4
- Best cell: layer 21, position -1
- Normalized patching score: `0.9955`
- 95% bootstrap CI: `[0.9751, 1.0148]`

SAE candidate circuit:

- Best SAE layer by top feature contribution: 21
- Matched pairs used for SAE tracing: 122 train pairs across all complications
- Strong axis-aligned features appear around layers 20-23, and layer 21 aligns with the residual-axis and patching peak
- Robust layer-21 candidates include features such as `F521` and `F3330`; exact IDs and scores are in `outputs/circuit_axis/axis_sae_summary.txt`

270M replay:

- Model: `google/gemma-3-270m-it`
- Best layer: 12
- Held-out accuracy: 0.707
- Null mean: 0.545
- Null p-value: 0.013
- Steering is also monotone increasing, from `-2.005` at alpha `-3` to `+2.005` at alpha `+3`

Interpretation:

- The residual-stream concept-axis finding is strong.
- Steering and activation patching support a causal role for the layer-21 axis in the 1B model.
- SAE tracing provides readable candidate features, but feature ablation is still needed before claiming a complete SAE-level mechanism.
- This is enough for a single-concept medical-axis case study; it is not enough to claim a universal circuit for all medical concepts.

### Pipeline

Each model/SAE-heavy stage runs in a separate Python process so Intel XPU memory is released between stages.

```text
0. scripts/check_hardware.py
   Check CPU, memory, Intel GPU, and PyTorch XPU availability.

1. scripts/step2_generate_prompts.py
   Generate 480 matched diabetes prompts.

2. scripts/step1_mvp.py
   Smoke-test Gemma + one Gemma Scope SAE.

3. scripts/step3_concept_axis.py
   Sweep residual layers, fit the Type1-vs-Type2 concept axis, and report
   held-out accuracy, bootstrap CI, shuffled-label null, DLA, and logit lens.

4. scripts/step4_axis_sae.py
   Trace Gemma Scope SAE features aligned with the learned residual axis.

5. scripts/step5_steering.py
   Add alpha * axis during forward passes and measure
   logit(insulin) - logit(metformin).

6. scripts/step6_patching.py
   Patch real Type2 residual activations into matched Type1 prompts.

7. scripts/step8_circuit_diagram.py
   Draw a black-and-white candidate circuit schematic.

8. scripts/step7_report.py
   Build `outputs/report.md` and `outputs/report.html`.
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default run is tuned for the local Intel Lunar Lake XPU setup:

```bash
DEVICE=xpu
DTYPE=bfloat16
```

### Run

Full clean run with the 1B main model and 270M replay:

```bash
rm -rf outputs/
LAYER_SWEEP=all \
PATCHING=1 \
RUN_270M=1 \
bash scripts/run_xpu_experiment.sh
```

Main 1B run only:

```bash
rm -rf outputs/
LAYER_SWEEP=all \
PATCHING=1 \
RUN_270M=0 \
bash scripts/run_xpu_experiment.sh
```

Fast smoke run:

```bash
rm -rf outputs/
PRESET=gemma3-270m-it-res \
LAYER_SWEEP=4 \
SAE_LAYERS=4 \
PATCHING_LAYERS=4 \
MAX_AXIS_SAE_PAIRS=8 \
MAX_PATCHING_PAIRS=1 \
MAX_STEERING_PROMPTS=2 \
NULL_TRIALS=5 \
BOOTSTRAP_TRIALS=10 \
RUN_270M=0 \
bash scripts/run_xpu_experiment.sh
```

Smoke-test outputs are only for checking that the pipeline runs. Delete `outputs/` before a real run.

### Main Outputs

Start with:

```bash
cat outputs/axis/axis_summary.txt
cat outputs/circuit_axis/axis_sae_summary.txt
cat outputs/steering/steering_summary.txt
cat outputs/patching/patching_summary.txt
```

Report:

```text
outputs/report.md
outputs/report.html
```

Figures:

```text
outputs/axis/accuracy_by_layer.png
outputs/axis/dla_and_logit_lens.png
outputs/circuit_axis/axis_sae_contributions_by_layer.png
outputs/circuit_axis/axis_sae_top_features.png
outputs/steering/steering_curve.png
outputs/patching/patching_heatmap.png
outputs/circuit_diagram/medical_circuit_diagram.png
outputs/circuit_diagram/medical_circuit_diagram.svg
```

### Recommended Claim Wording

Appropriate:

- "Gemma 3 1B contains a measurable Type1/Type2 diabetes residual-stream direction around layer 21."
- "The direction causally affects the insulin-vs-metformin readout under steering and activation patching."
- "Gemma Scope SAE features provide candidate components for a readable circuit."

Too strong:

- "This fully reproduces The Assistant Axis."
- "The SAE features are the complete mechanism."
- "All medical concepts share this circuit."
