# BRIDGE 2026 — Conclusions, confidence, and boundaries

Stage-4 terminal document. Every number cited here is in `MAIN_TABLE.md`; every decision that
produced it is in `docs/CHANGELOG_STAGE_GATES.md`.

---

## 1. What was asked

> When surface repetition is decorrelated from the underlying interaction state, can
> `gemma-3-*-it` distinguish a communication problem that is **still unresolved** from
> **surface-similar but legitimate repetition / confirmation**? And do interventions on residual
> directions and SAE features change that context judgement **specifically**, or only induce a
> generalized "ask more questions / help more" tendency?

## 2. What the answer is

**RQ-A (readout): no, at the pre-registered setting.**

- Behaviour is at chance. 4B zero-shot paired accuracy **0.600 [0.460, 0.720], p = 0.207**;
  12B **0.580 [0.460, 0.700], p = 0.330**. Chance is 0.500.
- Text-only baselines beat both models: TF-IDF char 3-5 gram reaches **0.760 [0.640, 0.880]**,
  a 19-feature surface model **0.720 [0.600, 0.830]**, both p < 0.005.
- At the **frozen** layer the residual probe scores **0.580** (4B) and **0.660** (12B) — *below* the
  0.720 surface reference. The increment of (surface + residuals) over surface alone is
  **-0.160 [-0.340, +0.010]** at 4B and **-0.060 [-0.200, +0.080]** at 12B. No layer anywhere shows
  a significant positive increment; the best layer *selected on test* reaches +0.100 with a CI
  spanning zero.

**RQ-B (selective causality): no, decisively.**

- Across 21 intervention arms at 4B and 17 at 12B, the largest absolute selectivity gap is
  **0.025** (4B) and **0.133** (12B), against main effects of up to 0.9 and 2.2 nats respectively.
  Target and non-target conditions move together, every time.
- Paired accuracy is **0.600 for every single 4B arm**, and the false-alarm rate never leaves
  0.68-0.74. No dose, direction, donor swap or feature edit moved the decision.
- The fitted axis is not even special among directions of its own size: at 4B, steering it at
  alpha = +-1 moves the answer ~0.02-0.07 nats while a *random* direction of equal norm moves it
  0.05-0.13 and the competing clarification axis 0.09-0.12.
- On free-form replies the interventions are close to inert: dense-axis steering leaves the
  generated text **byte-identical in 79%** of test items, SAE ablation in **87%**. Blind rubric
  scores for both are indistinguishable from the unmodified model (takes up an open issue: 0.149
  for original, dense-axis and SAE-ablate alike).

**What the model does instead: a blanket "something is wrong" bias.**

- On the test set the 4B model calls 70% of genuinely resolved dialogues unresolved; 12B calls
  92%. Adding "check the earlier turns first" pushes 4B to **1.000**.
- On 100 real human CCPE-M conversations (97 resolved), 4B false-triggers on **0.856
  [0.784, 0.919]** and 12B on **1.000 [1.000, 1.000]**. The rate is no higher on segments that
  actually contain repetition (4B: 0.816 with an echo vs 0.896 without), so this is not
  repetition-triggered — it is unconditional.
- On the 48 deliberately undecidable items the model is **more** confident than on decidable ones
  (mean |score| 5.40 vs 4.28; only 4.2% near indifference vs 18.0%). There is no abstention
  behaviour to speak of.

## 3. The one clearly positive finding

**The dialogue history does drive the judgement, even though the judgement is wrong.**

`swapped` replaces the history with the opposite-label partner's while keeping the item's own final
user turn; within a family the tail anchor is character-identical, so only the history changed.

| model | full | swapped | difference | 95% CI | p |
|---|---|---|---|---|---|
| 4B | 0.600 | 0.400 | +0.200 | [-0.060, +0.460] | 0.153 |
| **12B** | 0.580 | 0.320 | **+0.260** | **[+0.080, +0.460]** | **0.006** |

Swapping the history drives 12B *below* chance, which is what a model that follows the history
would do. So the information is reaching the decision; the decision rule is what fails. Shuffling
the history within speaker roles changes nothing (12B: -0.020), which fits — the shuffle preserves
the decisive fact, the swap replaces it.

A second positive control: a generic "this request is underspecified" direction, fitted on 60
single-turn prompts from domains this dataset never touches, is read out at **AUROC 0.93-0.98** in
late layers while being near-orthogonal to the target axis (cos = +0.10 at L29). The model is not
a model that represents nothing; it represents the neighbouring concept crisply and this one weakly.

## 4. Which title the evidence supports

Of the plan's candidates, the evidence does **not** support "Readable but Not Selectively
Controllable" — because at the pre-registered setting it is not cleanly readable either, once a
surface model is the reference. The defensible framing is narrower:

> **Beyond Repetition: conversational repair state is decorrelated from surface repetition by
> construction, is not recovered by Gemma-3-4B/12B above a bag-of-words baseline, and is not
> selectively controllable by residual or SAE interventions — while the models label almost all
> ordinary dialogue as unresolved.**

"Causal audit" is warranted (interventions were run, with matched controls). "Circuit" is not.
"Dementia axis", "early detection" and "digital biomarker" are not remotely supported and must not
appear.

## 5. Confidence, by claim

| claim | confidence | why |
|---|---|---|
| Repetition alone is uninformative in this dataset | **high** | `rep_rule` 0.520 [0.440, 0.600] dev, 0.550 [0.490, 0.610] test — chance by construction, replicated |
| Both models are at chance on the paired judgement | **high** | 4 splits x 2 models x 6 prompting arms, all CIs span 0.5 except two marginal arms that do not survive multiplicity |
| Both models massively over-predict "unresolved" | **very high** | 0.70-1.00 false alarm on constructed data; 0.856 and 1.000 on real human dialogue with tight CIs |
| Interventions are not selective | **high** | 38 arms across two models, matched random-direction / competing-axis / random-feature / reconstruction controls, gap <= 0.133 everywhere |
| The residual stream adds nothing over surface features | **moderate** | consistent across 2 models, 2 splits and 82 layers, but it is a null with ~25 families per split; a real effect below ~0.15 paired accuracy would not be detected |
| The history causally drives the judgement | **moderate** | significant at 12B on test (p = 0.006) but not at 4B (p = 0.153); one significant contrast among six |
| No cognitive imputation in free-form replies | **moderate-high** | 0/400 across four independent blind reviewers, but two reviewers flagged near-misses they scored under `false_trigger` instead |

## 6. Boundaries — what this does not show

1. **No cognitive-health claim of any kind.** The construct is an observable interaction state in
   ordinary task dialogue. Nothing here measures cognition, decline, or diagnosis. CCPE-M has no
   cognitive-state annotation and is not a healthy control group.
2. **The dataset is constructed, and it leaks.** 9 of 19 surface features separate the labels at
   the paired level (worst: type-token ratio, 0.706). Mean-matching hid a consistent within-pair
   direction, because a `resolved` cell has to contain the discharge. Every representational claim
   here is therefore an *incremental-over-surface* claim, and the incremental result is null.
3. **Annotation is LLM-assisted, not human.** Claude authored the dialogues; independent blind
   Claude passes re-annotated them (kappa = 0.950 on the 30-family gate set) and scored the
   free-form replies. There was no second human annotator and no clinical annotator. This must
   never be described as human or clinical annotation.
4. **Power.** 120 families, 24 per split-domain cell, ~25 families per evaluation split. Adequate
   for the large effects reported (false-alarm rates near 1.0) and for ruling out large selectivity
   effects; not adequate for small ones.
5. **One model family.** Gemma-3 4B and 12B only. This is a cross-scale replication, not a
   cross-architecture one.
6. **One decision format.** A forced A/B letter choice plus greedy free-form generation. A
   different elicitation could give different absolute rates, though the paired design makes the
   *contrasts* robust to constant bias.
7. **Interventions act on the prompt pass only.** The plan's own framing. Effects on multi-turn
   dynamics, and the feedback loop whereby a model's reply changes the next user turn, are not
   measured and are explicitly out of scope.
8. **Donor-based arms use ground truth.** As the plan requires, they support only bidirectional
   and selectivity conclusions, never a deployable correction claim.

## 7. Methodological findings worth reporting in their own right

These cost real time and each one nearly produced a false result.

1. **A single-token intervention at a late layer is causally inert.** At 4B layer 29 the read
   position sits 55-67 tokens before the answer with 4 layers left; a perturbation of twice the
   residual norm moves the final logits by 0.33 nats, versus 41.3 for the same perturbation at the
   final position. The first causal run returned exactly zero on every arm, including a full
   residual patch. A label-independent reachability check (E0.11) is now part of acceptance.
2. **SAE reconstruction damage swamps feature effects.** Replacing the residual with its SAE
   reconstruction and editing nothing costs -0.919 (4B) and -2.198 (12B); ablating the candidate
   features costs -0.102 and -0.034. Without the error-preserving path plus a reconstruction
   control, the damage alone reads as a large feature effect.
3. **Item-level shortcut audits miss paired-level leaks.** The same 19 features gave 0/19 flagged
   at item level and 9/19 at paired level.
4. **Capture and intervention sites must be identical.** The norm ratio between
   `hidden_states[n_layers]` and the last decoder output is 0.0128 (4B) and 0.0100 (12B) — a
   ~80-100x unit mismatch if the two are mixed.
5. **Ties matter in paired metrics.** Scoring ties as failures made a tie-producing rule baseline
   read 0.200 instead of 0.520. Relatedly, in 62% of pairs the final user turn is
   character-identical across the repair contrast, which is the design working as intended.

## 8. Files

- main results: `runs/bridge2026/stage4/MAIN_TABLE.md`
- error analysis: `runs/bridge2026/stage4/ERROR_ANALYSIS_4b.md`, `ERROR_ANALYSIS_12b.md`
- decision log: `docs/CHANGELOG_STAGE_GATES.md`
- specification: `docs/STAGE1_SPEC.md`
- frozen configs: `runs/bridge2026/frozen_config_{4b,12b}.json` (write-protected, evidence-hashed)
- dataset + freeze: `data/bridge2026/{families/,items.jsonl,splits.json,indeterminate.json}`
