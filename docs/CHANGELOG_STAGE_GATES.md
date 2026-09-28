# Stage-gate changelog

Every deviation, every pre-registration-relevant decision, and every time a model was run on data
before a freeze, recorded with the reason. Append-only.

## 2026-09-22 — Stage 1 opened

- Research question, construct, 2x2 design, annotation guidelines, split rule and hook-acceptance
  spec written to `docs/STAGE1_SPEC.md`. Frozen as the reference for Stages 2-4.
- Old ICD/CCS study demoted to pilot. Not reported as a result of this study.

### Environment / provenance decisions

- All Python dependencies installed **only** into the project-local `./venv` (python 3.11).
  No global or system-level installs. torch 2.11.0+cu128, transformers 5.17.0.
- Hardware: 3x RTX PRO 6000 Blackwell (sm_120, 97 GB each). GPU 0 is shared with another user's
  process and is left alone; this work uses GPU 1 (4B) and GPU 2 (12B).
- Weights: `google/gemma-3-4b-it` and `google/gemma-3-12b-it` pulled from the official Google repos
  with a user-supplied HF token (the repos are manually gated). No third-party mirror used.
- SAEs: `google/gemma-scope-2-{4b,12b}-it`, `resid_post`, ungated.
- **Run dtype fixed to float32.** Reason: in bfloat16 the E0.8 dose-linearity check fails at ~1.3%
  relative error, because the intervention (norm ~51) is added to a residual of norm ~24,700 in
  8-bit mantissa. The study measures small logit shifts, so dtype noise had to be removed rather
  than tolerated. Both models fit in fp32 on a single 97 GB card.

### E0 hook-site resolution

- Canonical site fixed to `resid_post(L)` = output of `model.model.language_model.layers[L]`,
  before the final norm. The Gemma Scope 2 configs independently confirm this choice: they declare
  `hf_hook_point_in = "model.layers.<L>.output"`.
- E0.1 confirms the hook is bit-identical to `hidden_states[L+1]` for `L < n_layers-1`.
- E0.2 quantifies the audit's concern: at `L = 33` the norm ratio
  `‖hidden_states[34]‖ / ‖hook(33)‖ = 0.0128`, i.e. the final-norm'd tensor is ~78x smaller.
  Capture and intervention would have been in incomparable units at the last layer. One site is now
  used for capture, probing, SAE, steering and patching.
- All eight E0 checks pass in float32 (`runs/bridge2026/stage1/e0_smoke.json`).

### Readout protocol decisions

- Old `readout_baseline` mixed a per-row own-vs-opposite sign with steering's fixed
  positive-vs-negative sign, and summed multi-token label log-probs (length + label-prior confound).
  Replaced by: single-token `A`/`B` options (verified single-token in both tokenizers) and one fixed
  sign convention, `score = logP(letter meaning unresolved) - logP(letter meaning resolved)`.
- The A/B question is appended **inside the final user turn** because Gemma's chat template requires
  strictly alternating roles. This also makes PURE / GEN / READOUT share a token prefix, so the
  canonical read position (last token of the final user text) is the same absolute index in all
  three, and answer-option tokens cannot leak into the extracted direction.

### Pre-freeze implementation smoke test (disclosed)

- On 2026-09-22, before any split was frozen, the readout was run on the 6 `appointment` families
  that existed at that moment (24 items), purely to verify the implementation end-to-end.
- Observed: the A/B readout carries a large option-position bias and a large wording-dependent label
  prior. Per (wording, order) cell the mean score ranged from -11.7 to +4.6, and under W1/order0 the
  model never chose the "unresolved" option.
- **Decision taken:** the primary behavioural endpoint is the **within-family paired contrast**
  (unresolved vs resolved at matched repetition level), where any constant option/wording/position
  bias cancels exactly. Raw per-item accuracy at threshold 0 is still reported alongside it, for
  transparency, together with AUROC (threshold-free).
- This is not tuning-on-results: the paired endpoint is what the plan already specified
  ("paired accuracy", "对不同模板和标签映射取配对结果"). What the smoke test changed is only that the
  raw unpaired threshold is now reported as a *diagnostic* rather than as the headline number.
- The split seed (20260922) is fixed and independent of these observations, and no item was edited,
  added or removed in response to them.

---

## 2026-09-23 — Stage 1 gate PASSED

| gate condition | result |
|---|---|
| 30 families x 4 conditions, schema-valid | 30 / 120 items, 6 per domain across 5 domains |
| validator clean (turn count, tail anchor, length band, `?` band, repetition thresholds, forbidden content, metadata leak, evidence-quote match) | `errors=0 warnings=0` |
| evidence spans checkable | every quote verified as an exact substring of its turn by the validator |
| blind re-annotation decidable | see below |
| split rule + hook acceptance spec written | `docs/STAGE1_SPEC.md` |

### Covariate balance achieved (120 items)

| | chars | words | `?` | turns | rep_any |
|---|---|---|---|---|---|
| unresolved (n=60) | 628.0 | 119.7 | 1.50 | 5.67 | 6.03 |
| resolved (n=60)   | 629.5 | 119.8 | 1.50 | 5.67 | 5.70 |

Repetition factor cleanly separated: `rep_*` cells median 9 (min 5), `norep_*` cells median 2 (max 3).
Positive subtypes 20/21/19; negative subtypes 18/25/17.

### Blind re-annotation

Four independent annotators, each given exactly one condition per family, so no annotator could
contrast two cells of the same family. 120/120 items returned.

- raw agreement 0.975, **Cohen's kappa 0.950**
- on the decidable subset (1 blind `indeterminate`): agreement 0.983, kappa 0.966
- subtype agreement where labels agree: 0.983
- by condition: norep_res 30/30, rep_res 30/30, rep_unres 29/30, norep_unres 28/30
- by domain: all 24/24 except `household` 21/24

### The three disputes, adjudicated

All three were `household` `unresolved` cells that the blind pass read as resolved/indeterminate. On
review the blind annotator was defensible in all three and clearly right in two. All three were
**repaired**, not deleted, and the repairs were minimal:

1. `household_001__norep_unres` — the bin-out action was implicit. Now: *"I'll put the first week's
   bin out myself as we leave, we're off at nine that morning"* against a half-six collection.
2. `household_004__norep_unres` — "she" had a topical winner (Noor). Turn 3 now names Freya as the
   most recent singular mention, so the pronoun genuinely has two live candidates.
3. `household_006__rep_unres` — the consequence was left to inference. Now: *"Theo is sleeping in
   there from Friday night"* against a Saturday-morning build.

The paired `resolved` cells were edited symmetrically so each minimal counterfactual still turns on
one word (nine/five, she/Noor, Friday/Sunday). These three failure modes were written into
`docs/AUTHORING_BRIEF.md` before any further families were authored.

### Repair re-check (with a disclosed blinding weakness)

A fresh annotator scored the 6 repaired items hidden among 14 unchanged fillers: 20/20 agreement,
all 6 repaired items correct. **Caveat, recorded because it matters:** both members of each repaired
minimal pair were placed in the same packet, so the annotator could see the manipulated variable and
said so explicitly. This is a repair check, not an independent agreement estimate; the kappa of
record for the dataset is the 0.950 from the primary pass.

### Honest description of the annotation

This is LLM-assisted authoring (Claude Opus 5) with independent blind LLM re-annotation and
rule-based checks. It is **not** human annotation, **not** multi-annotator human consensus, and
**not** clinical annotation. The model under test (`gemma-3-*-it`) generated none of the data, set
none of the ground truth, and evaluated none of its own outputs.

### Other Stage-1 infrastructure completed

- GPU note: another user's processes now occupy memory on all three cards; this work uses whichever
  card has the most free memory and does not displace them.
- Competing-explanation probe built: `data/bridge2026/clarification_probe.json`, 30 underspecified
  vs 30 fully specified single-turn requests, deliberately from domains the main set never touches.
- External test material prepared: 100 CCPE-M segments (CC BY 4.0), 50 with a >=5-token echo and 50
  without, matched to the main set's turn structure. Labels will not be read until the model and
  thresholds are frozen.

### Disclosed pre-freeze pipeline dry run (2026-09-23)

The full Stage-2 pipeline (build -> capture -> E2) was executed once on the 30-family set, writing
only to a scratch directory, to verify the code paths before the dataset was complete. The 30-family
split (20/5/5 families) gives 10 dev pairs, so every interval was uninformative (best mean-difference
dev AUROC 0.70, 95% CI [0.58, 0.91]). No layer, feature, dose or threshold was selected from it, and
the Stage-2 split is recomputed over the full 120-family set, so the dev membership differs. Recorded
because model output on not-yet-frozen data was produced.

---

## 2026-09-26 — Stage 2: dataset complete, split frozen, dev results

### Dataset

120 families x 4 conditions = 480 items, 24 families per domain, validator `errors=0 warnings=0`.
All 120 tail anchors unique. Split frozen: **280 train / 100 dev / 100 test items**
(14/5/5 families per domain), seed 20260922, content hash `c8bbd8021b606473`.

Mean covariate balance by label (n=240 each): chars 727.2 / 731.9, words 139.9 / 140.8,
`?` 1.35 / 1.32, turns identical. Repetition factor separated: `rep_*` 5–25, `norep_*` 1–4.

### The important Stage-2 finding: mean-matching hid a within-pair leak

The gate requires shortcut cues to be excluded. The item-level audit (single-feature AUROC with
within-family permutation) flagged **0 of 19** features. That audit was the wrong test. Re-running it
at the **paired** level — the level of the primary metric — flags **9 of 19**:

| feature | paired acc | 95% CI | rep pairs | norep pairs |
|---|---|---|---|---|
| `ttr` (type-token ratio) | **0.706** | [0.654, 0.760] | 0.708 | 0.704 |
| `n_words` | 0.369 | [0.323, 0.415] | 0.367 | 0.371 |
| `n_chars` | 0.360 | [0.304, 0.421] | 0.346 | 0.375 |
| `n_asst_words` | 0.425 | [0.392, 0.458] | 0.429 | 0.421 |
| `rep_user` | 0.560 | [0.533, 0.588] | 0.654 | 0.467 |
| `n_user_qmarks` | 0.556 | [0.537, 0.575] | 0.613 | 0.500 |

Matching the **means** across the corpus did not remove a consistent **within-pair direction**.
`resolved` cells are systematically slightly longer (they must contain the discharge — the answer,
the confirmation) and lexically less diverse; `unresolved` cells introduce a conflicting fact, which
raises type-token ratio. This is partly intrinsic to the construct rather than sloppy authoring, but
it is exploitable either way.

Note `ttr` leaks equally in `rep` (0.708) and `norep` (0.704) pairs, so it is **not** a repetition
artifact. The repetition factor itself is clean: the `rep_rule` baseline sits at
**0.520 [0.440, 0.600], p=0.798** — exactly chance, as the 2x2 was designed to guarantee.

**Consequence for the claims.** Any statement that the interaction state is "readable from the
residual stream" is only meaningful as an **incremental** claim over a surface model. That is now the
primary representational analysis (`scripts/e2_incremental.py`), and the 19-feature
`surface_lr` is the baseline everything must beat.

Also fixed: `paired_accuracy_from` counted ties as failures, which made tie-producing rule baselines
look far below chance (`rep_rule` read 0.200 before the fix, 0.520 after). Ties now count 0.5.

### Dev results, 4B (n=100 items / 25 families)

**Behaviour — at chance in every arm.**

| arm | paired acc | perm p | AUROC | false alarm |
|---|---|---|---|---|
| zero-shot, full history | 0.520 [0.400, 0.640] | 0.908 | 0.503 | 0.700 |
| zero-shot, last turn only | 0.260 [0.160, 0.360] | 0.162 | 0.512 | 0.640 |
| zero-shot, shuffled history | 0.420 [0.280, 0.560] | 0.305 | 0.500 | 0.820 |
| zero-shot, swapped history | 0.400 [0.260, 0.540] | 0.202 | 0.457 | 0.740 |
| + "check the context first" | 0.540 [0.420, 0.660] | 0.709 | 0.497 | **0.980** |
| few-shot k=4 | 0.580 [0.420, 0.740] | 0.309 | 0.538 | 0.460 |

The prompt-only instruction does not create discrimination; it converts the model into an
almost-always-"unresolved" responder (false-alarm rate 0.98). That is the generalized
"help-more" shift the plan warned about, observed directly.

**Text baselines beat the model's own behaviour.**

| baseline | paired acc | AUROC |
|---|---|---|
| majority | - | - |
| `rep_rule` (>=5-token echo) | 0.520 [0.440, 0.600] | 0.502 |
| length rule | 0.390 [0.270, 0.510] | 0.487 |
| **`surface_lr` (19 features)** | **0.810 [0.700, 0.910]** | 0.647 |
| TF-IDF word 1-2 | 0.550 [0.430, 0.660] | 0.647 |
| TF-IDF char_wb 3-5 | 0.660 [0.540, 0.800] | 0.608 |

**Representation — readable, but not beyond surface.** Best probe L26 paired 0.820 [0.700, 0.920].
Best increment of (surface + residuals) over surface alone: **L26 +0.030 [-0.100, +0.160], p=0.72**.
No layer shows a significant increment. On the surface-matched third of pairs the probe reaches
0.824 while surface falls to 0.676, but that subset is selected on the surface margin and has ~40
pairs, so it is suggestive only.

**Competing explanation is strongly represented.** A generic "this request is underspecified"
direction, fitted on 60 single-turn prompts from untouched domains, reaches probe AUROC
0.93-0.98 in late layers — far above anything the interaction-state axis achieves — while being
near-orthogonal to it. The model is not failing to represent anything; it represents a neighbouring
concept crisply and this one weakly.

### Intervention conditions selected on dev (pre-registered candidate set)

The plan pre-specified 4B SAE layers {9, 17, 22, 29}. Selecting within that set (not over all 34
layers) limits dev-overfitting. **L29 is best on every method**: mean-difference AUROC 0.578
[0.544, 0.624] perm p=0.045 and beats the random-direction null; logistic 0.616; SAE 16k k=16
paired 0.680. cos(unresolved, repetition) = -0.135 at L29, i.e. near-orthogonal.

### Frozen intervention configuration (4B)

`runs/bridge2026/frozen_config_4b.json`, write-protected, evidence hash over the dev sweep files.

- layer **29**, hook site `resid_post(29)`, read position = last token of the final user turn's text
- `sd_train` (projection SD on train at L29) = 1242.36; doses alpha in {-2, -1, -0.5, 0, 0.5, 1, 2}
- SAE `gemma-scope-2-4b-it/resid_post/layer_29_width_16k_l0_medium`, k = 16, features ranked on
  **train only**
- controls: random direction and the repetition axis at matched perturbation norm, random features
  matched on activation frequency and magnitude, SAE reconstruction path, whole-residual donor patch

**Tie-break fix, disclosed.** The first freeze attempt selected L22, because L22 and L29 both reach
0.680 SAE paired accuracy and `max` resolved the tie by file order; it also printed the dense
numbers of a different layer than the one it selected. The selection rule is now explicit and
deterministic — SAE paired accuracy, then SAE AUROC, then the dense readout at the same layer — and
the config records both the selected layer's dense numbers and the best dense layer overall. This
changed the selected layer from 22 to 29. It was made on dev evidence only, before any test item was
scored, and L29 dominates L22 on every dense metric (mean-difference AUROC 0.578 vs 0.526, paired
0.680 vs 0.500, permutation p 0.045 vs 0.381).

---

## 2026-09-26 — Stage 3: causal experiments (4B, dev)

### A bug found before it became a "finding": the intervention site was causally inert

The first E3 run returned **exactly zero effect on every arm**, including a whole-residual donor
patch. That is not a result; a full patch must change something. Diagnosis: the hook writes
correctly (verified, `|h'-h| = 10000.0` for a requested 10000), but at layer 29 the canonical read
position sits 55-67 tokens before the answer token with only 4 decoder layers left. Measured:

| perturbation | site | max abs delta logit |
|---|---|---|
| norm 1e3 | read position, L29 | 0.016 |
| norm 1e5 (2x the residual norm) | read position, L29 | 0.334 |
| norm 1e3 | final position, L29 | 2.609 |
| norm 1e4 | final position, L29 | 41.296 |
| zero the entire residual | read position, L29 | 0.255 |
| zero the entire residual | read position, L5 | 7.949 |

By late layers the information has already moved out of that token, so a single-token intervention
there cannot reach the answer. **A null causal result at that site would have been an artefact.**

### New acceptance check E0.11 — causal reachability

Added `scripts/e0_reachability.py`. It perturbs a candidate site with **random** directions
at alpha = 1 sd_train and measures the mean absolute change in the A-B logit margin. It is
label-independent, so using it to choose a site cannot bias the direction of any causal result.

| layer | read_pos | read_to_end | prompt_all |
|---|---|---|---|
| 9  | 0.0055 (inert) | 0.157 | 0.497 |
| 17 | 0.0003 (inert) | 0.085 | 0.058 |
| 22 | 0.0005 (inert) | 0.084 | 0.153 |
| 29 | 0.0007 (inert) | 0.061 | 0.080 |

The single-token site is inert at **every** candidate layer. The frozen config was amended to
`position_mode = read_to_end`, which the plan pre-specified as the next step ("fixed position at the
last user-turn boundary first, then extend the span"). The inert `read_pos` arm is still run and
reported, so the finding is visible in the results table rather than hidden in a footnote.

Also fixed: the SAE donor arm broadcast one donor code per *row* onto a *flattened span* of
positions; donor codes are now indexed by the row each masked position belongs to.

### E3 dev results (4B, layer 29, span read_to_end, n=100 items / 25 families)

Sign convention throughout: + = shifted toward "unresolved".

| arm | delta target (unresolved) | 95% CI | delta non-target (resolved) | selectivity gap | paired acc | false alarm |
|---|---|---|---|---|---|---|
| clean | - | - | - | - | 0.520 | 0.700 |
| dense_axis a=-2 | +0.069 | [-0.044, +0.177] | +0.076 | -0.007 | 0.520 | 0.700 |
| dense_axis a=-1 | +0.021 | [-0.030, +0.070] | +0.025 | -0.004 | 0.520 | 0.700 |
| dense_axis a=+1 | +0.004 | [-0.037, +0.046] | -0.001 | +0.005 | 0.520 | 0.700 |
| dense_axis a=+2 | +0.025 | [-0.050, +0.104] | +0.014 | +0.011 | 0.520 | 0.700 |
| dense_axis a=+2, **read_pos only** | +0.000 | [-0.000, +0.000] | +0.000 | +0.000 | 0.520 | 0.700 |
| random_dir a=-1 | +0.102 | [+0.061, +0.145] | +0.103 | -0.000 | 0.500 | 0.700 |
| random_dir a=+1 | -0.050 | [-0.088, -0.007] | -0.051 | +0.001 | 0.500 | 0.700 |
| repetition_axis a=+1 | +0.050 | [+0.024, +0.076] | +0.048 | +0.002 | 0.520 | 0.700 |
| clarification_axis a=-1 | +0.115 | [+0.068, +0.158] | +0.122 | -0.007 | 0.500 | 0.700 |
| clarification_axis a=+1 | -0.093 | [-0.131, -0.050] | -0.096 | +0.004 | 0.520 | 0.700 |
| component_swap (donor) | +0.026 | [-0.047, +0.093] | +0.022 | +0.004 | 0.540 | 0.700 |
| full_patch (donor, read_pos) | -0.000 | [-0.001, +0.000] | +0.000 | -0.000 | 0.520 | 0.700 |
| **sae_recon** (no feature edit) | **-0.636** | [-0.849, -0.422] | -0.648 | +0.013 | 0.500 | 0.680 |
| sae_ablate (k=16 candidates) | -0.084 | [-0.137, -0.027] | -0.086 | +0.003 | 0.520 | 0.700 |
| sae_random_ablate (matched) | -0.022 | [-0.030, -0.013] | -0.024 | +0.002 | 0.520 | 0.700 |
| sae_donor (k=16 candidates) | +0.220 | [+0.073, +0.362] | +0.252 | -0.032 | 0.520 | 0.700 |
| sae_random_donor (matched) | +0.160 | [+0.083, +0.228] | +0.145 | +0.015 | 0.540 | 0.700 |

Four things this table says, none of them the hoped-for result:

1. **No selectivity anywhere.** The largest |gap| across 21 arms is 0.032. Target and non-target
   conditions move together, every time. Paired accuracy never leaves 0.500-0.540 and the
   false-alarm rate never leaves 0.68-0.70, under any intervention at any dose.
2. **The fitted axis is weaker than its own controls.** Steering along the unresolved axis at
   alpha=+-1 moves the answer by ~0.02 nats; a *random* direction of the same norm moves it by
   0.05-0.10, and the competing clarification axis by 0.09-0.12. The direction that best reads the
   label out is not the direction that best moves the answer, and is not even special among
   directions of equal size.
3. **SAE reconstruction damage dwarfs the candidate-feature effect.** Replacing the residual with
   its SAE reconstruction and editing nothing costs -0.636; ablating the 16 candidate features
   costs -0.084, and ablating 16 frequency- and magnitude-matched *random* features costs -0.022.
   Had the error-preserving path and the reconstruction control been omitted, the reconstruction
   damage alone would have looked like a large feature effect.
4. **The inert arm is visibly inert**, confirming E0.11 in the main table.

---

## 2026-09-26 — Stage 3 completed: 12B replication, external set, gate closed

### 12B implementation acceptance

9 of 10 E0 checks pass. `E0.2` norm ratio `hs/hook = 0.01003` at the last layer (the same hook
hazard as 4B, ~100x). **E0.5 flagged**: layer 12's SAE fires at L0 40.3 against a released config of
52 (ratio 0.775, just under the pre-set 0.8 band), although cosine is 0.9966. The threshold was NOT
relaxed. Layer 12 is simply not used; the selected layer 31 is inside the band (ratio 1.015,
cos 0.9920). Reported as a flagged-but-unused check rather than a pass.

### 12B reachability (E0.11)

| layer | read_pos | read_to_end | prompt_all |
|---|---|---|---|
| 12 | 0.0140 (inert) | 0.231 | 0.225 |
| 24 | 0.0009 (inert) | 0.060 | 0.069 |
| 31 | 0.0002 (inert) | 0.045 (inert) | 0.074 |
| 41 | 0.0002 (inert) | 0.071 | 0.232 |

The single-token site is inert at every layer in both models. At the selected 12B layer (31)
`read_to_end` is marginal, so the 12B span is **`prompt_all`**, chosen to give the intervention the
most power — the conservative choice when the expected outcome is a null. The span therefore
differs between models (4B `read_to_end`, 12B `prompt_all`) and this is reported, not smoothed over.

### 12B frozen config

`runs/bridge2026/frozen_config_12b.json`, write-protected. Layer **31**, `sd_train` = 2434.11,
SAE `resid_post/layer_31_width_16k_l0_medium`, k = 4, features `[1422, 337, 402, 6647]`, span
`prompt_all`. Each model extracted its own directions and its own features; nothing was carried
across models.

### 12B dev results

**Behaviour: also at chance, with a worse false-alarm rate than 4B.**

| arm | paired acc | perm p | AUROC | false alarm |
|---|---|---|---|---|
| zero-shot, full | 0.600 [0.460, 0.740] | 0.209 | 0.516 | 0.920 |
| zero-shot, last only | 0.490 [0.400, 0.580] | 0.990 | 0.461 | 1.000 |
| zero-shot, shuffled | 0.580 [0.440, 0.720] | 0.321 | 0.515 | 1.000 |
| zero-shot, swapped | 0.440 [0.300, 0.560] | 0.489 | 0.462 | 0.940 |
| + "check the context first" | 0.640 [0.500, 0.780] | 0.066 | 0.554 | 1.000 |
| few-shot k=4 | 0.580 [0.440, 0.720] | 0.298 | 0.549 | 0.980 |

The 12B model labels **92-100% of genuinely resolved dialogues as unresolved**. Scaling up made the
false-alarm problem worse, not better.

**Representation: same picture.** Best mean-difference L25 AUROC 0.612 [0.569, 0.670] paired 0.780;
best logistic L24 0.657 paired 0.780; best SAE L31 k=4 paired 0.700. Best increment over the
19-feature surface model: **L23 +0.070 [-0.040, +0.180], p=0.26** — larger than 4B's +0.030 but
still not distinguishable from zero.

**Causal: no selectivity, at much larger main effects.**

| arm | delta target | delta non-target | gap |
|---|---|---|---|
| dense_axis a=+2 | -1.245 | -1.207 | -0.037 |
| dense_axis a=-1 | +0.641 | +0.618 | +0.024 |
| dense_axis a=+2, read_pos only | -0.002 | -0.001 | -0.001 |
| random_dir a=-1 | -0.848 | -0.781 | -0.067 |
| repetition_axis a=-1 | +0.145 | +0.111 | +0.034 |
| component_swap (donor) | -0.220 | -0.260 | +0.040 |
| **sae_recon** | **-2.198** | -2.197 | -0.002 |
| sae_ablate (k=4) | -0.034 | -0.032 | -0.002 |
| sae_random_ablate (matched) | +0.041 | +0.040 | +0.000 |
| sae_donor (k=4) | +0.582 | +0.450 | +0.133 |
| sae_random_donor (matched) | -0.432 | -0.419 | -0.013 |

At 12B the interventions move the answer by up to 2.2 nats, so this is not a power problem — and
the selectivity gap is still at most 0.133. SAE reconstruction damage (-2.198) is ~65x the
candidate-feature ablation effect (-0.034), and the candidate features are not distinguishable from
frequency-matched random features (-0.034 vs +0.041).

### External test material prepared and annotated (labels not yet read against any model)

100 CCPE-M segments, blind-annotated by four independent annotators under the same guidelines:
**97 resolved, 3 unresolved, 0 indeterminate** (confidence: 80 high, 19 medium, 1 low). 49 of the
resolved segments contain a >=5-token echo and 48 do not. With that distribution, discrimination
metrics are meaningless and will not be reported; the set measures one thing only — the
false-trigger rate on ordinary human dialogue, split by whether repetition is actually present.

### Stage 3 gate: CLOSED

- Layers, features, doses, spans frozen in two write-protected configs, each carrying a SHA-256 of
  the dev evidence it was derived from.
- No test-split item has been scored by any model at this point.
- Every selection was made on dev or on label-independent reachability evidence.

---

## 2026-09-26 — Correction to the 4B dev `last_only` number, and what it revealed

The 4B dev readout was produced **before** `paired_accuracy_from` was fixed to count ties as 0.5,
so one number in the Stage-2 table above is wrong as printed:

| arm | as first reported | corrected |
|---|---|---|
| `zero_shot/last_only` (4B dev) | 0.260 | **0.570 [0.490, 0.650], p=0.162** |

No other arm moved: recomputing every arm from the saved per-item scores under the current code
reproduces them exactly (`runs/bridge2026/stage2/e1_model_readout_dev_fixed.csv`). The reason is
that `last_only` is the only arm that produces exact ties — and it produces a lot of them:
**31 of 50 dev pairs are tied**.

That is a design property, not a defect, and it is worth stating plainly: in 62% of pairs the final
user turn is **character-identical** between the unresolved and the resolved cell, because the
decisive difference was placed in an earlier turn. For those pairs the `last_only` ablation hands
the model literally the same input twice, so the label provably cannot be read off the final turn.
The corrected 0.570 is the right reading of that control; the old 0.260 was an artefact of scoring
a forced tie as a failure.

All test-split numbers were produced after the fix.

### An analysis the ablations earned: full vs swapped

`swapped` gives the model the opposite-label partner's history with the item's own final user turn.
Because the tail anchor is character-identical within a family, the only thing that changed is the
history. So the sign is meaningful: a model that ignores history scores the same as `full`; a model
that *follows* history flips below chance.

| | full | swapped | difference | 95% CI | p |
|---|---|---|---|---|---|
| 4B dev | 0.520 | 0.400 | +0.120 | [-0.100, +0.340] | 0.348 |
| 12B dev | 0.600 | 0.440 | +0.160 | [-0.060, +0.380] | 0.178 |

Both positive, neither significant on dev. This contrast is reported on test as the primary
measure of context use, because it is immune to the constant option/wording bias that makes the
absolute paired accuracies hard to interpret.

---

## 2026-09-26 — Stage 4 executed and closed. Pipeline terminated.

The test split was read **once**, by `scripts/run_stage4_test.sh`, which reads the two
write-protected frozen configs and selects nothing. Integrity verified after the fact:

- families content hash `c8bbd8021b6064736c12900d` matches the frozen split hash — no family file
  changed between the freeze and the test read.
- both frozen configs' `evidence_hash` still match the dev sweep files they were derived from.
- validator: `families=120 items=528 errors=0 warnings=0`.

### Headline test results

See `runs/bridge2026/stage4/MAIN_TABLE.md` (9 panels, no truncation) and
`runs/bridge2026/stage4/CONCLUSIONS.md`.

- Behaviour at chance: 4B 0.600 [0.460, 0.720] p=0.207; 12B 0.580 [0.460, 0.700] p=0.330.
- Text baselines beat both models: char-ngram TF-IDF 0.760 [0.640, 0.880].
- At the **frozen** layer the probe is *below* the surface reference (4B 0.580, 12B 0.660 vs 0.720);
  increment over surface -0.160 [-0.340, +0.010] and -0.060 [-0.200, +0.080].
- No selectivity: largest |gap| 0.025 across 21 arms (4B), 0.133 across 17 arms (12B).
- Free-form: dense-axis steering leaves the reply byte-identical in 79% of items, SAE ablation 87%.
  Blind rubric (400 responses, 4 independent reviewers): both arms identical to the unmodified
  model on taking up an open issue (0.149). `prompt_only` gains +0.106 there but costs +0.160 in
  false triggers — non-selective, exactly as the plan anticipated.
- `cognitive_inference` **0/400** across all arms and all four reviewers.
- External (CCPE-M, 97 resolved of 100): false-trigger 0.856 [0.784, 0.919] at 4B and
  **1.000 [1.000, 1.000]** at 12B, and no higher on segments that contain repetition.
- Ambiguity set: the model is *more* confident on the 48 undecidable items than on decidable ones
  (mean |score| 5.40 vs 4.28); 4.2% near indifference vs 18.0%. No abstention behaviour.
- One clearly positive result: `full - swapped` = **+0.260 [+0.080, +0.460], p=0.006** at 12B, so
  the history does causally drive the judgement even though the judgement is wrong.

### Two fixes applied during Stage 4, both disclosed

1. `e1_baselines.py` computed paired metrics but its CSV writer still had the pre-paired column
   list, so they were dropped on write. Fixed; the baselines were regenerated (deterministic,
   no model involved, no tuning).
2. Two-sided bootstrap p-values could exceed 1.0 when the draw distribution had mass at exactly
   zero. Clamped to <= 1.0 in three places. Affects reported p-values only, never a point
   estimate or an interval.

### Scope note

The plan's Stage-4 scope was the main table and the error analysis. Both are delivered, together
with a confidence-and-boundaries assessment. No paper, slides or PDF were produced, per the
instruction to stop here.

**Pipeline terminated at Stage 4 as specified.**

---

## 2026-09-27 — Generality check (third model family) and paper draft

Deadline re-verified per the plan's instruction: the workshop CFP now states **October 5, 2026,
23:59 AoE** (extended). 4 pages for a short paper, double-blind.

### Qwen2.5-7B-Instruct added as a generality check

Not a tuned arm: it reuses the identical prompt set, sign convention and scoring code, so there are
no free parameters to select for it. No SAE or causal arms (Gemma Scope has no Qwen SAEs); the
question is only whether the over-attribution finding is specific to Gemma-3.

| | test paired acc | p | FA on test | FA on CCPE-M |
|---|---|---|---|---|
| Gemma-3-4B | 0.600 [0.460, 0.720] | 0.207 | 0.700 | 0.856 [0.784, 0.919] |
| Gemma-3-12B | 0.580 [0.460, 0.700] | 0.330 | 0.920 | 1.000 [1.000, 1.000] |
| **Qwen2.5-7B** | 0.540 [0.450, 0.630] | 0.676 | 0.780 | **0.835 [0.760, 0.908]** |

The finding replicates across model families. Qwen's CCPE-M false-alarm rate is also flat in
repetition (0.837 with a >=5-token echo vs 0.833 without), confirming the effect is unconditional.

`full - swapped` is now significant in **two of three** models: 12B +0.260 [+0.080, +0.460] p=0.006,
Qwen +0.200 [+0.040, +0.360] p=0.021, 4B +0.200 p=0.153. Two independent confirmations that the
history reaches the decision while the decision rule fails.

`runtime.py` was made model-agnostic (Gemma3 nests decoder layers under `language_model`; Qwen does
not). Gemma results are unaffected — the accessor returns the same module list.

### Human annotation packet issued

`human_annotation/` — 100 blind items (60 constructed, stratified 15 per condition and 12 per
domain; 40 CCPE-M, half with a >=5-token echo), with the answer key held separately at mode 600.
This is to convert "LLM-drafted, LLM-re-annotated" into "…with an author's independent annotation
on a subset and a reported agreement rate", which is the single largest remaining weakness.

### Paper draft

`paper/main.tex` (IEEEtran conference, double-blind), two figures and one table.
Estimated 3.6 of 4 pages. **Not compiled** — no LaTeX toolchain on this machine.

Every number in the text is emitted by `scripts/paper_numbers.py` into `paper/NUMBERS.json`,
and a cross-check script verifies that each 3-decimal figure in the body traces to that file.
That check caught **three places where development-set numbers had been written into the test
results**, all now corrected:

| claim | was (dev) | now (test) |
|---|---|---|
| largest 12B selectivity gap | 0.133 | **0.151** |
| 12B SAE random-feature ablation | +0.041 | **-0.017** |
| 4B final-layer norm ratio | 0.0128 (n=4 pilot) | **0.0175** (n=96 acceptance run) |

Also corrected: 12B reconstruction damage -2.198 (dev) to **-2.464** (test), 12B candidate-feature
ablation -0.034 to **-0.028**, the dose-comparison figures (dev to test), and the abstract's
per-model percentages, which were listed out of order relative to the models.

Bibliography metadata was verified against the sources rather than written from memory; three
entries (Ngo et al. LREC 2026, Bouzid et al. arXiv:2507.12950, Mu & Chen arXiv:2608.27397) had
their exact titles and author lists fetched and corrected.
