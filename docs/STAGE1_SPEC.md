# Stage 1 Specification — Research Question, Data Design, Annotation Guidelines, Hook Acceptance Spec

Project: BRIDGE 2026 redesign. Supersedes the ICD/CCS study, which is retained only as pilot.
Frozen: 2026-09-22. Any later change must be recorded in `docs/CHANGELOG_STAGE_GATES.md` with a reason.

---

## 1. Research question

**RQ.** When surface repetition is decorrelated from the underlying interaction state, can
`gemma-3-*-it` distinguish a *communication problem that is still unresolved* from *surface-similar
but legitimate repetition / confirmation*? And do interventions on residual directions and SAE
features change that **context judgement specifically**, or only induce a generalized
"ask-more-questions / help-more" tendency?

Two sub-questions, mapped to the experiment matrix:

- **RQ-A (readout).** Is the interaction state linearly readable from the residual stream at the end
  of the last user turn, when lexical repetition cues are balanced by design?
- **RQ-B (selective causality).** Does intervening on that direction / on SAE features change the
  model's judgement **on the conditions that need it** without equally corrupting the conditions
  that do not (i.e. is the effect selective, not a global response-style shift)?

**Out of scope claims.** No cognitive-decline measurement, no diagnostic performance, no care
benefit, no clinical validation. The construct labelled here is an *observable interaction state*,
nothing else.

---

## 2. Construct and label set

Unit of analysis: one short task-oriented dialogue between `user` and `assistant`, ending on a
**user** turn.

Label (the thing the model must judge), assigned to the dialogue **as of its final turn**:

| label | meaning |
|---|---|
| `unresolved` | A communication problem is still open at the end of the dialogue. |
| `resolved` | No open communication problem; the exchange is on track. |
| `indeterminate` | Evidence is genuinely insufficient to decide. Held out of the main A/B analysis. |

**Positive (`unresolved`) requires explicit, quotable evidence** of one of exactly three subtypes:

- `unanswered_question` — the user requested information/explanation and it has still not been
  provided by the assistant at the end of the dialogue.
- `contradiction` — two pieces of task-critical information conflict and the conflict has not been
  raised or reconciled.
- `ambiguous_reference` — a referent required to act is still under-determined (≥2 live candidates
  in context) and has not been disambiguated.

**Never sufficient on their own for `unresolved`:** a single repetition, disfluency, short answers,
politeness formulae, stated age, hesitation markers, or any mention of memory/health. Age and health
mentions are *forbidden* as label-bearing evidence (see §4.5).

**`resolved` requires** that every user request in the dialogue has been answered, no task-critical
contradiction is open, and every referent needed for the next action is determined.

---

## 3. Factorial design

Each **scenario family** is realised in **4 conditions**, a full crossing of

`repetition_surface ∈ {rep, norep}` × `repair_state ∈ {unres, res}`

| condition id | repetition | label | construction |
|---|---|---|---|
| `rep_unres`  | present | `unresolved` | user re-asks; the key question is still unanswered |
| `rep_res`    | present | `resolved`   | read-back / confirmation repetition; nothing is open |
| `norep_unres`| absent  | `unresolved` | no repeated wording; contradiction or ambiguous referent open |
| `norep_res`  | absent  | `resolved`   | information complete and consistent, task proceeds |

This crossing is the core control: a classifier that keys on repetition alone scores exactly 50%.

### 3.1 Within-family matching constraints (hard)

For the 4 conditions of one family:

1. **Same domain, same entities, same task goal.**
2. **Identical turn count** `n_turns`, identical speaker sequence, ending on `user`.
3. **Shared tail anchor.** The **final sentence of the final user turn is character-identical**
   across all 4 conditions. This is the token neighbourhood where representations are read out, so
   it must not itself carry the label.
4. **Length matching.** Each condition's total character length within **±18%** of the family mean.
5. **Question-mark matching.** `count('?')` differs by **≤1** between any two conditions of a family.
6. **No metadata leakage.** Condition ids, labels, subtypes, evidence spans never appear in text.

### 3.2 Repetition operationalisation (computable)

Tokens: lowercased, punctuation stripped, whitespace split. `echo(a,b)` = length of the longest
contiguous common token run between turn `a` and an earlier turn `b`.

- `rep_user` = max `echo` over pairs of **distinct user turns**.
- `rep_any`  = max `echo` where the repeating turn is a **user** turn and the source is any earlier turn.

Requirements, enforced by `bridge2026/validate.py`:

- `rep`   conditions: `rep_any >= 5`.
- `norep` conditions: `rep_any <= 4`.

Named entities repeated in isolation (a date, a name) do not reach 5 contiguous tokens, so they do
not accidentally flip the factor.

### 3.3 Indeterminate set

Separate items (**not** part of any 2×2 family), authored to be genuinely undecidable — e.g. the
user's request could plausibly already have been answered by an earlier assistant turn, or the
ambiguity may or may not matter for the next action. Target ≈10% of the main-set item count.
Used **only** for abstention / uncertainty analysis, reported separately. They are assigned to the
ambiguity set **before** any model is run.

---

## 4. Annotation guidelines

### 4.1 Procedure per item

The annotator reads only the dialogue text (no metadata) and records:

1. `label` ∈ {unresolved, resolved, indeterminate}
2. `subtype` (positives only)
3. `evidence_spans`: list of `{turn_index, quote}` — the *minimal* spans that establish the label.
   For `unanswered_question`: the turn that asks **and** the fact that no later assistant turn
   answers it (record the assistant turn that failed to answer).
   For `contradiction`: **both** conflicting spans.
   For `ambiguous_reference`: the ambiguous mention **and** the ≥2 candidate referents.
   For `resolved`: the span that discharges the last open item.
4. `confidence` ∈ {high, medium, low}. `low` ⇒ re-route to `indeterminate`.

### 4.2 Decision order (apply in order, stop at first that fires)

1. Is there a user request with no answer in any later assistant turn? → `unanswered_question`.
2. Are two task-critical facts in conflict with no reconciliation? → `contradiction`.
3. Is a referent needed for the next action under-determined? → `ambiguous_reference`.
4. Otherwise, if all requests answered, no conflict, referents fixed → `resolved`.
5. If any of 1–3 is arguable both ways → `indeterminate`.

### 4.3 What does *not* count

- The user repeating something the assistant asked them to repeat (read-back) — that is `resolved`.
- The user restating a preference for emphasis after it was already accepted — `resolved`.
- The assistant asking a clarifying question that the user then answers — `resolved`.
- Stylistic verbosity, hedging, filler.
- A question the user themself withdraws or answers.

### 4.4 Generation vs. adjudication separation

- Dialogues are **drafted by Claude (Opus 5)**, not by the model under test (`gemma-3-*-it`).
- Ground truth is **not** set by the model under test.
- The model under test never evaluates its own outputs as truth.
- Every item then passes (a) the rule-based validator and (b) a **blind re-annotation pass** by an
  independent agent that sees only shuffled dialogue text with no condition id, no family id and no
  intended label. Agreement (Cohen's κ) between the intended label and the blind pass is reported.
- **This is LLM-assisted authoring with an independent blind LLM re-annotation and rule checks. It is
  NOT human annotation and NOT clinical annotation and must never be described as either.** No
  second human annotator was available; this is reported as a limitation, not papered over.

### 4.5 Forbidden content

No mention of dementia, cognitive decline, memory problems, diagnosis, age as an explanation, or any
health attribution. The dialogues are about ordinary everyday tasks. Rationale: the study must not
let the model (or the annotator) shortcut from a health cue to the label, and must not imply the
construct measures cognition.

### 4.6 Post-hoc deletion ban

Items may not be removed after model results are seen. The inclusion pipeline and all counts are
reported in full. Any exclusion must be timestamped **before** the split freeze.

---

## 5. Domains and stratification

Five task domains, equal target counts:

`appointment`, `shopping`, `transit`, `cooking`, `household` (family/home scheduling & maintenance).

Stage 1 gate set: **30 families** (6 per domain) = 120 dialogues.
Stage 2 full set: **120 families** (24 per domain) = 480 dialogues + ~48 indeterminate items.

---

## 6. Split rule (frozen at Stage 2)

- Split granularity is the **family**. All 4 conditions, all paraphrases and all generation seeds of
  a family live in the same split. No family crosses a split.
- 120 families → **72 train / 24 dev / 24 test**, stratified by domain (train 14–15, dev 5, test 5
  per domain, exact counts recorded).
- One domain is additionally reserved as a **transfer probe**: `household` families are marked so a
  leave-one-domain-out generalisation check can be run without creating a second small test set.
- Test labels are read **once**, at Stage 4. Dev is the only tuning surface.
- The split assignment is written to `data/bridge2026/splits.json` with a SHA-256 of the item file;
  any later change invalidates the freeze.

---

## 7. Hook-site acceptance specification (E0)

The audit found capture (`hidden_states[layer+1]`) and intervention (decoder-layer forward hook) can
disagree, and that Gemma3's last hidden state has the final norm applied while the last decoder
output does not. **One site is used everywhere.**

**Canonical site** — `resid_post(L)` := the output hidden state of decoder layer `L`, i.e. the tensor
returned by `model.model.language_model.layers[L]`, **before** any final norm. This is the tensor
Gemma Scope 2 `resid_post` SAEs are trained on. All of capture, probing, SAE encode/decode, steering
and patching attach to this same site, via the same hook implementation.

Acceptance checks that must pass before any Stage 2 result is reported:

| id | check | pass criterion |
|---|---|---|
| E0.1 | hook-vs-`output_hidden_states` identity for `L < n_layers-1` | max abs diff `== 0` (same dtype) |
| E0.2 | final-layer divergence is real and explained | `hidden_states[n_layers]` ≠ hook at `L=n_layers-1`; ratio of norms recorded |
| E0.3 | zero intervention is a no-op | steering with `alpha=0` reproduces clean logits bit-exactly |
| E0.4 | self-patch is a no-op | patching a capture from the *same* prompt reproduces clean logits bit-exactly |
| E0.5 | SAE round-trip | `fvu` (fraction of variance unexplained) reported per layer on in-domain activations; L0 matches the released config within 20% |
| E0.6 | SAE identity path | `h' = h + W_dec(z'-z)` with `z'=z` reproduces clean logits bit-exactly |
| E0.7 | determinism | same prompt twice ⇒ identical logits |
| E0.8 | dose linearity | `‖Δh‖ / ‖h‖` at the hook equals `|alpha|·sd_train(h·d)/‖h‖` within 1e-3 |

---

## 8. Readout protocol

- Dialogue rendered with the model's own chat template, ending after the final **user** turn.
- **Representation position** (for probes / directions / patching): the last token of the final user
  turn in the *pure dialogue* rendering — no answer options, no generation prompt appended, so the
  direction cannot encode option tokens.
- **Behavioural readout**: a separate rendering that appends the A/B question. Option order is
  randomised per item; each item is scored under **both** orders and the paired result is used, so
  option-position bias cancels.
- Two natural-language label wordings are used as a robustness check (`W1`, `W2`).
- Scoring: log-probability of the single option letter token (`A` / `B`), verified to be a single
  token for each model's tokenizer. This avoids the multi-token length/prior confound in the old
  `readout_baseline`.
- Sign convention: **fixed** as `score = logP(option meaning "unresolved") - logP(option meaning
  "resolved")`, identical for readout, steering and patching. (The old code mixed per-row own-vs-
  opposite with fixed positive-vs-negative; that is eliminated.)

---

## 9. Stage 1 exit gate

1. 30 families × 4 conditions authored, schema-valid.
2. Validator passes on 100% of items for: turn count match, tail-anchor identity, length band,
   question-mark band, repetition factor thresholds, forbidden-content scan, metadata-leak scan.
3. Every item carries evidence spans that a reviewer can check against the text; every positive
   has a quotable span, every negative has a discharge span.
4. Blind re-annotation agreement on the 30-family set is high enough that the differences are
   *decidable*; items that the blind pass disputes are either fixed or moved to `indeterminate`,
   and the dispute is logged.
5. Split rule and hook acceptance spec written (this document).

Only then may Stage 2 begin.
