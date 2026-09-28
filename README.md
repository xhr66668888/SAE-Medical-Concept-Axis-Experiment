# Beyond Repetition

A mechanistic audit of whether language models can tell an **unresolved communication problem**
from **surface-similar but legitimate repetition**, and whether residual-stream or SAE
interventions change that judgement selectively.

Prepared for the BRIDGE workshop @ IEEE BIBM 2026. The construct is an *observable interaction
state* in ordinary task dialogue. **Nothing here measures cognition, decline, or diagnosis.**

---

## The result in one table

Test split, one-shot. Primary metric is the within-family paired contrast (chance = 0.500).

| | paired acc. | p | false alarm on resolved | CCPE-M false alarm |
|---|---|---|---|---|
| best text-only baseline (char 3–5 gram) | **0.760** [0.640, 0.880] | <0.001 | 0.280 | — |
| Gemma-3-4B | 0.600 [0.460, 0.720] | 0.207 | 0.700 | 0.856 [0.784, 0.919] |
| Gemma-3-12B | 0.580 [0.460, 0.700] | 0.330 | 0.920 | **1.000** |
| Qwen2.5-7B | 0.540 [0.450, 0.630] | 0.676 | 0.780 | 0.835 [0.760, 0.908] |

Three models across two families sit at chance; a bag of characters beats all of them; and on 97
ordinary human conversations annotated as having nothing open, they flag 84–100% as containing an
unresolved problem — no more often when the conversation actually repeats. No intervention among
40 arms is selective (largest gap 0.151). Full numbers:
[MAIN_TABLE.md](runs/bridge2026/stage4/MAIN_TABLE.md),
[CONCLUSIONS.md](runs/bridge2026/stage4/CONCLUSIONS.md).

## Layout

```
bridge2026/        the library: schema, validator, hook-site runtime, SAE loader,
                   readout protocol, interventions, family-clustered statistics
scripts/           the pipeline, one stage per script (see below)
data/bridge2026/   the corpus: 120 families x 4 conditions + 48 indeterminate items,
                   frozen splits, the competing-explanation probe set
data/external/     CCPE-M source and the extracted segments
runs/bridge2026/   stage1..stage5 results, frozen configs, logs
docs/              specification, authoring brief, stage-gate log, rubric, the original plan
paper/             LaTeX source, figures, and NUMBERS.json (every cited value)
human_annotation/  the prepared author-annotation packet (see TODO.md)
pilot_icd/         the earlier ICD/CCS concept-axis study, demoted to a pilot
_attic/            unrelated files found in the repo root, kept pending a decision
```

## Design, in one paragraph

Each **family** is one scenario written in four conditions: a full crossing of surface repetition
against repair state. Because the crossing is complete, a classifier keying on repetition alone
scores exactly chance — measured at 0.550 [0.490, 0.610], p=0.223. Within a family we hold constant
the entities, task goal, turn count, length (±18%), question-mark count (±1) and — critically — the
**final sentence of the last user turn, character-identical across all four conditions**, because
that is where representations are read. In 62% of pairs the entire final user turn is identical.

## Running it

```bash
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
echo "HF_TOKEN=hf_..." > .hf_env && chmod 600 .hf_env    # Gemma repos are manually gated

./venv/bin/python -m bridge2026.validate                  # corpus constraints
./venv/bin/python scripts/build_dataset.py                # assemble + freeze splits
./venv/bin/python scripts/e0_acceptance.py --device cuda:0
./venv/bin/python scripts/e0_reachability.py --device cuda:0
./venv/bin/python scripts/capture_acts.py --device cuda:0 --out runs/bridge2026/cache/acts_4b.npz
bash scripts/run_stage4_test.sh                           # the one-shot test pass
```

`runs/bridge2026/cache/*.npz` is not committed (463 MB); regenerate with `capture_acts.py`.

## Three checks that changed the answer

These are in the paper because each one, omitted, produces a confident false finding.

1. **Causal reachability** (`scripts/e0_reachability.py`). The first causal run returned exactly
   zero on every arm, including a whole-residual patch. At layer 29 the read position sits 55–67
   tokens before the answer with four layers left: a perturbation of *twice the residual norm*
   moves the final logits by 0.33 nats, against 41.3 at the final position. The single-token site
   is inert at every candidate layer in both models. A null there would have been an artefact of
   hook placement.
2. **SAE reconstruction damage.** Replacing the residual with its SAE reconstruction and editing
   nothing costs −0.919 (4B) and −2.464 (12B); ablating the candidate features costs −0.102 and
   −0.028. Without the error-preserving path and a reconstruction control, the damage reads as a
   large feature effect.
3. **Paired-level shortcut audit.** The same 19 surface features give 0/19 flagged at item level
   and 9/19 at paired level. Mean-matching the corpus hid a consistent within-pair direction, so
   every representational claim here is an increment-over-surface claim.

One hook site is used for capture, probing, SAE, steering and patching: `resid_post(L)`, the
decoder layer output before the final norm. The norm ratio between that and the final hidden state
is 0.0175 (4B) and 0.0100 (12B) — an 57–100× unit mismatch if the two are mixed.

## Honest description of the labels

Dialogues were drafted by Claude and re-annotated blind by four independent Claude passes, each
seeing exactly one condition per family (κ = 0.950). **This is not human annotation, not
multi-annotator human consensus, and not clinical annotation.** The model under test generated none
of the data and set none of the labels. See [TODO.md](TODO.md) §1.

## Provenance

`data/bridge2026/splits.json` carries a SHA-256 over the family files; the frozen intervention
configs carry a SHA-256 over the dev evidence they were derived from. Both were verified unchanged
after the test read. Every decision, including the ones that were initially wrong, is in
[docs/CHANGELOG_STAGE_GATES.md](docs/CHANGELOG_STAGE_GATES.md).

CCPE-M is CC BY 4.0 (google-research-datasets/ccpe). It carries no cognitive-state annotation and
is not a healthy control group.
