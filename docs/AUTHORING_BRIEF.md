# Authoring brief — BRIDGE 2026 scenario families

You are authoring controlled dialogue data for a mechanistic-interpretability study. Read
`docs/STAGE1_SPEC.md` first; it is binding. This brief is the operational checklist.

## What one family is

A scenario (domain + entities + task goal) realised in **4 conditions**, a full crossing of
surface repetition × repair state:

| condition | repetition | label | subtype must be one of |
|---|---|---|---|
| `rep_unres`   | present | `unresolved` | `unanswered_question`, `contradiction`, `ambiguous_reference` |
| `rep_res`     | present | `resolved`   | `readback_confirmation`, `emphasis_repeat`, `clean_progress` |
| `norep_unres` | absent  | `unresolved` | `unanswered_question`, `contradiction`, `ambiguous_reference` |
| `norep_res`   | absent  | `resolved`   | `readback_confirmation`, `emphasis_repeat`, `clean_progress` |

## Hard constraints (the validator rejects violations — you must run it)

1. Dialogue **starts with `user`**, **ends with `user`**, speakers strictly alternate ⇒ odd turn
   count. Use 5 or 7 turns. All 4 conditions of a family have the **same** turn count.
2. **Tail anchor**: the **final sentence of the last user turn is character-identical** across all
   4 conditions, and equals the family's declared `tail_anchor`. Everything before it may differ.
   The anchor must be neutral — it must not itself reveal whether anything is open.
3. **Length**: every condition within ±18% of the family's mean character count. Aim within ±6%.
4. **Question marks**: `count('?')` identical across all 4 conditions (spread ≤1, target 0).
5. **Repetition factor**, measured as the longest contiguous run of shared tokens between a user
   turn and any earlier turn:
   - `rep_*` conditions: **≥5** tokens. Build a deliberate near-verbatim echo of ≥6 words.
   - `norep_*` conditions: **≤4** tokens. Avoid echoing any phrase; vary wording deliberately.
6. **Evidence spans**: each `quote` must be an exact case-insensitive substring of the turn at
   `turn_index`. Positives need the span(s) that create the problem *and* the assistant turn that
   fails to close it. Negatives need the span that discharges the last open item.
7. **Forbidden**: no dementia / cognitive / memory-problem / diagnosis / age-as-explanation /
   impairment wording anywhere. These are ordinary everyday tasks.
8. **No leakage**: the literal words `unresolved`, `resolved`, `indeterminate`, `contradiction`, or
   any condition id must never appear in dialogue text.

## Quality bar (what makes this study work)

- **Minimal counterfactual is the goal.** Wherever you can, make `norep_unres` and `norep_res`
  differ by as little as one decisive word or clause, and likewise `rep_unres` vs `rep_res`.
  See the worked exemplar `data/bridge2026/families/appointment_001.json`, where the
  contradiction is created by `afternoons` vs `mornings` and nothing else.
- **`rep_res` must be tempting.** The repetition has to look, on the surface, exactly like trouble:
  a read-back the assistant explicitly asked for, or the user re-stating a preference that was
  already accepted. A lexical classifier should be fooled.
- **`norep_unres` must be quiet.** No repeated wording, no "sorry?", no "what do you mean" — just a
  contradiction or an under-determined referent that is genuinely still open.
- **Ambiguous reference** means ≥2 live candidates are actually present in the dialogue (e.g. two
  different appointments, two different bags, two different buses have been mentioned) and the
  user's referring expression does not pick one out. Do not invent ambiguity that has only one
  candidate in context.
- **Contradiction** must be task-critical and both sides must be explicit in the text.
- **Unanswered question**: the assistant must plausibly move on (answer a *different* part, ask for
  something else) rather than obviously stonewalling. Do not make the assistant rude.
- Natural, ordinary English. Do not distort the dialogue to hit the constraints — re-plan the
  scenario instead. Vary entities, names, tone and syntax across families; do not template.
- Subtypes: across the families you write, spread positive subtypes roughly evenly
  (`unanswered_question` / `contradiction` / `ambiguous_reference`) and the same for negatives.

## File format

One JSON file per family at `data/bridge2026/families/<family_id>.json`, exactly this shape:

```json
{
  "family_id": "shopping_003",
  "domain": "shopping",
  "task_goal": "short phrase",
  "n_turns": 5,
  "tail_anchor": "the identical final sentence",
  "notes": "what the minimal difference is",
  "conditions": {
    "rep_unres":   {"label":"unresolved","subtype":"...","confidence":"high",
                    "evidence_spans":[{"turn_index":0,"quote":"...","role":"..."}],
                    "turns":[{"speaker":"user","text":"..."},{"speaker":"assistant","text":"..."}]},
    "rep_res":     {...}, "norep_unres": {...}, "norep_res": {...}
  }
}
```

## Verify before you finish (mandatory)

```bash
cd /home/xu0064/SAE-Medical-Concept-Axis-Experiment-main
./venv/bin/python -m bridge2026.validate --out-rows /tmp/rows.csv
```

Iterate until it prints `errors=0`. Do not hand back work with any error. Do not edit
`bridge2026/validate.py`, `bridge2026/schema.py`, `docs/STAGE1_SPEC.md`, or any family file whose
`family_id` prefix was not assigned to you.

---

## Failure modes found in the first 30 families (blind re-annotation, kappa = 0.95)

An independent blind annotator disputed 3 of the first 120 items. All three were `unresolved` cells
that read as `resolved`. Learn from them; these are the ways a positive cell quietly dies.

**1. The action that creates the conflict was left implicit.**
Rejected: *"The lorry comes round about half six ... I'll deal with the first one myself on the way
out, we're off at nine that morning."* — "deal with the first one" never says the bin is put out at
nine, so the reader cannot be sure a conflict exists.
Fixed: *"I'll put the first week's bin out myself as we leave, we're off at nine that morning."*
**Rule:** for a `contradiction`, both conflicting quantities AND the action that ties them together
must be on the page. A reader should not have to supply a step.

**2. The pronoun had a topical winner.**
Rejected: two girls in the family, but the preceding turns were all about Noor, so "she" resolved to
Noor by topic continuity and the ambiguity evaporated.
Fixed: the assistant turn immediately before names the *other* girl, so the bare pronoun has both
the topic candidate and the most-recent-mention candidate.
**Rule:** for `ambiguous_reference`, check that the intended competitor is either the most recent
singular mention or is re-activated right before the referring expression. Two candidates existing
somewhere in the dialogue is not enough.

**3. The consequence of the conflict was left to inference.**
Rejected: *"Theo gets in on Friday evening and we are building it on the Saturday morning, so the
room will be ready for him."* — a reader can imagine Theo sleeping elsewhere on the Friday.
Fixed: *"Theo is sleeping in there from Friday night ..."*
**Rule:** state why the conflicting fact actually blocks the task. "Arrives before X" is weak;
"needs X from the moment they arrive" is decidable.

**Self-check for every positive cell before you write it out:** could a careful, sceptical reader who
has never seen the design read this dialogue and say "that's fine, nothing is open"? If yes, the
cell is not ready. Conversely, for every `resolved` cell: could that same reader point at something
still open? If yes, the cell is not ready either.
