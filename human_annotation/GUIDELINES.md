# Annotation guidelines (read only this; do not look for other project files)

You are labelling short task-oriented dialogues between a `user` and an `assistant`. For each
dialogue, judge the state **as of its final turn**.

## Labels

- `unresolved` — a communication problem is still open at the end of the dialogue.
- `resolved` — no open communication problem; the exchange is on track.
- `indeterminate` — the evidence is genuinely insufficient to decide.

`unresolved` requires explicit, quotable evidence of exactly one of:

- `unanswered_question` — the user requested information or an explanation and it has still not been
  provided by the assistant by the end of the dialogue.
- `contradiction` — two pieces of task-critical information conflict, and the conflict has not been
  raised or reconciled.
- `ambiguous_reference` — a referent needed to act on is still under-determined (two or more live
  candidates appear in the dialogue) and has not been disambiguated.

`resolved` requires that every user request has been answered, no task-critical contradiction is
open, and every referent needed for the next action is determined. Record which of these applies:

- `readback_confirmation` — the user repeats something because the assistant asked them to confirm it.
- `emphasis_repeat` — the user restates a preference that was already accepted.
- `clean_progress` — information is complete and consistent and the task simply proceeds.

## Decision order — apply in order, stop at the first that fires

1. Is there a user request with no answer in any later assistant turn? → `unanswered_question`
2. Are two task-critical facts in conflict with no reconciliation? → `contradiction`
3. Is a referent needed for the next action under-determined? → `ambiguous_reference`
4. Otherwise, all requests answered / no conflict / referents fixed → `resolved`
5. If any of 1–3 is genuinely arguable both ways → `indeterminate`

## What does NOT make something `unresolved`

- The user repeating something the assistant explicitly asked them to repeat.
- The user restating a preference that was already accepted.
- A clarifying question from the assistant that the user then answers.
- A single repetition on its own, disfluency, short answers, politeness, hedging or verbosity.
- A question the user withdraws or answers themself.

Repetition on its own is **never** evidence either way. Some dialogues repeat wording and are fine;
some never repeat anything and still have an open problem. Judge only whether something is open.

## Output

For every item, output an object with:
- `id` (copy it exactly)
- `label`: one of `unresolved` / `resolved` / `indeterminate`
- `subtype`: the matching subtype name from above (use `indeterminate` if the label is indeterminate)
- `evidence`: a short verbatim quote from the dialogue that decided it
- `confidence`: `high` / `medium` / `low`
