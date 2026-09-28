# Author annotation pass

Open the annotator: **https://claude.ai/artifact/T8zi9o6LP4PRvbyRyuC7bf**

Two keystrokes per item: `1` unresolved / `2` resolved / `3` indeterminate, then `q` `w` `e` for the
subtype. `c` cycles confidence, `x` clears the current item, `←` `→` navigate, `?` toggles the
decision procedure. Answers save as you go and the page reopens where you stopped.

**The first 50 items are already a balanced sample** (8 rep-unres / 8 rep-res / 7 norep-unres /
7 norep-res / 20 CCPE-M), so stopping at 50 still gives a valid agreement estimate with a wider
interval. All 100 is better.

Do not open `data/`, `runs/` or `.key.json` while annotating — that is what makes this pass count.

When done, either tell Claude and it will read the answers out of the artifact store, or press
**copy CSV**, paste into `human_annotation/answers_filled.csv`, and run:

```bash
./venv/bin/python scripts/score_human_agreement.py
```

It reports two things separately: agreement with the *intended* labels on the constructed items
(does the designed contrast survive a human reader?) and agreement with the *blind LLM pass* on the
CCPE-M items (is the external-test labelling trustworthy?).
