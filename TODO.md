# Outstanding work

Status: Stages 1–4 complete, one-shot test read, paper drafted. Workshop deadline **2026-10-05
23:59 AoE** (extended from 09-27; re-verify before submitting).

---

## 1. Author annotation pass — DEFERRED

**What it is.** 100 blind items (60 constructed + 40 CCPE-M) prepared in `human_annotation/`, with a
keyboard-driven tool at <https://claude.ai/artifact/T8zi9o6LP4PRvbyRyuC7bf> (~30–40 min; answers
persist server-side; the first 50 are already a balanced sample if you stop early). Scoring script
is written and ready: `scripts/score_human_agreement.py`.

**Why it was queued.** Every label in this study currently comes from an LLM. Claude drafted the
dialogues *and* set the intended labels; the blind re-annotation (κ=0.950) was four more Claude
passes. That is reported honestly, but it means there is no human anywhere in the label chain.

**What deferring costs.** The paper must keep saying, in the abstract's vicinity and in
Limitations, that annotation is *not human and not clinical*. Reviewers at a health-adjacent
workshop are likely to press on this. Doing the pass would convert the claim to "LLM-drafted, with
an author's independent annotation on a 100-item subset (κ=…)", which is a materially stronger
position.

**The sharpest reason to reconsider.** The headline finding — models call 84–100% of ordinary human
conversation "unresolved" — rests entirely on the CCPE-M labels (97 resolved / 3 unresolved), and
those labels are LLM-produced. 40 of the 100 packet items are exactly those CCPE-M segments. Even
annotating only that subset would put a human under the paper's strongest claim.

**If picked up:** open the tool, or fill `human_annotation/answers.csv` from
`human_annotation/items.md`, then run `./venv/bin/python scripts/score_human_agreement.py`.
It reports constructed-vs-intended and CCPE-M-vs-blind-LLM separately.

---

## 2. Before submission

- [ ] **Compile the paper.** No LaTeX toolchain on this machine, so `paper/main.tex` has never been
      built. Page estimate is 3.6 / 4.0 — needs confirming, and the two figures need a look at
      final size. Overleaf is fastest.
- [ ] **Re-verify the deadline** on the workshop site (the plan flags that it can move; it already
      moved once).
- [ ] **Check two references** I could only cite by URL: the Assistant Axis post and the ClarifySAE
      repository. Confirm whether either has a formal paper version.
      The other seven were fetched and verified against source.
- [ ] Confirm the paper is double-blind clean (no author names, no institution, no absolute paths
      in the reproducibility note).

## 3. Optional strengthening, in value order

- [ ] **Expand CCPE-M** from 100 to ~300 segments. Narrows the 4B false-alarm interval
      ([0.784, 0.919] now); needs ~200 more blind annotations.
- [ ] **A fourth model family** (Llama-3.1 is gated but the token may reach it; Mistral is open).
      Three models across two families already support the generality claim; a third family from a
      different pretraining lineage would make it hard to argue away.
- [ ] **Reduce the corpus leak.** 9 of 19 surface features separate labels at the paired level
      (type-token ratio 0.706). Partly intrinsic — a `resolved` cell must contain the discharge, so
      it runs longer and less lexically diverse — so this is a re-authoring project with a real
      ceiling, not a bug fix. The incremental-over-surface analysis already handles it correctly;
      only attempt this if a reviewer demands it.
- [ ] **Multi-turn dynamics.** Every intervention here acts on the prompt pass. The feedback loop
      by which a model's reply changes the next user turn is unmeasured and is the natural
      follow-up study.

## 4. Housekeeping

- [ ] `_attic/eeg-ema-sample-code/` is EEG/EMA signal-processing code from an unrelated project
      that was sitting in the repo root. Kept rather than deleted — decide whether to remove it.
- [ ] `runs/bridge2026/cache/*.npz` (463 MB of captured activations) is gitignored. Regenerate with
      `scripts/capture_acts.py` rather than committing it.
- [ ] `pilot_icd/` code expects to run with `pilot_icd/` as the working directory; its imports were
      not rewritten for the new location.
