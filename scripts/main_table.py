#!/usr/bin/env python3
"""The Main Table.

Rule: for anything that was frozen on dev, this reports the number at the FROZEN setting, not the
best setting on test. A "best layer on test" figure is a selected maximum and is reported only
where it is explicitly labelled as such.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path


def rd(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else []


def jd(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def g(r, k, n=3):
    try:
        return f"{float(r[k]):.{n}f}"
    except (KeyError, TypeError, ValueError):
        return "-"


def ci(r, k, n=3):
    if not r or r.get(k) in (None, ""):
        return "-"
    s = g(r, k, n)
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    if lo not in (None, "") and hi not in (None, ""):
        s += f" [{float(lo):.{n}f}, {float(hi):.{n}f}]"
    return s


def pick(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bridge2026/stage4/MAIN_TABLE.md")
    a = ap.parse_args()
    S2, S3, S4 = "runs/bridge2026/stage2", "runs/bridge2026/stage3", "runs/bridge2026/stage4"
    cfg4 = jd("runs/bridge2026/frozen_config_4b.json")
    cfg12 = jd("runs/bridge2026/frozen_config_12b.json")
    L = []
    A = L.append

    A("# BRIDGE 2026 — Main Table (test split, one-shot)\n")
    sp = jd("data/bridge2026/splits.json")
    A(f"Dataset: {sp['n_families']} families / {sp['n_items_main']} main items "
      f"+ {sp['n_items_ambiguity']} ambiguity items. Split frozen "
      f"`{sp['content_hash'][:16]}…` at `{sp['frozen_at']}`. "
      f"Test = {sp['item_counts']['test']} items / 25 families.\n")
    A(f"Frozen 4B: layer {cfg4['layer']}, span `{cfg4['position_mode']}`, SAE k={cfg4['sae']['k']}. "
      f"Frozen 12B: layer {cfg12['layer']}, span `{cfg12['position_mode']}`, SAE k={cfg12['sae']['k']}. "
      "Both selected on dev; neither changed after any test item was scored.\n")
    A("Primary metric is the **within-family paired contrast**: within a family, at a matched "
      "repetition level, does the unresolved member score higher than the resolved one? Any "
      "constant option-position, wording or label-prior bias cancels exactly. Chance = 0.500.\n")

    # ---------------- Panel A: can anything read the label off the text? ----------------
    A("## Panel A — reading the interaction state off the dialogue (test)\n")
    A("| system | paired acc | perm p | AUROC | false alarm on resolved |")
    A("|---|---|---|---|---|")
    base = rd(f"{S4}/e1_baselines_test.csv")
    for nm in ("majority", "rep_rule(>=5-token echo)", "surface_lr(19 feats)",
               "tfidf_word(1-2)", "tfidf_char_wb(3-5)", "tfidf_word+char"):
        r = pick(base, method=nm)
        if r:
            A(f"| {nm} | {ci(r,'paired_acc')} | {g(r,'paired_perm_p',4)} | {ci(r,'auroc')} | "
              f"{g(r,'false_alarm_on_resolved')} |")
    for tag, path in (("gemma-3-4b-it", f"{S4}/e1_model_readout_test.csv"),
                      ("gemma-3-12b-it", f"{S4}/e1_model_readout_test_12b.csv")):
        mr = rd(path)
        for arm in ("zero_shot/full", "prompt_context_instruction/full", "few_shot_k4/full"):
            r = pick(mr, arm=arm)
            if r:
                A(f"| **{tag}** {arm} | {ci(r,'paired_acc')} | {g(r,'paired_perm_p',4)} | "
                  f"{ci(r,'auroc')} | {g(r,'false_alarm')} |")
    A("")

    # ---------------- Panel B: is it in the residual stream, beyond surface? -------------
    A("## Panel B — is it in the residual stream, beyond surface statistics? (test)\n")
    A("A 19-feature surface model is the reference, because the paired-level shortcut audit showed "
      "surface features separate the labels within families. The question is the **increment**.\n")
    A("| model | layer | probe paired | surface paired | increment (joint - surface) | p |")
    A("|---|---|---|---|---|---|")
    for tag, path, cfg in (("4B", f"{S4}/e2_incremental_test.csv", cfg4),
                           ("12B", f"{S4}/e2_incremental_test_12b.csv", cfg12)):
        inc = rd(path)
        fr = pick(inc, layer=str(cfg["layer"]))
        if fr:
            A(f"| {tag} | **{cfg['layer']} (frozen)** | {g(fr,'probe_paired')} "
              f"| {g(fr,'surface_paired')} | {g(fr,'increment')} "
              f"[{g(fr,'increment_lo')}, {g(fr,'increment_hi')}] | {g(fr,'increment_p',4)} |")
        if inc:
            best = max(inc, key=lambda r: float(r["increment"]))
            A(f"| {tag} | {best['layer']} (best on test, selected) | {g(best,'probe_paired')} "
              f"| {g(best,'surface_paired')} | {g(best,'increment')} "
              f"[{g(best,'increment_lo')}, {g(best,'increment_hi')}] | {g(best,'increment_p',4)} |")
    A("")

    # ---------------- Panel C: does the history cause the judgement? ----------------
    A("## Panel C — does the dialogue history cause the judgement? (test)\n")
    A("`swapped` replaces the history with the opposite-label partner's, keeping the item's own "
      "final user turn. The tail anchor is character-identical within a family, so only the history "
      "changed. A model that ignores history scores the same as `full`; a model that follows it "
      "falls below chance.\n")
    A("| model | contrast | full | ablated | difference | 95% CI | p |")
    A("|---|---|---|---|---|---|---|")
    for tag, path in (("4B", f"{S4}/context_contrast_test_4b.csv"),
                      ("12B", f"{S4}/context_contrast_test_12b.csv")):
        for r in rd(path):
            A(f"| {tag} | {r['contrast'].replace('zero_shot/','')} | {g(r,'paired_a')} | "
              f"{g(r,'paired_b')} | {g(r,'diff')} | [{g(r,'lo')}, {g(r,'hi')}] | {g(r,'p_two_sided',4)} |")
    A("")

    # ---------------- Panel D: causal selectivity ----------------
    A("## Panel D — causal selectivity (test)\n")
    A("`delta target` = shift on **unresolved** items (should move). `delta non-target` = shift on "
      "**resolved** items (should not). `gap` is the selectivity. Sign: + = toward 'unresolved'.\n")
    for tag, path in (("4B (layer 29, span read_to_end)", f"{S4}/e3_causal_test.csv"),
                      ("12B (layer 31, span prompt_all)", f"{S4}/e3_causal_test_12b.csv")):
        ca = rd(path)
        if not ca:
            continue
        A(f"### {tag}\n")
        A("| arm | delta target | 95% CI | delta non-target | **gap** | paired acc | false alarm |")
        A("|---|---|---|---|---|---|---|")
        for r in ca:
            A(f"| `{r['arm']}` | {g(r,'delta_target')} | [{g(r,'delta_target_lo')}, "
              f"{g(r,'delta_target_hi')}] | {g(r,'delta_nontarget')} | **{g(r,'selectivity_gap')}** | "
              f"{g(r,'paired_acc')} | {g(r,'false_alarm')} |")
        gaps = [abs(float(r["selectivity_gap"])) for r in ca if r.get("selectivity_gap")]
        A(f"\nLargest absolute selectivity gap across {len(ca)} arms: **{max(gaps):.3f}**\n")

    # ---------------- Panel E: external ----------------
    A("## Panel E — external test, CCPE-M (real human dialogue)\n")
    A("100 segments, blind-annotated under the same guidelines: 97 resolved, 3 unresolved, 0 "
      "indeterminate. With that distribution discrimination metrics are meaningless and are not "
      "reported. This measures the **false-trigger rate on ordinary human conversation**. "
      "CCPE-M has no cognitive-state annotation; it is not a healthy control group and not "
      "clinical validation.\n")
    A("| model | arm | false alarm | 95% CI | FA on segments WITH a >=5-token echo | FA WITHOUT |")
    A("|---|---|---|---|---|---|")
    for tag, path in (("4B", f"{S4}/e4_external_4b.csv"), ("12B", f"{S4}/e4_external_12b.csv")):
        for r in rd(path):
            A(f"| {tag} | `{r['arm']}` | {g(r,'false_alarm')} | [{g(r,'false_alarm_lo')}, "
              f"{g(r,'false_alarm_hi')}] | {g(r,'false_alarm_rep')} | {g(r,'false_alarm_norep')} |")
    A("")

    # ---------------- Panel F: free-form ----------------
    ff = rd(f"{S4}/e4_freeform_test.csv")
    if ff:
        A("## Panel F — free-form next-turn responses (test), rule-based measures\n")
        A("| arm | n | mean questions | multi-question rate | echo of user | cognitive inference | words |")
        A("|---|---|---|---|---|---|---|")
        for arm in sorted({r["arm"] for r in ff}):
            rs = [r for r in ff if r["arm"] == arm]
            m = lambda k: sum(float(r[k]) for r in rs) / len(rs)
            A(f"| `{arm}` | {len(rs)} | {m('n_questions'):.2f} | {m('multi_question'):.2f} | "
              f"{m('echo_of_user'):.2f} | {m('cognitive_inference'):.3f} | {m('resp_words'):.0f} |")
        A("")

    # ---------------- Panel G: blind rubric ----------------
    rb = rd(f"{S4}/rubric_summary.csv")
    if rb:
        A("## Panel G — free-form responses, blind rubric (test)\n")
        A("400 responses (100 test dialogues x 4 arms), arm names stripped and order randomised, "
          "scored by four independent reviewers who each saw only one shard.\n")
        A("| arm | takes up a genuinely open issue | false trigger when nothing is open | "
          "keeps task facts | cognitive inference | overall appropriate |")
        A("|---|---|---|---|---|---|")
        for r in rb:
            A(f"| `{r['arm']}` | {ci(r,'addresses_open_issue_on_unresolved')} | "
              f"{ci(r,'false_trigger_on_resolved')} | {ci(r,'maintains_task_facts')} | "
              f"{ci(r,'cognitive_inference')} | {ci(r,'overall_appropriate')} |")
        A("")
        orig = pick(rb, arm="original")
        if orig:
            A("Target improvement vs non-target damage, relative to `original`:\n")
            A("| arm | change in taking up open issues | change in false triggers |")
            A("|---|---|---|")
            for r in rb:
                if r["arm"] == "original":
                    continue
                d1 = float(r["addresses_open_issue_on_unresolved"]) - float(orig["addresses_open_issue_on_unresolved"])
                d2 = float(r["false_trigger_on_resolved"]) - float(orig["false_trigger_on_resolved"])
                A(f"| `{r['arm']}` | {d1:+.3f} | {d2:+.3f} |")
            A("")

    # ---------------- Panel H: did the interventions change the text at all? ----------
    ff2 = rd(f"{S4}/e4_freeform_test.csv")
    if ff2:
        from collections import defaultdict as _dd
        by = _dd(dict)
        for r in ff2:
            by[r["item_id"]][r["arm"]] = r["response"]
        A("## Panel H — did the interventions change the generated reply at all? (test)\n")
        A("| arm | responses byte-identical to `original` |")
        A("|---|---|")
        for arm in sorted({r["arm"] for r in ff2}):
            if arm == "original":
                continue
            same = sum(1 for d in by.values() if d.get(arm) == d.get("original"))
            A(f"| `{arm}` | {same}/{len(by)} ({same/len(by):.0%}) |")
        A("")

    # ---------------- Panel I: ambiguity set ----------------
    amb = rd(f"{S4}/ambiguity_4b.csv")
    if amb:
        A("## Panel I — behaviour on the 48 deliberately undecidable items (4B)\n")
        A("Routed to the ambiguity set before any model was run. There is no ground truth to be "
          "accurate against; what matters is whether the model is *less certain* here.\n")
        A("| set | n | mean score | mean abs score | fraction near indifference (abs < 1) | "
          "fraction predicting 'unresolved' |")
        A("|---|---|---|---|---|---|")
        for r in amb:
            A(f"| {r['set']} | {r['n']} | {g(r,'mean_score')} | {g(r,'mean_abs_score')} | "
              f"{g(r,'frac_abs_score_lt_1')} | {g(r,'frac_pred_unresolved')} |")
        A("")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"-> {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
