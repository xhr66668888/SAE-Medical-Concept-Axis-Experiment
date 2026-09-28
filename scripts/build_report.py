#!/usr/bin/env python3
"""Assemble the main table and the experiment report from whatever result files exist.

Design rules carried over from the audit of the old pipeline:
  * NO truncation. Every condition, every arm and every layer that was run is printed. The old
    report silently showed only the top 8 axes and hid two negative results.
  * Negative results are printed in the same table as positive ones, with the same formatting.
  * Every number carries its family-clustered interval.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path


def read_csv(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else []


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=3):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "-"


def ci(r, k, n=3):
    if r.get(k) in (None, ""):
        return "-"
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    s = f(r[k], n)
    if lo not in (None, "") and hi not in (None, ""):
        s += f" [{f(lo,n)}, {f(hi,n)}]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/bridge2026")
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    R = Path(args.rundir); S = R / args.stage
    out = Path(args.out) if args.out else S / f"REPORT_{args.split}.md"
    L = []
    A = L.append

    A(f"# BRIDGE 2026 — results report ({args.stage}, eval split = `{args.split}`)\n")
    splits = read_json("data/bridge2026/splits.json")
    if splits:
        A("## Dataset freeze\n")
        A(f"- frozen at `{splits['frozen_at']}`, seed `{splits['seed']}`")
        A(f"- content hash `{splits['content_hash'][:16]}…`")
        A(f"- {splits['n_families']} families, {splits['n_items_main']} main items, "
          f"{splits['n_items_ambiguity']} ambiguity items")
        A(f"- item counts: `{splits['item_counts']}`")
        A(f"- transfer-probe domain: `{splits['transfer_domain']}`\n")
        A("| domain | families | train | dev | test |")
        A("|---|---|---|---|---|")
        for d, v in splits["per_domain"].items():
            A(f"| {d} | {v['n_families']} | {v['train']} | {v['dev']} | {v['test']} |")
        A("")

    e0 = read_json(S / "e0_acceptance.json") or read_json(R / "stage1/e0_smoke.json")
    if e0:
        A("## E0 — implementation acceptance\n")
        A(f"model `{e0['model']}`, dtype `{e0.get('dtype')}`, {e0['n_layers']} layers, "
          f"d_model {e0['d_model']}, acceptance set n={e0.get('n_items','?')}\n")
        A("| check | pass | note |")
        A("|---|---|---|")
        for k, v in e0["checks"].items():
            note = v.get("note", "")
            if k.startswith("E0.2"):
                note = f"norm ratio hs/hook = {f(v.get('norm_ratio_hs_over_hook'),4)}. " + note
            if k.startswith("E0.5"):
                pl = v["per_layer"]
                note = "; ".join(f"L{l}: L0={f(x['l0_mean'],1)}/{x['l0_cfg']}, cos={f(x['cos_mean'],4)}, "
                                 f"rel_err={f(x['rel_err'],3)}" for l, x in pl.items())
            A(f"| `{k}` | {'PASS' if v.get('pass') else 'FAIL'} | {note} |")
        A(f"\n**ALL_PASS = {e0['all_pass']}**\n")

    base = read_csv(S / f"e1_baselines_{args.split}.csv")
    if base:
        A("## E1a — text-only and surface baselines\n")
        A("All read only the dialogue text. Replaces the old ontology-oracle 'lexical baseline'.\n")
        A("| method | n | accuracy | macro-F1 | AUROC | hit rate | false alarm |")
        A("|---|---|---|---|---|---|---|")
        for r in base:
            A(f"| `{r['method']}` | {r['n']} | {ci(r,'acc')} | {ci(r,'macro_f1')} | {ci(r,'auroc')} | "
              f"{f(r.get('hit_rate_unresolved'))} | {f(r.get('false_alarm_on_resolved'))} |")
        A("")
    sc = read_csv(S / f"e1_baselines_{args.split}_shortcut_audit.csv")
    if sc:
        A("### Shortcut audit — single surface feature vs the label\n")
        A("Within-family permutation; Benjamini-Hochberg at q=0.05.\n")
        A("| feature | AUROC | p (perm) | flagged |")
        A("|---|---|---|---|")
        for r in sc:
            A(f"| `{r['feature']}` | {f(r['auroc'])} | {f(r['p_perm'],4)} | "
              f"{'**YES**' if r.get('bh_flagged')=='True' else 'no'} |")
        n_flag = sum(r.get("bh_flagged") == "True" for r in sc)
        A(f"\n{n_flag} of {len(sc)} surface features flagged.\n")

    mr = read_csv(S / f"e1_model_readout_{args.split}.csv")
    if mr:
        A("## E1b — model behavioural readout and context ablations\n")
        A("Primary metric is the **within-family paired contrast**: constant option-position, "
          "wording and label-prior bias cancels exactly. Raw accuracy is a diagnostic.\n")
        A("| arm | n | paired acc | perm p | AUROC | raw acc | hit | false alarm |")
        A("|---|---|---|---|---|---|---|---|")
        for r in mr:
            A(f"| `{r['arm']}` | {r['n_items']} | {ci(r,'paired_acc')} | {f(r.get('paired_perm_p'),4)} | "
              f"{ci(r,'auroc')} | {ci(r,'acc_raw')} | {f(r.get('hit_rate'))} | {f(r.get('false_alarm'))} |")
        A("\n### Mean readout score by condition (sign convention: + = 'unresolved')\n")
        conds = ["rep_unres", "rep_res", "norep_unres", "norep_res"]
        A("| arm | " + " | ".join(f"`{c}`" for c in conds) + " |")
        A("|---|" + "---|" * len(conds))
        for r in mr:
            A(f"| `{r['arm']}` | " + " | ".join(f(r.get(f"mean_score_{c}")) for c in conds) + " |")
        A("")

    sw = read_csv(S / f"e2_layer_sweep_{args.split}.csv")
    if sw:
        A("## E2 — representation, full layer sweep (no truncation)\n")
        for meth in sorted({r["method"] for r in sw}):
            rs = [r for r in sw if r["method"] == meth]
            A(f"### `{meth}`\n")
            A("| layer | AUROC | paired acc | paired(rep) | paired(norep) | perm p | beats random null | cos(unres,rep) |")
            A("|---|---|---|---|---|---|---|---|")
            for r in sorted(rs, key=lambda x: int(x["layer"])):
                A(f"| {r['layer']} | {ci(r,'auroc')} | {ci(r,'paired_acc')} | "
                  f"{f(r.get('paired_acc_rep'))} | {f(r.get('paired_acc_norep'))} | "
                  f"{f(r.get('perm_p'),4)} | {r.get('beats_random_null','-')} | "
                  f"{f(r.get('cos_unres_rep'))} |")
            A("")

    sf = read_csv(S / f"e2_sae_features_{args.split}.csv")
    if sf:
        A("### SAE candidate features (ranked on TRAIN only)\n")
        A("| layer | rank | feature | train SMD | train AUROC | train act rate |")
        A("|---|---|---|---|---|---|")
        for r in sf:
            A(f"| {r['layer']} | {r['rank']} | {r['feature']} | {f(r['train_smd'])} | "
              f"{f(r['train_auroc'])} | {f(r['train_act_rate'])} |")
        A("")

    cc = read_csv(S / "competing_cosines.csv")
    if cc:
        A("### Competing 'generic clarification need' direction\n")
        A("| layer | probe AUROC | cos(clarify, unresolved) | cos(clarify, repetition) |")
        A("|---|---|---|---|")
        for r in cc:
            A(f"| {r['layer']} | {f(r.get('clarify_probe_auroc'))} | "
              f"{f(r.get('cos_clarify_unresolved'))} | {f(r.get('cos_clarify_repetition'))} |")
        A("")

    for name, title in ((f"e3_causal_{args.split}.csv", "E3 — causal selectivity"),
                        (f"e3_causal_{args.split}_12b.csv", "E3 — causal selectivity (12B replication)")):
        ca = read_csv(S / name)
        if not ca:
            continue
        A(f"## {title}\n")
        A("`delta_target` = change on **unresolved** items (should move). "
          "`delta_nontarget` = change on **resolved** items (should NOT move). "
          "`gap` = selectivity.\n")
        A("| arm | delta target | delta non-target | gap | paired acc | AUROC | hit | false alarm |")
        A("|---|---|---|---|---|---|---|---|")
        for r in ca:
            A(f"| `{r['arm']}` | {ci(r,'delta_target')} | {ci(r,'delta_nontarget')} | "
              f"{f(r.get('selectivity_gap'))} | {ci(r,'paired_acc')} | {ci(r,'auroc')} | "
              f"{f(r.get('hit_rate'))} | {f(r.get('false_alarm'))} |")
        A("")

    ff = read_csv(S / f"e4_freeform_{args.split}.csv")
    if ff:
        A("## Free-form response endpoint — rule-based measures\n")
        arms = sorted({r["arm"] for r in ff})
        A("| arm | n | mean questions | multi-question rate | echo of user | cognitive inference | words |")
        A("|---|---|---|---|---|---|---|")
        for a in arms:
            rs = [r for r in ff if r["arm"] == a]
            g = lambda k: sum(float(r[k]) for r in rs) / len(rs)
            A(f"| `{a}` | {len(rs)} | {g('n_questions'):.2f} | {g('multi_question'):.2f} | "
              f"{g('echo_of_user'):.2f} | {g('cognitive_inference'):.3f} | {g('resp_words'):.0f} |")
        A("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
