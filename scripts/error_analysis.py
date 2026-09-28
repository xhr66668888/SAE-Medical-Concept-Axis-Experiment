#!/usr/bin/env python3
"""Stage-4 deliverable: the error picture, not a highlight reel.

Produces (a) a structured breakdown of where the frozen system fails, sliced by every pre-registered
factor, and (b) a case book of concrete failures with the full dialogue and the numbers attached.
Cases are selected by rule (most confident errors, sign flips, condition extremes), never by hand,
so the case book cannot be curated after the fact.
"""
from __future__ import annotations

import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_items(path):
    return {r["item_id"]: r for r in (json.loads(l) for l in open(path))}


def dialogue_md(item):
    return "\n".join(f"> **{t['speaker']}**: {t['text']}" for t in item["turns"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", required=True, help="e1_model_readout_*_per_item.csv or e3 per-item")
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--arm", default="", help="which arm counts as the system under analysis")
    ap.add_argument("--compare-arm", default="", help="optional second arm, for flip analysis")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_items(args.items)
    rows = list(csv.DictReader(open(args.per_item)))
    for r in rows:
        r["score"] = float(r["score"]); r["y"] = int(r["y"]); r["pred"] = int(r["pred"])
    arms = sorted({r["arm"] for r in rows})
    arm = args.arm or arms[0]
    main_rows = [r for r in rows if r["arm"] == arm]
    if not main_rows:
        raise SystemExit(f"arm {arm!r} not in {arms}")

    L, A = [], lambda s: L.append(s)
    A(f"# Error analysis — arm `{arm}`\n")
    A(f"Source: `{args.per_item}` ({len(main_rows)} items, "
      f"{len({r['family_id'] for r in main_rows})} families). Arms present: {arms}\n")

    # ---- slices ----
    def slice_table(keyfn, title):
        A(f"## {title}\n")
        g = defaultdict(list)
        for r in main_rows:
            g[keyfn(r)].append(r)
        A("| slice | n | accuracy | hit rate | false alarm | mean score |")
        A("|---|---|---|---|---|---|")
        for k in sorted(g):
            rs = g[k]
            y = np.array([r["y"] for r in rs]); p = np.array([r["pred"] for r in rs])
            s = np.array([r["score"] for r in rs])
            acc = float((y == p).mean())
            hit = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
            fa = float(p[y == 0].mean()) if (y == 0).any() else float("nan")
            A(f"| `{k}` | {len(rs)} | {acc:.3f} | {hit:.3f} | {fa:.3f} | {s.mean():+.3f} |")
        A("")

    slice_table(lambda r: r["condition"], "By condition (the 2x2 cells)")
    slice_table(lambda r: r.get("subtype", "?"), "By evidence subtype")
    slice_table(lambda r: r["domain"], "By domain")
    slice_table(lambda r: r["label"], "By label")

    # ---- family-level: does the system get both cells of a pair right? ----
    byfam = defaultdict(dict)
    for r in main_rows:
        byfam[r["family_id"]][r["condition"]] = r
    full, partial, none_ = 0, 0, 0
    for fam, d in byfam.items():
        ok = sum(int(r["pred"] == r["y"]) for r in d.values())
        if ok == len(d):
            full += 1
        elif ok == 0:
            none_ += 1
        else:
            partial += 1
    A("## Family-level consistency\n")
    A(f"- families with **all 4 cells** correct: {full}")
    A(f"- families with a mix: {partial}")
    A(f"- families with **no** cell correct: {none_}\n")

    # ---- error inventory ----
    errs = [r for r in main_rows if r["pred"] != r["y"]]
    A("## Error inventory\n")
    A(f"- total errors: {len(errs)} / {len(main_rows)} ({len(errs)/len(main_rows):.1%})")
    A(f"- misses (unresolved called resolved): {sum(r['y']==1 for r in errs)}")
    A(f"- false alarms (resolved called unresolved): {sum(r['y']==0 for r in errs)}")
    A(f"- by condition: {dict(Counter(r['condition'] for r in errs))}")
    A(f"- by subtype: {dict(Counter(r.get('subtype','?') for r in errs))}")
    A(f"- by domain: {dict(Counter(r['domain'] for r in errs))}\n")

    # ---- flip analysis against a comparison arm ----
    if args.compare_arm:
        comp = {r["item_id"]: r for r in rows if r["arm"] == args.compare_arm}
        fl = {"fixed": [], "broken": [], "both_wrong": [], "both_right": []}
        for r in main_rows:
            c = comp.get(r["item_id"])
            if c is None:
                continue
            a_ok, b_ok = r["pred"] == r["y"], c["pred"] == c["y"]
            key = ("both_right" if a_ok and b_ok else "both_wrong" if not a_ok and not b_ok
                   else "fixed" if a_ok else "broken")
            fl[key].append((r, c))
        A(f"## Change vs `{args.compare_arm}`\n")
        A(f"- fixed by `{arm}`: {len(fl['fixed'])}")
        A(f"- broken by `{arm}`: {len(fl['broken'])}")
        A(f"- wrong in both: {len(fl['both_wrong'])}")
        A(f"- right in both: {len(fl['both_right'])}\n")
        for key in ("broken", "fixed"):
            if not fl[key]:
                continue
            A(f"### Items {key}\n")
            for r, c in fl[key][:args.n_cases]:
                A(f"- `{r['item_id']}` ({r['condition']}, {r.get('subtype','')}): "
                  f"{args.compare_arm} score {c['score']:+.3f} -> {arm} score {r['score']:+.3f} "
                  f"(true = {r['label']})")
            A("")

    # ---- case book: rule-selected, not hand-picked ----
    A("## Case book\n")
    A("Selected by rule: the most confidently wrong items in each direction, then the items closest "
      "to the decision boundary. No manual curation.\n")
    misses = sorted([r for r in errs if r["y"] == 1], key=lambda r: r["score"])[:args.n_cases // 2]
    fas = sorted([r for r in errs if r["y"] == 0], key=lambda r: -r["score"])[:args.n_cases // 2]
    borderline = sorted(main_rows, key=lambda r: abs(r["score"]))[:4]
    for title, group in (("Confident misses (truly unresolved, called resolved)", misses),
                         ("Confident false alarms (truly resolved, called unresolved)", fas),
                         ("Borderline items (|score| closest to zero)", borderline)):
        A(f"### {title}\n")
        if not group:
            A("_none_\n"); continue
        for r in group:
            it = items.get(r["item_id"])
            A(f"#### `{r['item_id']}` — {r['condition']} / {r.get('subtype','')} / {r['domain']}\n")
            A(f"true label **{r['label']}**, score **{r['score']:+.3f}**, "
              f"predicted **{'unresolved' if r['pred'] else 'resolved'}**\n")
            if it:
                A(dialogue_md(it) + "\n")
                if it.get("evidence_spans"):
                    A("Annotated evidence:\n")
                    for sp in it["evidence_spans"]:
                        A(f"- turn {sp.get('turn_index')}: \"{sp.get('quote','')}\" "
                          f"_({sp.get('role','')})_")
                    A("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"errors {len(errs)}/{len(main_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
