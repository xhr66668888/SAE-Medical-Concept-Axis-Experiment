#!/usr/bin/env python3
"""full-vs-swapped: how much does the dialogue HISTORY move the judgement?

`swapped` hands the model the opposite-label partner's history together with the item's own final
user turn. Within a family the tail anchor is character-identical, so the only thing that changed
is the history. If the model ignores history, swapped == full. If the model follows the history,
swapped flips below chance while full sits above it.

The contrast paired_acc(full) - paired_acc(swapped) is therefore a direct, sign-meaningful measure
of context use, and it is immune to the constant option/wording bias that makes the absolute
numbers hard to read.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import stats as ST


def load(path, arm_sub):
    rows = []
    for r in csv.DictReader(open(path)):
        if arm_sub in r["arm"]:
            rows.append({"family_id": r["family_id"], "repetition": r["repetition"],
                         "label": r["label"], "condition": r["condition"],
                         "score": float(r["score"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = []
    for a, b in (("zero_shot/full", "zero_shot/swapped"),
                 ("zero_shot/full", "zero_shot/shuffled"),
                 ("zero_shot/full", "zero_shot/last_only")):
        ra, rb = load(args.per_item, a), load(args.per_item, b)
        if not ra or not rb:
            continue
        pa, pb = ST.make_pairs(ra), ST.make_pairs(rb)
        fams = sorted({p["family_id"] for p in pa} & {p["family_id"] for p in pb})
        fa = {f: [p for p in pa if p["family_id"] == f] for f in fams}
        fb = {f: [p for p in pb if p["family_id"] == f] for f in fams}
        rng = np.random.default_rng(args.seed)
        point = ST.paired_accuracy_from(pa) - ST.paired_accuracy_from(pb)
        draws = np.empty(args.boot)
        for i in range(args.boot):
            pick = rng.integers(0, len(fams), size=len(fams))
            sa = [p for k in pick for p in fa[fams[k]]]
            sb = [p for k in pick for p in fb[fams[k]]]
            draws[i] = ST.paired_accuracy_from(sa) - ST.paired_accuracy_from(sb)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        row = {"contrast": f"{a} - {b}",
               "paired_a": ST.paired_accuracy_from(pa), "paired_b": ST.paired_accuracy_from(pb),
               "diff": point, "lo": float(lo), "hi": float(hi),
               "p_two_sided": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
               "n_families": len(fams)}
        out.append(row)
        print(f"  {row['contrast']:38s} {row['paired_a']:.3f} - {row['paired_b']:.3f} = "
              f"{point:+.3f} [{lo:+.3f},{hi:+.3f}] p={row['p_two_sided']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
