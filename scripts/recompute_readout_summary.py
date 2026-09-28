#!/usr/bin/env python3
"""Recompute readout summary metrics from a saved per-item file, under the CURRENT statistics code.

Needed because the 4B dev readout was produced before `paired_accuracy_from` was fixed to count
ties as 0.5. Ties matter here: in many families the final user turn is character-identical across
the repair-state contrast (the decisive difference sits in an earlier turn), so the `last_only`
ablation feeds the model literally the same input for both members of the pair.
"""
from __future__ import annotations

import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import stats as ST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_arm = defaultdict(list)
    for r in csv.DictReader(open(args.per_item)):
        by_arm[r["arm"]].append({**r, "score": float(r["score"]), "y": int(r["y"]), "pred": int(r["pred"])})

    rows = []
    for arm, rs in by_arm.items():
        pairs = ST.make_pairs(rs)
        n_tied = sum(1 for p in pairs if p["pos"]["score"] == p["neg"]["score"])
        pb = ST.paired_bootstrap(pairs, ST.paired_accuracy_from, n=args.boot, seed=args.seed)
        sp = ST.paired_sign_permutation(pairs, ST.paired_accuracy_from, n=2000, seed=args.seed)
        y = np.array([r["y"] for r in rs]); pr = np.array([r["pred"] for r in rs])
        sc = np.array([r["score"] for r in rs]); fam = np.array([r["family_id"] for r in rs])
        ab = ST.clustered_bootstrap(ST.auroc, fam, y, sc, n=args.boot, seed=args.seed)
        rows.append({"arm": arm, "n_items": len(rs), "n_pairs": len(pairs), "n_tied_pairs": n_tied,
                     "paired_acc": pb["point"], "paired_acc_lo": pb["lo"], "paired_acc_hi": pb["hi"],
                     "paired_perm_p": sp["p_two_sided"],
                     "auroc": ab["point"], "auroc_lo": ab["lo"], "auroc_hi": ab["hi"],
                     "acc_raw": float((y == pr).mean()),
                     "hit_rate": ST.hit_rate(y, pr), "false_alarm": ST.false_alarm_rate(y, pr)})
        r = rows[-1]
        print(f"  {arm:36s} paired={r['paired_acc']:.3f}[{r['paired_acc_lo']:.3f},{r['paired_acc_hi']:.3f}] "
              f"p={r['paired_perm_p']:.4f} tied={n_tied}/{len(pairs)} AUROC={r['auroc']:.3f} "
              f"FA={r['false_alarm']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
