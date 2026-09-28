#!/usr/bin/env python3
"""Aggregate the blind rubric scores by arm (un-blinding only at analysis time)."""
from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import stats as ST

BIN = ["false_trigger", "maintains_task_facts", "re_requests_given_info",
       "cognitive_inference", "overall_appropriate", "truncated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric-dir", default="runs/bridge2026/stage4/rubric")
    ap.add_argument("--responses", default="runs/bridge2026/stage4/e4_freeform_test.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.rubric_dir)
    key = json.loads((d / "rubric_key.json").read_text())
    scores = {}
    for p in sorted(d.glob("scores_*.json")):
        for r in json.loads(p.read_text()):
            scores[r["response_id"]] = r
    print(f"scored {len(scores)} of {len(key)} responses")
    if not scores:
        raise SystemExit("no rubric scores found yet")

    fam = {}
    for r in csv.DictReader(open(args.responses)):
        fam[(r["arm"], r["item_id"])] = r["family_id"]

    by_arm = defaultdict(list)
    for rid, sc in scores.items():
        k = key.get(rid)
        if not k:
            continue
        by_arm[k["arm"]].append({**sc, **k, "family_id": fam.get((k["arm"], k["item_id"]), k["item_id"])})

    rows = []
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        r = {"arm": arm, "n": len(rs)}
        fams = np.array([x["family_id"] for x in rs])
        for f in BIN:
            v = np.array([1.0 if str(x.get(f)) in ("1", "True", "true") else 0.0 for x in rs])
            b = ST.clustered_bootstrap(lambda a: float(np.mean(a)), fams, v, n=args.boot, seed=args.seed)
            r[f] = b["point"]; r[f"{f}_lo"] = b["lo"]; r[f"{f}_hi"] = b["hi"]
        # addresses_open_issue, restricted to items where something IS open
        opn = [x for x in rs if x["label"] == "unresolved" and str(x.get("addresses_open_issue")) in ("0", "1")]
        if opn:
            v = np.array([float(x["addresses_open_issue"]) for x in opn])
            fo = np.array([x["family_id"] for x in opn])
            b = ST.clustered_bootstrap(lambda a: float(np.mean(a)), fo, v, n=args.boot, seed=args.seed)
            r["addresses_open_issue_on_unresolved"] = b["point"]
            r["addresses_open_issue_lo"] = b["lo"]; r["addresses_open_issue_hi"] = b["hi"]
            r["n_unresolved_scored"] = len(opn)
        # false trigger restricted to items where nothing is open
        clo = [x for x in rs if x["label"] == "resolved"]
        if clo:
            v = np.array([1.0 if str(x.get("false_trigger")) in ("1", "True", "true") else 0.0 for x in clo])
            fc = np.array([x["family_id"] for x in clo])
            b = ST.clustered_bootstrap(lambda a: float(np.mean(a)), fc, v, n=args.boot, seed=args.seed)
            r["false_trigger_on_resolved"] = b["point"]
            r["false_trigger_on_resolved_lo"] = b["lo"]; r["false_trigger_on_resolved_hi"] = b["hi"]
            r["n_resolved_scored"] = len(clo)
        ql = [str(x.get("question_load", "")) for x in rs]
        for lvl in ("none", "one", "several"):
            r[f"q_{lvl}"] = float(np.mean([q == lvl for q in ql]))
        rows.append(r)
        print(f"  {arm:14s} n={r['n']:3d} "
              f"addresses_open={r.get('addresses_open_issue_on_unresolved', float('nan')):.3f} "
              f"false_trigger_on_resolved={r.get('false_trigger_on_resolved', float('nan')):.3f} "
              f"facts={r['maintains_task_facts']:.3f} cog_inf={r['cognitive_inference']:.3f} "
              f"appropriate={r['overall_appropriate']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for x in rows for k in x}))
        w.writeheader(); w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
