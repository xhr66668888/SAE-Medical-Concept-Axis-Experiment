#!/usr/bin/env python3
"""Behaviour on the 48 deliberately-undecidable items (spec sec.3.3).

These were written to be genuinely arguable and were routed to the ambiguity set BEFORE any model
was run. A well-calibrated system should sit closer to indifference here than on the decidable
main set. The comparison is the distribution of |score|, not accuracy (there is no ground truth to
be accurate against).
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import readout as RO
from bridge2026 import runtime as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_all = [json.loads(l) for l in open(args.items)]
    amb = [r for r in rows_all if r["set"] == "ambiguity"]
    main_test = [r for r in rows_all if r["set"] == "main" and r["split"] == "test"]
    print(f"ambiguity items: {len(amb)}   main test items: {len(main_test)}")

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    out = []
    for tag, items in (("ambiguity", amb), ("main_test", main_test)):
        rr, texts, pos = RO.build_rows(rt.tokenizer, items, context="full")
        sc, _ = RO.score_rows(rt, rr, texts, pos, batch_size=args.batch_size)
        agg = RO.aggregate(rr, sc)
        s = np.array([a["score"] for a in agg])
        out.append({"set": tag, "n": len(agg),
                    "mean_score": float(s.mean()), "median_score": float(np.median(s)),
                    "mean_abs_score": float(np.abs(s).mean()),
                    "median_abs_score": float(np.median(np.abs(s))),
                    "frac_pred_unresolved": float((s > 0).mean()),
                    "frac_abs_score_lt_1": float((np.abs(s) < 1).mean()),
                    "frac_abs_score_lt_0p5": float((np.abs(s) < 0.5).mean()),
                    "mean_rendering_sd": float(np.mean([a["score_sd_over_renderings"] for a in agg]))})
        r = out[-1]
        print(f"  {tag:10s} n={r['n']:3d} mean={r['mean_score']:+7.3f} "
              f"mean|score|={r['mean_abs_score']:6.3f} median|score|={r['median_abs_score']:6.3f} "
              f"pred_unres={r['frac_pred_unresolved']:.2f} |s|<1: {r['frac_abs_score_lt_1']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
