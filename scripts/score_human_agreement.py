#!/usr/bin/env python3
"""Score the author's independent annotation against the two LLM label sources.

Two comparisons, reported separately because they answer different questions:
  constructed items -> vs the INTENDED label (does the designed contrast survive a human reader?)
  CCPE-M items      -> vs the BLIND LLM pass (is the external-test labelling trustworthy?)
"""
from __future__ import annotations

import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026.stats import cohen_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="human_annotation/answers_filled.csv",
                    help="csv with columns id,label,subtype,confidence")
    ap.add_argument("--key", default="human_annotation/.key.json")
    ap.add_argument("--out", default="runs/bridge2026/stage4/human_agreement.json")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    human = {}
    for r in csv.DictReader(open(args.human)):
        if r.get("label", "").strip():
            human[r["id"].strip()] = {k: (r.get(k) or "").strip() for k in ("label", "subtype", "confidence")}
    print(f"human annotations supplied: {len(human)} of {len(key)}")
    if not human:
        raise SystemExit("no labels found — fill the `label` column first")

    intended = {r["item_id"]: r for r in (json.loads(l) for l in open("data/bridge2026/items.jsonl"))}
    llm = {}
    for p in sorted(Path("runs/bridge2026/stage3/ccpem").glob("labels_*.json")):
        for r in json.loads(p.read_text()):
            llm[r["id"]] = r
    idx = json.loads(Path("runs/bridge2026/stage3/ccpem/index.json").read_text())
    ccpem_by_item = {v["item_id"]: llm.get(a) for a, v in idx.items()}

    out = {"n_supplied": len(human), "n_total": len(key)}
    for src, name in (("constructed", "vs intended label"), ("ccpem", "vs blind LLM pass")):
        hs, rs, ids = [], [], []
        for hid, meta in key.items():
            if meta["source"] != src or hid not in human:
                continue
            ref = (intended[meta["item_id"]]["label"] if src == "constructed"
                   else (ccpem_by_item.get(meta["item_id"]) or {}).get("label"))
            if not ref:
                continue
            hs.append(human[hid]["label"]); rs.append(ref); ids.append((hid, meta["item_id"]))
        if not hs:
            continue
        h, r = np.array(hs), np.array(rs)
        dec = h != "indeterminate"
        blk = {
            "comparison": name, "n": len(h),
            "human_labels": dict(Counter(hs)),
            "raw_agreement": float((h == r).mean()),
            "cohen_kappa": cohen_kappa(h, r),
            "n_human_indeterminate": int((~dec).sum()),
            "raw_agreement_decidable": float((h[dec] == r[dec]).mean()) if dec.any() else None,
            "cohen_kappa_decidable": cohen_kappa(h[dec], r[dec]) if dec.sum() > 1 else None,
            "disagreements": [{"packet_id": a, "item_id": b, "human": hh, "reference": rr,
                               "human_confidence": human[a]["confidence"]}
                              for (a, b), hh, rr in zip(ids, hs, rs) if hh != rr],
        }
        out[src] = blk
        print(f"\n=== {src}: {name} (n={blk['n']}) ===")
        print(f"  human labels      : {blk['human_labels']}")
        print(f"  raw agreement     : {blk['raw_agreement']:.3f}")
        print(f"  Cohen's kappa     : {blk['cohen_kappa']:.3f}")
        if blk["cohen_kappa_decidable"] is not None:
            print(f"  decidable subset  : agreement {blk['raw_agreement_decidable']:.3f}, "
                  f"kappa {blk['cohen_kappa_decidable']:.3f} "
                  f"({blk['n_human_indeterminate']} human indeterminate)")
        print(f"  disagreements     : {len(blk['disagreements'])}")
        for d in blk["disagreements"][:12]:
            print(f"    {d['item_id']:28s} human={d['human']:13s} ref={d['reference']:11s} conf={d['human_confidence']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
