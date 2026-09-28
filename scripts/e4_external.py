#!/usr/bin/env python3
"""External test on CCPE-M (plan sec.4).

CCPE-M is 502 crowd-sourced Wizard-of-Oz movie-preference dialogues (CC BY 4.0). It carries no
cognitive-state annotation. It is NOT a healthy control group and NOT clinical validation.

Blind re-annotation under the same guidelines returned 97 `resolved`, 3 `unresolved`, 0
`indeterminate`. With that distribution, discrimination metrics are meaningless and are NOT
reported. The set is used for exactly one thing, which is what the plan asked of it: measuring how
often a frozen model false-triggers on ordinary human dialogue, split by whether the segment
actually contains surface repetition.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import interventions as IV
from bridge2026 import readout as RO
from bridge2026 import runtime as R
from bridge2026 import stats as ST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="data/external/ccpem_segments.json")
    ap.add_argument("--labels-dir", default="runs/bridge2026/stage3/ccpem")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--config", default="")
    ap.add_argument("--directions", default="runs/bridge2026/stage2/e2_directions.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    segs = {s["item_id"]: s for s in json.loads(Path(args.segments).read_text())["items"]}
    idx = json.loads((Path(args.labels_dir) / "index.json").read_text())
    lab = {}
    for p in sorted(Path(args.labels_dir).glob("labels_*.json")):
        for r in json.loads(p.read_text()):
            lab[r["id"]] = r
    items, meta = [], {}
    for aid, info in idx.items():
        if aid not in lab:
            continue
        s = segs[info["item_id"]]
        L = lab[aid]["label"]
        items.append({"item_id": info["item_id"], "family_id": info["item_id"],
                      "domain": "ccpem", "condition": "external",
                      "repetition": "rep" if info["rep_any"] >= 5 else "norep",
                      "label": L, "subtype": lab[aid].get("subtype", ""), "split": "external",
                      "turns": s["turns"]})
        meta[info["item_id"]] = {"rep_any": info["rep_any"], "confidence": lab[aid].get("confidence")}
    from collections import Counter
    print(f"CCPE-M: {len(items)} annotated segments, labels {dict(Counter(i['label'] for i in items))}")
    dec = [i for i in items if i["label"] != "indeterminate"]
    print(f"decidable: {len(dec)}  rep>=5: {sum(i['repetition']=='rep' for i in dec)}")

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    arms = [("original", "", None), ("prompt_only", RO.CONTEXT_INSTRUCTION, None)]
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        d = np.load(args.directions)[f"L{cfg['layer']}_unresolved"]
        a = IV.dense_axis(cfg["layer"], d, cfg["freeform_alpha"], cfg["sd_train"],
                          span=cfg.get("position_mode", "read_to_end"))
        a.name = f"dense_axis(a={cfg['freeform_alpha']:+g})"
        arms.append((a.name, "", a))

    rows = []
    for name, prefix, arm in arms:
        rr, texts, pos = RO.build_rows(rt.tokenizer, dec, context="full", prefix=prefix)
        sc, _ = RO.score_rows(rt, rr, texts, pos, intervene=arm, batch_size=args.batch_size)
        agg = RO.aggregate(rr, sc)
        y = np.array([a_["y"] for a_ in agg]); pr = np.array([a_["pred"] for a_ in agg])
        fam = np.array([a_["family_id"] for a_ in agg])
        rep = np.array([a_["repetition"] for a_ in agg])
        r = {"arm": name, "n": len(agg), "n_resolved": int((y == 0).sum()),
             "n_unresolved": int((y == 1).sum())}
        b = ST.clustered_bootstrap(lambda yy, pp: ST.false_alarm_rate(yy, pp), fam, y, pr,
                                   n=args.boot, seed=args.seed)
        r["false_alarm"], r["false_alarm_lo"], r["false_alarm_hi"] = b["point"], b["lo"], b["hi"]
        for lvl in ("rep", "norep"):
            m = (rep == lvl) & (y == 0)
            r[f"false_alarm_{lvl}"] = float(pr[m].mean()) if m.any() else float("nan")
            r[f"n_{lvl}_resolved"] = int(m.sum())
        r["hit_rate_on_3_unresolved"] = ST.hit_rate(y, pr)
        r["mean_score"] = float(np.mean([a_["score"] for a_ in agg]))
        rows.append(r)
        print(f"  {name:22s} FA={r['false_alarm']:.3f} [{r['false_alarm_lo']:.3f},{r['false_alarm_hi']:.3f}]"
              f"  FA(rep)={r['false_alarm_rep']:.3f}  FA(norep)={r['false_alarm_norep']:.3f}"
              f"  mean_score={r['mean_score']:+.3f}", flush=True)
        for a_ in agg:
            a_["arm"] = name
            a_.update(meta.get(a_["item_id"], {}))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
