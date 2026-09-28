#!/usr/bin/env python3
"""Score the blind re-annotation pass against the intended labels (Stage-1 gate item 4)."""
from __future__ import annotations

import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026.stats import cohen_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", default="runs/bridge2026/stage1/blind")
    ap.add_argument("--out", default="runs/bridge2026/stage1/blind_agreement.json")
    args = ap.parse_args()

    d = Path(args.blind_dir)
    key = json.loads((d / "answer_key.json").read_text())
    got = {}
    for p in sorted(d.glob("labels_*.json")):
        for r in json.loads(p.read_text()):
            got[r["id"]] = r
    missing = sorted(set(key) - set(got))
    print(f"intended={len(key)}  blind={len(got)}  missing={len(missing)}")

    ids = [i for i in key if i in got]
    intended = np.array([key[i]["label"] for i in ids])
    blind = np.array([got[i]["label"] for i in ids])

    # binary agreement on the decidable set (blind 'indeterminate' handled separately)
    dec = blind != "indeterminate"
    k_all = cohen_kappa(intended, blind)
    k_dec = cohen_kappa(intended[dec], blind[dec]) if dec.sum() else float("nan")
    agree_dec = float((intended[dec] == blind[dec]).mean()) if dec.sum() else float("nan")

    # subtype agreement, only where both say the same label and it is not indeterminate
    sub_ok = [i for i in ids if key[i]["label"] == got[i]["label"] != "indeterminate"]
    sub_agree = float(np.mean([key[i]["subtype"] == got[i].get("subtype") for i in sub_ok])) if sub_ok else float("nan")

    disputes = [{"id": i, "item_id": key[i]["item_id"], "condition": key[i]["condition"],
                 "domain": key[i]["domain"], "intended": key[i]["label"],
                 "intended_subtype": key[i]["subtype"], "blind": got[i]["label"],
                 "blind_subtype": got[i].get("subtype"), "blind_conf": got[i].get("confidence"),
                 "blind_evidence": got[i].get("evidence", "")[:220]}
                for i in ids if key[i]["label"] != got[i]["label"]]

    by_cond = defaultdict(lambda: [0, 0])
    for i in ids:
        c = key[i]["condition"]
        by_cond[c][1] += 1
        by_cond[c][0] += int(key[i]["label"] == got[i]["label"])
    by_dom = defaultdict(lambda: [0, 0])
    for i in ids:
        dm = key[i]["domain"]
        by_dom[dm][1] += 1
        by_dom[dm][0] += int(key[i]["label"] == got[i]["label"])

    out = {
        "n_scored": len(ids), "n_missing": len(missing),
        "blind_label_counts": dict(Counter(blind.tolist())),
        "blind_confidence_counts": dict(Counter(got[i].get("confidence", "?") for i in ids)),
        "raw_agreement_all": float((intended == blind).mean()),
        "cohen_kappa_all": k_all,
        "n_blind_indeterminate": int((~dec).sum()),
        "raw_agreement_decidable": agree_dec,
        "cohen_kappa_decidable": k_dec,
        "subtype_agreement_where_label_agrees": sub_agree,
        "agreement_by_condition": {c: {"agree": v[0], "n": v[1], "rate": v[0] / v[1]} for c, v in sorted(by_cond.items())},
        "agreement_by_domain": {c: {"agree": v[0], "n": v[1], "rate": v[0] / v[1]} for c, v in sorted(by_dom.items())},
        "disputes": disputes,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"blind label counts    : {out['blind_label_counts']}")
    print(f"blind confidence      : {out['blind_confidence_counts']}")
    print(f"raw agreement (all)   : {out['raw_agreement_all']:.3f}   kappa={k_all:.3f}")
    print(f"raw agreement (decid.): {agree_dec:.3f}   kappa={k_dec:.3f}   "
          f"(blind indeterminate: {out['n_blind_indeterminate']})")
    print(f"subtype agreement     : {sub_agree:.3f}")
    print("by condition:", {c: f"{v['agree']}/{v['n']}" for c, v in out["agreement_by_condition"].items()})
    print("by domain   :", {c: f"{v['agree']}/{v['n']}" for c, v in out["agreement_by_domain"].items()})
    print(f"\n{len(disputes)} disputed items -> {args.out}")
    for r in disputes:
        print(f"  {r['item_id']:28s} intended={r['intended']:11s} blind={r['blind']:13s} "
              f"conf={r['blind_conf']}")


if __name__ == "__main__":
    main()
