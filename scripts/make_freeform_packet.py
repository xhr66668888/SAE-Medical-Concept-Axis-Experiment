#!/usr/bin/env python3
"""Build a BLIND rubric-review packet from free-form responses.

Arm names are stripped, response order within an item is shuffled, and item ids are replaced. The
mapping is written separately and is not part of the packet.
"""
from __future__ import annotations

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    dial = {}
    for line in open(args.items):
        r = json.loads(line)
        dial[r["item_id"]] = r["turns"]

    by_item = defaultdict(list)
    for r in csv.DictReader(open(args.responses)):
        by_item[r["item_id"]].append(r)

    rng = random.Random(args.seed)
    packet, key = [], {}
    for n, (iid, rs) in enumerate(sorted(by_item.items())):
        rng.shuffle(rs)
        pid = f"F{n:04d}"
        opts = []
        for k, r in enumerate(rs):
            oid = f"{pid}_{chr(ord('a') + k)}"
            opts.append({"response_id": oid, "response": r["response"]})
            key[oid] = {"item_id": iid, "arm": r["arm"], "label": r["label"],
                        "condition": r["condition"], "domain": r["domain"]}
        packet.append({"id": pid, "dialogue": dial[iid], "responses": opts})
    rng.shuffle(packet)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    per = (len(packet) + args.n_shards - 1) // args.n_shards
    for s in range(args.n_shards):
        chunk = packet[s * per:(s + 1) * per]
        if chunk:
            (out / f"rubric_{s}.json").write_text(json.dumps(chunk, indent=2))
    (out / "rubric_key.json").write_text(json.dumps(key, indent=2))
    print(f"{len(packet)} items, {len(key)} responses, {args.n_shards} shards -> {out}")


if __name__ == "__main__":
    main()
