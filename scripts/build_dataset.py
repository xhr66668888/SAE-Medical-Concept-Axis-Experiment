#!/usr/bin/env python3
"""Assemble items, assign the family-level split, and FREEZE it with a content hash.

Refuses to silently change an existing freeze: re-running after the data changed requires --force,
and the previous freeze is archived. This is the mechanism that makes 'test is read once' auditable.
"""
from __future__ import annotations

import argparse, hashlib, json, random, shutil, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026.schema import CONDITIONS, DOMAINS, load_family, load_indeterminate

TRANSFER_DOMAIN = "household"


def sha256_of(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def allocate(n: int, fracs=(0.6, 0.2, 0.2)) -> tuple[int, int, int]:
    """Largest-remainder allocation so the totals always sum to n."""
    raw = [n * f for f in fracs]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(range(3), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(rem):
        base[order[i % 3]] += 1
    return tuple(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-dir", default="data/bridge2026/families")
    ap.add_argument("--indeterminate", default="data/bridge2026/indeterminate.json")
    ap.add_argument("--out-items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--out-splits", default="data/bridge2026/splits.json")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(args.families_dir).glob("*.json"))
    fams = [load_family(p) for p in paths]
    if not fams:
        raise SystemExit("no families found")

    by_domain = defaultdict(list)
    for f in fams:
        by_domain[f.domain].append(f.family_id)

    rng = random.Random(args.seed)
    split_of: dict[str, str] = {}
    per_domain = {}
    for dom in sorted(by_domain):
        ids = sorted(by_domain[dom])
        rng.shuffle(ids)
        ntr, nde, nte = allocate(len(ids))
        for i, fid in enumerate(ids):
            split_of[fid] = "train" if i < ntr else ("dev" if i < ntr + nde else "test")
        per_domain[dom] = {"n_families": len(ids), "train": ntr, "dev": nde, "test": nte}

    rows = []
    for f in fams:
        for cond in CONDITIONS:
            it = f.conditions[cond]
            rows.append({
                "item_id": it.item_id, "family_id": f.family_id, "domain": f.domain,
                "condition": cond, "repetition": it.repetition, "label": it.label,
                "subtype": it.subtype, "confidence": it.confidence, "source": it.source,
                "split": split_of[f.family_id],
                "is_transfer_domain": f.domain == TRANSFER_DOMAIN,
                "n_turns": len(it.turns), "tail_anchor": f.tail_anchor,
                "task_goal": f.task_goal,
                "turns": [{"speaker": t.speaker, "text": t.text} for t in it.turns],
                "evidence_spans": it.evidence_spans,
                "set": "main",
            })

    ip = Path(args.indeterminate)
    n_indet = 0
    if ip.exists():
        for it in load_indeterminate(ip):
            rows.append({
                "item_id": it.item_id, "family_id": it.family_id, "domain": it.domain,
                "condition": "indeterminate", "repetition": "na", "label": "indeterminate",
                "subtype": it.subtype, "confidence": it.confidence, "source": it.source,
                "split": "ambiguity", "is_transfer_domain": it.domain == TRANSFER_DOMAIN,
                "n_turns": len(it.turns), "tail_anchor": "", "task_goal": "",
                "turns": [{"speaker": t.speaker, "text": t.text} for t in it.turns],
                "evidence_spans": it.evidence_spans, "set": "ambiguity",
            })
            n_indet += 1

    # The split is determined ONLY by the family files; indeterminate items never enter
    # train/dev/test, so adding them later must not invalidate the freeze.
    content_hash = sha256_of(paths)
    indeterminate_hash = sha256_of([ip]) if ip.exists() else None
    out_splits = Path(args.out_splits)
    if out_splits.exists() and not args.force:
        prev = json.loads(out_splits.read_text())
        if prev.get("content_hash") != content_hash:
            raise SystemExit(
                "REFUSING to overwrite a frozen split: the source data changed.\n"
                f"  frozen hash : {prev.get('content_hash')}\n"
                f"  current hash: {content_hash}\n"
                "Re-run with --force only if the freeze is being deliberately re-issued, and record why "
                "in docs/CHANGELOG_STAGE_GATES.md."
            )
        print("split already frozen and data unchanged; rewriting items only")
    elif out_splits.exists() and args.force:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy(out_splits, out_splits.with_suffix(f".{ts}.bak.json"))

    Path(args.out_items).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_items, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "content_hash": content_hash,
        "indeterminate_hash": indeterminate_hash,
        "n_families": len(fams),
        "n_items_main": len(fams) * len(CONDITIONS),
        "n_items_ambiguity": n_indet,
        "transfer_domain": TRANSFER_DOMAIN,
        "per_domain": per_domain,
        "item_counts": dict(counts),
        "family_split": split_of,
    }
    out_splits.write_text(json.dumps(payload, indent=2))

    print(f"families={len(fams)} main_items={len(fams)*4} ambiguity_items={n_indet}")
    print("per-domain family split:")
    for d, v in per_domain.items():
        print(f"  {d:12s} n={v['n_families']:3d}  train={v['train']:3d} dev={v['dev']:3d} test={v['test']:3d}")
    print("item counts:", dict(counts))
    print("content_hash:", content_hash[:16], "->", out_splits)


if __name__ == "__main__":
    main()
