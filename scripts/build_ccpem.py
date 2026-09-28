#!/usr/bin/env python3
"""Extract CCPE-M segments for the external test (plan sec.4).

CCPE-M is 502 crowd-sourced Wizard-of-Oz movie-preference dialogues, CC BY 4.0. It carries NO
cognitive-state annotation and is NOT a healthy control group and NOT clinical validation. It is
used for one purpose: natural human dialogue, re-annotated under the same guidelines, to measure
how often the frozen model false-triggers on ordinary repetition it was never designed around.

Segments are matched to the main set's shape (5 or 7 alternating turns, starting and ending on a
user turn) and sampled to over-represent segments that DO contain surface repetition, because that
is the condition the false-trigger claim is about.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026.schema import Turn, repetition_scores

SPEAK = {"USER": "user", "ASSISTANT": "assistant"}


def merge(utts):
    out = []
    for u in utts:
        sp = SPEAK[u["speaker"]]
        txt = u["text"].strip()
        if not txt:
            continue
        if out and out[-1][0] == sp:
            out[-1][1] = (out[-1][1].rstrip() + " " + txt).strip()
        else:
            out.append([sp, txt])
    return out


def windows(turns, sizes=(5, 7)):
    for n in sizes:
        for i in range(0, len(turns) - n + 1):
            w = turns[i:i + n]
            if w[0][0] == "user" and w[-1][0] == "user":
                if all(w[j][0] != w[j + 1][0] for j in range(n - 1)):
                    yield i, n, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/ccpe_data.json")
    ap.add_argument("--out", default="data/external/ccpem_segments.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--min-chars", type=int, default=260)
    ap.add_argument("--max-chars", type=int, default=1100)
    args = ap.parse_args()

    convs = json.loads(Path(args.src).read_text())
    cands = []
    for c in convs:
        ts = merge(c["utterances"])
        picked = None
        for i, n, w in windows(ts):
            blob = " ".join(t[1] for t in w)
            if not (args.min_chars <= len(blob) <= args.max_chars):
                continue
            rep = repetition_scores([Turn(s, t) for s, t in w])
            cand = {"conversation_id": c["conversationId"], "start": i, "n_turns": n,
                    "rep_any": rep["rep_any"], "rep_user": rep["rep_user"],
                    "n_chars": len(blob), "n_qmarks": blob.count("?"),
                    "turns": [{"speaker": s, "text": t} for s, t in w]}
            # prefer, within a conversation, the window with the most surface repetition
            if picked is None or cand["rep_any"] > picked["rep_any"]:
                picked = cand
        if picked:
            cands.append(picked)

    rng = random.Random(args.seed)
    hi = [c for c in cands if c["rep_any"] >= 5]
    lo = [c for c in cands if c["rep_any"] <= 4]
    rng.shuffle(hi); rng.shuffle(lo)
    half = args.n // 2
    sel = hi[:half] + lo[:args.n - min(half, len(hi))]
    sel = sel[:args.n]
    rng.shuffle(sel)
    for k, s in enumerate(sel):
        s["item_id"] = f"ccpem_{k:03d}"
        s["source"] = "CCPE-M (CC BY 4.0), google-research-datasets/ccpe"
    Path(args.out).write_text(json.dumps({"items": sel}, indent=2))

    import statistics as st
    print(f"candidate windows: {len(cands)} (from {len(convs)} conversations); "
          f"rep_any>=5: {len(hi)}, <=4: {len(lo)}")
    print(f"selected: {len(sel)}  rep>=5: {sum(s['rep_any']>=5 for s in sel)}  "
          f"rep<=4: {sum(s['rep_any']<=4 for s in sel)}")
    print(f"  chars median {st.median(s['n_chars'] for s in sel):.0f}  "
          f"turns {dict((n, sum(s['n_turns']==n for s in sel)) for n in (5,7))}  "
          f"qmarks median {st.median(s['n_qmarks'] for s in sel):.0f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
