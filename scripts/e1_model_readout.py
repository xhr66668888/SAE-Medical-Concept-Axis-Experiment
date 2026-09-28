#!/usr/bin/env python3
"""E1 model side - can the model read the interaction state, and is it using the context?

Arms:
  zero_shot / full         full dialogue history
  zero_shot / last_only    only the final user turn        -> anything above chance here is a
                                                              final-turn shortcut, not context use
  zero_shot / shuffled     history permuted within speaker -> destroys order, keeps the words
  zero_shot / swapped      history from the opposite-label partner, own final user turn
                           -> the sharpest test: the tail anchor is identical within a family, so
                              this isolates the contribution of the history alone
  prompt_only / full       + an explicit 'check the earlier turns first' instruction
  few_shot_k4 / full       + 4 labelled train examples in the preamble

Primary metric is the within-family paired contrast (constant option/wording bias cancels).
Raw thresholded accuracy and AUROC are reported alongside as diagnostics.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import readout as RO
from bridge2026 import runtime as R
from bridge2026 import stats as ST


def load_items(path, split):
    return [r for r in (json.loads(l) for l in open(path))
            if r["set"] == "main" and r["split"] == split and r["label"] != "indeterminate"]


def partner_map(items):
    by = {}
    for it in items:
        by.setdefault((it["family_id"], it["repetition"]), {})[it["label"]] = it
    out = {}
    for _k, d in by.items():
        if "unresolved" in d and "resolved" in d:
            out[d["unresolved"]["item_id"]] = d["resolved"]
            out[d["resolved"]["item_id"]] = d["unresolved"]
    return out


def few_shot_prefix(train_items, k, seed):
    rng = np.random.default_rng(seed)
    pos = [i for i in train_items if i["label"] == "unresolved"]
    neg = [i for i in train_items if i["label"] == "resolved"]
    pick = []
    for j in range(k):
        src = pos if j % 2 == 0 else neg
        pick.append(src[int(rng.integers(0, len(src)))])
    parts = ["Here are some worked examples of the judgement I will ask you to make.\n"]
    for ex in pick:
        body = "\n".join(f"{t['speaker']}: {t['text']}" for t in ex["turns"])
        verdict = ("Something the user raised has still not been dealt with."
                   if ex["label"] == "unresolved"
                   else "Everything the user raised has been dealt with.")
        parts.append(f"Example conversation:\n{body}\nVerdict: {verdict}\n")
    parts.append("Now here is the real conversation.\n\n")
    return "\n".join(parts)


def run_arm(rt, items, name, context, prefix, pmap, boot, seed, batch_size):
    rows, texts, pos = RO.build_rows(rt.tokenizer, items, context=context,
                                     partners=pmap if context == "swapped" else None,
                                     prefix=prefix, seed=seed)
    if not rows:
        return None, []
    sc, _ = RO.score_rows(rt, rows, texts, pos, batch_size=batch_size)
    agg = RO.aggregate(rows, sc)
    y = np.array([a["y"] for a in agg]); s = np.array([a["score"] for a in agg])
    pr = np.array([a["pred"] for a in agg]); fam = np.array([a["family_id"] for a in agg])
    pairs = ST.make_pairs(agg)
    pb = ST.paired_bootstrap(pairs, ST.paired_accuracy_from, n=boot, seed=seed)
    sp = ST.paired_sign_permutation(pairs, ST.paired_accuracy_from, n=2000, seed=seed)
    ab = ST.clustered_bootstrap(ST.auroc, fam, y, s, n=boot, seed=seed)
    acc = ST.clustered_bootstrap(ST.accuracy, fam, y, pr, n=boot, seed=seed)
    r = {"arm": name, "context": context, "n_items": len(agg), "n_families": len(set(fam)),
         "paired_acc": pb["point"], "paired_acc_lo": pb["lo"], "paired_acc_hi": pb["hi"],
         "paired_perm_p": sp["p_two_sided"],
         "paired_acc_rep": ST.paired_accuracy_from([p for p in pairs if p["level"] == "rep"]),
         "paired_acc_norep": ST.paired_accuracy_from([p for p in pairs if p["level"] == "norep"]),
         "within_family_auroc": ST.paired_accuracy_from(ST.cross_pairs(agg)),
         "auroc": ab["point"], "auroc_lo": ab["lo"], "auroc_hi": ab["hi"],
         "acc_raw": acc["point"], "acc_raw_lo": acc["lo"], "acc_raw_hi": acc["hi"],
         "hit_rate": ST.hit_rate(y, pr), "false_alarm": ST.false_alarm_rate(y, pr),
         "mean_score": float(s.mean()), "rendering_sd": float(np.mean([a["score_sd_over_renderings"] for a in agg]))}
    for cond in ("rep_unres", "rep_res", "norep_unres", "norep_res"):
        sub = [a for a in agg if a["condition"] == cond]
        if sub:
            r[f"mean_score_{cond}"] = float(np.mean([a["score"] for a in sub]))
            r[f"pred_unres_rate_{cond}"] = float(np.mean([a["pred"] for a in sub]))
    return r, [{**a, "arm": name} for a in agg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-am-running-the-final-test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_running_the_final_test:
        raise SystemExit("test split is one-shot: pass --i-am-running-the-final-test deliberately")

    items = load_items(args.items, args.split)
    train = load_items(args.items, "train")
    pmap = partner_map(items)
    print(f"{args.model} | split={args.split} n={len(items)} families={len({i['family_id'] for i in items})}")
    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))

    arms = [
        ("zero_shot", "full", ""),
        ("zero_shot", "last_only", ""),
        ("zero_shot", "shuffled", ""),
        ("zero_shot", "swapped", ""),
        ("prompt_context_instruction", "full", RO.CONTEXT_INSTRUCTION),
        ("few_shot_k4", "full", few_shot_prefix(train, 4, args.seed)),
    ]
    results, per_item = [], []
    for name, ctx, pref in arms:
        label = f"{name}/{ctx}"
        r, pi = run_arm(rt, items, label, ctx, pref, pmap, args.boot, args.seed, args.batch_size)
        if r is None:
            print(f"  {label}: skipped"); continue
        results.append(r); per_item += pi
        print(f"  {label:36s} paired={r['paired_acc']:.3f}[{r['paired_acc_lo']:.3f},{r['paired_acc_hi']:.3f}] "
              f"p={r['paired_perm_p']:.4f} AUROC={r['auroc']:.3f} raw_acc={r['acc_raw']:.3f} "
              f"FA={r['false_alarm']:.3f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(results)
    pi_path = out.with_name(out.stem + "_per_item.csv")
    pk = sorted({k for r in per_item for k in r})
    with open(pi_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pk, extrasaction="ignore"); w.writeheader(); w.writerows(per_item)
    print(f"-> {out}\n-> {pi_path}")


if __name__ == "__main__":
    main()
