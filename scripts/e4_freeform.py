#!/usr/bin/env python3
"""Free-form response endpoint (plan sec.7) - generate the next assistant turn under each arm.

Arms: original model, prompt-only instruction, dense-axis steering, SAE feature edit.
Interventions act on the PROMPT pass at the canonical read position (the decode passes have no such
position), which is the generation analogue of the readout experiment.

Outputs raw responses plus rule-based measures. The rubric judgement itself is done blind, by
reviewers who never see the arm name, using the packet written by make_freeform_packet.py.
"""
from __future__ import annotations

import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import interventions as IV
from bridge2026 import readout as RO
from bridge2026 import runtime as R
from bridge2026 import sae as S
from bridge2026.schema import norm_tokens, longest_common_run

COGNITIVE = re.compile(
    r"\b(dementia|alzheimer|cognitive|cognition|memory (?:loss|problem|issue)|forgetful|"
    r"confus(?:ed|ion)|are you (?:ok|alright)|do you remember|senior|elderly|impair)\w*\b", re.I)


def load_items(path, split, limit=None, seed=0):
    rows = [r for r in (json.loads(l) for l in open(path))
            if r["set"] == "main" and r["split"] == split and r["label"] != "indeterminate"]
    if limit and len(rows) > limit:
        # sample whole FAMILIES so the paired structure survives
        fams = sorted({r["family_id"] for r in rows})
        rng = np.random.default_rng(seed)
        rng.shuffle(fams)
        keep, out = set(), []
        for f in fams:
            if len(out) + 4 > limit:
                break
            keep.add(f)
        out = [r for r in rows if r["family_id"] in keep]
        return out
    return rows


def rule_measures(item, response):
    """Cheap, auditable measures. They do not replace the blind rubric; they bound it."""
    r_tok = norm_tokens(response)
    user_tok = [norm_tokens(t["text"]) for t in item["turns"] if t["speaker"] == "user"]
    echo = max((longest_common_run(r_tok, u) for u in user_tok), default=0)
    qs = response.count("?")
    return {
        "resp_chars": len(response),
        "resp_words": len(r_tok),
        "n_questions": qs,
        "multi_question": int(qs >= 2),
        "echo_of_user": echo,
        "cognitive_inference": int(bool(COGNITIVE.search(response))),
        "cognitive_span": (COGNITIVE.search(response).group(0) if COGNITIVE.search(response) else ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--directions", default="runs/bridge2026/stage2/e2_directions.npz")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-am-running-the-final-test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_running_the_final_test:
        raise SystemExit("test split is one-shot: pass --i-am-running-the-final-test deliberately")

    cfg = json.loads(Path(args.config).read_text())
    layer = int(cfg["layer"]); sd = float(cfg["sd_train"])
    items = load_items(args.items, args.split, args.limit, args.seed)
    print(f"free-form: {len(items)} items / {len({i['family_id'] for i in items})} families, layer {layer}")

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    dirs = np.load(args.directions)
    d_un = dirs[f"L{layer}_unresolved"]
    alpha = float(cfg.get("freeform_alpha", cfg["alphas"][-1]))

    arms = [("original", "", None), ("prompt_only", RO.CONTEXT_INSTRUCTION, None),
            ("dense_axis", "", IV.dense_axis(layer, d_un, alpha, sd))]
    sae = None
    if cfg.get("sae"):
        sae = S.load_sae(args.model, layer, width=cfg["sae"].get("width", "16k"),
                         l0=cfg["sae"].get("l0", "medium"), device=rt.device)
        feats = [int(f) for f in cfg["sae"]["features"]]
        scale = float(cfg["sae"].get("freeform_amplify", 0.0))
        if scale > 0:
            arms.append(("sae_amplify", "",
                         IV.sae_feature_edit("sae_amplify", layer, sae, feats, "amplify", scale=scale)))
        arms.append(("sae_ablate", "", IV.sae_feature_edit("sae_ablate", layer, sae, feats, "ablate")))

    rows = []
    for name, prefix, arm in arms:
        outs = RO.generate_responses(rt, items, prefix=prefix, intervene=arm,
                                     max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        by = {o["item_id"]: o for o in outs}
        for it in items:
            o = by[it["item_id"]]
            rows.append({"arm": name, **{k: it[k] for k in
                         ("item_id", "family_id", "domain", "condition", "label", "subtype", "split")},
                         "response": o["response"], **rule_measures(it, o["response"])})
        sub = [r for r in rows if r["arm"] == name]
        print(f"  {name:14s} n={len(sub):3d} mean_q={np.mean([r['n_questions'] for r in sub]):.2f} "
              f"multiQ={np.mean([r['multi_question'] for r in sub]):.2f} "
              f"cog_inf={np.mean([r['cognitive_inference'] for r in sub]):.3f} "
              f"words={np.mean([r['resp_words'] for r in sub]):.0f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
