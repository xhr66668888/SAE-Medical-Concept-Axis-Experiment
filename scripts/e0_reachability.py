#!/usr/bin/env python3
"""E0.11 - causal reachability of a candidate intervention site.

An intervention site is only usable if perturbing it can change the model's output AT ALL. This is
a LABEL-INDEPENDENT check: it uses random directions and measures the magnitude of the logit change
at the answer position. It says nothing about whether the intervention moves the answer in the
right direction, so using it to choose a site cannot bias the causal result.

Motivation: at layer 29 the canonical read position sits 55-67 tokens before the answer token with
only 4 layers left. A perturbation of twice the residual norm there moves the final logits by 0.33
nats, while the same perturbation at the final position moves them by 41. A null causal result at
that site would have been an artefact of unreachability, not evidence about the representation.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import readout as RO
from bridge2026 import runtime as R

MIN_REACH = 0.05  # nats of mean |delta logit| at the answer position, at alpha = 1 sd_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--acts", default="runs/bridge2026/cache/acts_4b.npz")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layers", default="9,17,22,29")
    ap.add_argument("--modes", default="read_pos,read_to_end,prompt_all")
    ap.add_argument("--n-items", type=int, default=16)
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="runs/bridge2026/stage2/e0_reachability.csv")
    args = ap.parse_args()

    items = [r for r in (json.loads(l) for l in open(args.items))
             if r["set"] == "main" and r["split"] == args.split][:args.n_items]
    z = np.load(args.acts, allow_pickle=True)
    A, split, label = z["acts"], z["split"], z["label"]
    tr = np.flatnonzero((split == "train") & (label != "indeterminate"))

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    rows_, texts, pos = RO.build_rows(rt.tokenizer, items, context="full", wordings=("W1",))
    ids, attn, ap_ = R.encode_batch(rt, texts, pos)
    tok_a = rt.tokenizer("A", add_special_tokens=False)["input_ids"][0]
    tok_b = rt.tokenizer("B", add_special_tokens=False)["input_ids"][0]

    def last_logits(iv=None):
        out = []
        for i in range(0, ids.shape[0], 8):
            sl = slice(i, i + 8)
            f = None
            if iv is not None:
                m = R.span_mask(ap_[sl], attn[sl], iv[1])
                f = {iv[0]: R.add_direction_mask(iv[2], iv[3], m)}
            _, lg = R.forward_capture(rt, ids[sl], attn[sl], [], None, want_logits=True, intervene=f)
            out.append(lg[:, -1, :])
        return torch.cat(out, 0)

    clean = last_logits()
    rng = np.random.default_rng(args.seed)
    rows = []
    for L in [int(x) for x in args.layers.split(",")]:
        H = A[L][tr]
        for mode in args.modes.split(","):
            dl, dab, dn = [], [], []
            for _ in range(args.n_dirs):
                v = rng.standard_normal(A.shape[2]); v = v / np.linalg.norm(v)
                sd = float((H @ v).std())
                d = torch.as_tensor(v, dtype=torch.float32)
                lg = last_logits((L, mode, d, sd))
                dl.append(float((lg - clean).abs().max()))
                dab.append(float(((lg[:, tok_a] - lg[:, tok_b]) -
                                  (clean[:, tok_a] - clean[:, tok_b])).abs().mean()))
                dn.append(sd)
            r = {"layer": L, "mode": mode, "sd_train_random_dir": float(np.mean(dn)),
                 "max_abs_dlogit": float(np.mean(dl)),
                 "mean_abs_d_AB_margin": float(np.mean(dab)),
                 "reachable": bool(np.mean(dab) >= MIN_REACH)}
            rows.append(r)
            print(f"  L{L:2d} {mode:12s} sd={r['sd_train_random_dir']:8.1f}  "
                  f"max|dlogit|={r['max_abs_dlogit']:8.4f}  "
                  f"|d(A-B)|={r['mean_abs_d_AB_margin']:8.4f}  "
                  f"{'REACHABLE' if r['reachable'] else 'inert'}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nthreshold: mean |delta (A-B) margin| >= {MIN_REACH} nats at alpha = 1 sd_train")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
