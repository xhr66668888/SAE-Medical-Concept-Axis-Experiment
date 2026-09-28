#!/usr/bin/env python3
"""Fit the competing 'generic clarification need' direction and compare it to the target axis.

The plan requires a competing explanation to be measured, not asserted: if steering the target axis
also moves a generic 'this request is underspecified, ask a question' axis, then the intervention may
just be turning up a global help-more tendency. This direction is fitted on single-turn prompts from
domains the main dataset never touches, so it cannot be a relabelling of the construct.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import runtime as R
from bridge2026.prompts import read_position, render_pure
from bridge2026.schema import Turn


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/bridge2026/clarification_probe.json")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--directions", default="runs/bridge2026/stage2/e2_directions.npz")
    ap.add_argument("--out-dirs", default="runs/bridge2026/stage2/competing_directions.npz")
    ap.add_argument("--out-csv", default="runs/bridge2026/stage2/competing_cosines.csv")
    args = ap.parse_args()

    items = json.loads(Path(args.probe).read_text())["items"]
    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    layers = list(range(rt.n_layers))
    texts, pos = [], []
    for it in items:
        turns = [Turn(t["speaker"], t["text"]) for t in it["turns"]]
        texts.append(render_pure(rt.tokenizer, turns))
        pos.append(read_position(rt.tokenizer, turns))
    acts = {L: [] for L in layers}
    for i in range(0, len(texts), 8):
        sl = slice(i, i + 8)
        ids, attn, ap_ = R.encode_batch(rt, texts[sl], pos[sl])
        st, _ = R.forward_capture(rt, ids, attn, layers, ap_)
        for L in layers:
            acts[L].append(st[L])
    A = np.stack([torch.cat(acts[L], 0).numpy() for L in layers], 0)
    amb = np.array([it["kind"] == "ambiguous" for it in items])
    print(f"probe: {amb.sum()} ambiguous / {(~amb).sum()} clear, layers={A.shape[0]}")

    existing = dict(np.load(args.directions)) if Path(args.directions).exists() else {}
    out, rows = {}, []
    for L in layers:
        d_cl = unit(A[L][amb].mean(0) - A[L][~amb].mean(0))
        out[f"L{L}_clarification"] = d_cl
        # separability of the probe itself, as a sanity check that the axis means something
        s = A[L] @ d_cl
        from bridge2026.stats import auroc
        row = {"layer": L, "clarify_probe_auroc": auroc(amb.astype(int), s),
               "sd_proj_probe": float(s.std())}
        for name in ("unresolved", "repetition"):
            k = f"L{L}_{name}"
            if k in existing:
                row[f"cos_clarify_{name}"] = float(d_cl @ existing[k])
        rows.append(row)
    np.savez_compressed(args.out_dirs, **out)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    for r in rows[::4]:
        extra = "".join(f" cos_{k.split('_')[-1]}={r[k]:+.3f}" for k in r if k.startswith("cos_"))
        print(f"  L{r['layer']:2d} probe AUROC={r['clarify_probe_auroc']:.3f}{extra}")
    print(f"-> {args.out_dirs}\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
