#!/usr/bin/env python3
"""Capture residual activations at the canonical read position, all layers, PURE rendering.

PURE has no answer options and no generation prompt, so nothing about the A/B task can enter the
extracted direction. E0.10 verifies that the residual at this position is identical inside the
READOUT rendering, so directions fitted here transfer exactly to the behavioural experiments.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import runtime as R
from bridge2026.prompts import read_position, render_pure
from bridge2026.schema import Turn


def load_items(path, sets=("main",)):
    return [r for r in (json.loads(l) for l in open(path)) if r["set"] in sets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sets", default="main,ambiguity")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_items(args.items, tuple(args.sets.split(",")))
    print(f"items: {len(items)}")
    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    layers = list(range(rt.n_layers))

    texts, pos = [], []
    for it in items:
        turns = [Turn(t["speaker"], t["text"]) for t in it["turns"]]
        texts.append(render_pure(rt.tokenizer, turns))
        pos.append(read_position(rt.tokenizer, turns))

    acts = {L: [] for L in layers}
    for i in range(0, len(texts), args.batch_size):
        sl = slice(i, i + args.batch_size)
        ids, attn, ap_ = R.encode_batch(rt, texts[sl], pos[sl])
        store, _ = R.forward_capture(rt, ids, attn, layers, ap_)
        for L in layers:
            acts[L].append(store[L])
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(texts[sl])}/{len(texts)}", flush=True)

    arr = np.stack([torch.cat(acts[L], 0).numpy() for L in layers], axis=0)  # [n_layers, n_items, d]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        acts=arr.astype(np.float32),
        item_id=np.array([it["item_id"] for it in items]),
        family_id=np.array([it["family_id"] for it in items]),
        domain=np.array([it["domain"] for it in items]),
        condition=np.array([it["condition"] for it in items]),
        repetition=np.array([it["repetition"] for it in items]),
        label=np.array([it["label"] for it in items]),
        subtype=np.array([it["subtype"] for it in items]),
        split=np.array([it["split"] for it in items]),
        n_tokens=np.array(pos),
        model=np.array([args.model]),
    )
    print(f"saved {arr.shape} -> {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
