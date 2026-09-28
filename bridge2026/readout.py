"""A/B readout, context ablations, prompt variants, free-form generation.

Everything funnels through one scoring function so that E1 (behaviour), E3 (causal) and E4
(external / replication) are numerically comparable, and every intervention acts at the canonical
hook site on the canonical read position.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from bridge2026 import runtime as R
from bridge2026.prompts import (
    CONTEXT_VARIANTS, WORDINGS, all_renderings, read_position, render_gen, render_readout,
)
from bridge2026.schema import Turn

CONTEXT_INSTRUCTION = (
    "When you answer, first check the earlier turns of this conversation and decide whether anything "
    "I raised is still open, rather than judging only from my last message.\n\n"
)


def _apply_context(item, variant, partner=None, seed=0):
    fn = CONTEXT_VARIANTS[variant]
    turns = fn(item, partner, seed) if variant == "shuffled" else fn(item, partner)
    return turns


def _with_prefix(turns, prefix: str):
    if not prefix:
        return turns
    first = turns[0]
    return [replace(first, text=prefix + first.text)] + list(turns[1:])


def build_rows(tok, items, *, context="full", partners=None, prefix="", wordings=("W1", "W2"), seed=0):
    """One row per (item, wording, option-order). Returns (rows, texts, positions)."""
    rows, texts, positions = [], [], []
    for it in items:
        partner = (partners or {}).get(it["item_id"])
        turns = _apply_context(_ItemView(it), context, _ItemView(partner) if partner else None, seed)
        if turns is None:
            continue
        turns = _with_prefix(turns, prefix)
        pos = read_position(tok, turns)
        for w in wordings:
            for o in (0, 1):
                r = render_readout(tok, turns, w, o)
                rows.append({
                    "item_id": it["item_id"], "family_id": it["family_id"], "domain": it["domain"],
                    "condition": it["condition"], "repetition": it.get("repetition", "na"),
                    "label": it["label"], "subtype": it.get("subtype", ""), "split": it.get("split", ""),
                    "context": context, "prefix": "none" if not prefix else "context_instruction",
                    "wording": w, "order": o,
                    "unresolved_letter": r.unresolved_letter, "resolved_letter": r.resolved_letter,
                })
                texts.append(r.text)
                positions.append(pos)
    return rows, texts, positions


class _ItemView:
    """Adapts a dict row to the attribute access the context functions expect."""
    __slots__ = ("item_id", "turns")

    def __init__(self, d):
        self.item_id = d["item_id"]
        self.turns = [Turn(t["speaker"], t["text"]) for t in d["turns"]]


@torch.inference_mode()
def score_rows(rt, rows, texts, positions, *, intervene=None, batch_size=8, capture_layers=()):
    """Score = logP(letter meaning UNRESOLVED) - logP(letter meaning RESOLVED) at the last position.

    `intervene`: callable(slice, abs_positions, attention_mask) -> {layer: fn(hidden)->hidden}, so
    the caller can build a batch-specific intervention over an arbitrary token span.
    `capture_layers`: also return residuals at those layers, gathered at the read position.
    """
    tok = rt.tokenizer
    a_id = tok("A", add_special_tokens=False)["input_ids"][0]
    b_id = tok("B", add_special_tokens=False)["input_ids"][0]
    out_scores, captures = [], {L: [] for L in capture_layers}
    for i in range(0, len(texts), batch_size):
        sl = slice(i, i + batch_size)
        ids, attn, pos = R.encode_batch(rt, texts[sl], positions[sl])
        iv = intervene(sl, pos, attn) if intervene is not None else None
        store, logits = R.forward_capture(rt, ids, attn, list(capture_layers), pos,
                                          want_logits=True, intervene=iv)
        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        for k, row in enumerate(rows[sl]):
            ua = lp[k, a_id if row["unresolved_letter"] == "A" else b_id].item()
            ra = lp[k, b_id if row["unresolved_letter"] == "A" else a_id].item()
            out_scores.append({"logp_unresolved": ua, "logp_resolved": ra, "score": ua - ra,
                               "p_mass_ab": float(np.exp(ua) + np.exp(ra))})
        for L in capture_layers:
            captures[L].append(store[L])
    caps = {L: torch.cat(v, 0) for L, v in captures.items()} if capture_layers else {}
    return out_scores, caps


def aggregate(rows, scores):
    """Average the 4 renderings (2 wordings x 2 orders) per item -> one paired score per item."""
    import collections
    acc = collections.defaultdict(list)
    meta = {}
    for r, s in zip(rows, scores):
        acc[r["item_id"]].append(s["score"])
        meta[r["item_id"]] = r
    out = []
    for iid, vals in acc.items():
        m = dict(meta[iid])
        m.pop("wording", None); m.pop("order", None)
        m.pop("unresolved_letter", None); m.pop("resolved_letter", None)
        m["score"] = float(np.mean(vals))
        m["score_sd_over_renderings"] = float(np.std(vals))
        m["pred"] = int(m["score"] > 0)
        m["y"] = 1 if m["label"] == "unresolved" else 0
        out.append(m)
    return out


@torch.inference_mode()
def generate_responses(rt, items, *, prefix="", intervene=None, max_new_tokens=160, batch_size=4,
                       positions_from_read=True):
    """Free-form next assistant turn (Stage-4 secondary endpoint). Greedy decoding."""
    tok = rt.tokenizer
    texts, positions, meta = [], [], []
    for it in items:
        turns = _with_prefix(_ItemView(it).turns, prefix)
        texts.append(render_gen(tok, turns))
        positions.append(read_position(tok, turns))
        meta.append(it)
    outs = []
    for i in range(0, len(texts), batch_size):
        sl = slice(i, i + batch_size)
        ids, attn, pos = R.encode_batch(rt, texts[sl], positions[sl])
        iv = intervene(sl, pos, attn) if intervene is not None else None
        ctx = R.layer_hooks(rt, iv) if iv else _null_ctx()
        with ctx:
            gen = rt.model.generate(
                input_ids=ids.to(rt.device), attention_mask=attn.to(rt.device),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True,
            )
        for k in range(gen.shape[0]):
            new = gen[k, ids.shape[1]:]
            outs.append({**{kk: meta[i + k][kk] for kk in
                            ("item_id", "family_id", "domain", "condition", "label", "split")},
                         "response": tok.decode(new, skip_special_tokens=True).strip()})
    return outs


import contextlib


@contextlib.contextmanager
def _null_ctx():
    yield
