"""Rendering, readout protocol and context ablations.

Three renderings, all sharing one token prefix:

  PURE     dialogue only, chat template, no generation prompt.
  GEN      dialogue + generation prompt  -> free-form response endpoint (spec sec.7 of the plan).
  READOUT  dialogue with the A/B question appended INSIDE the final user turn + generation prompt.

Gemma's chat template requires strictly alternating roles, so the question cannot be a new user
turn. Appending it inside the final user turn is also what makes the three renderings share the
token prefix that ends at the last token of the original final user text. That token is the
canonical READ POSITION: representations, directions, patches and steering all act there, and it is
the same absolute index in PURE, GEN and READOUT. The answer options therefore cannot leak into the
extracted direction (spec sec.8).

Fixed sign convention everywhere:
    score = logP(letter meaning UNRESOLVED) - logP(letter meaning RESOLVED)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

ROLE = {"user": "user", "assistant": "model"}
SEP = "\n\n---\n"

WORDINGS = {
    "W1": {
        "stem": "Before you reply: which of these is the better description of where things stand right now?",
        "unresolved": "Something I raised has still not been dealt with.",
        "resolved": "Everything I raised has been dealt with, and we are on track.",
        "instruction": "Answer with a single letter, A or B.",
    },
    "W2": {
        "stem": "Before you reply: which statement is true at this point in our conversation?",
        "unresolved": "A point I brought up is still open.",
        "resolved": "No point I brought up is still open.",
        "instruction": "Reply with only A or B.",
    },
}


@dataclass
class Rendering:
    text: str
    read_pos: int               # left-counted index of the canonical read position
    unresolved_letter: str
    resolved_letter: str
    wording: str
    order: int                  # 0 = unresolved is option A, 1 = resolved is option A


def _msgs(turns) -> list[dict]:
    return [{"role": ROLE[t.speaker], "content": t.text} for t in turns]


def render_pure(tok, turns) -> str:
    return tok.apply_chat_template(_msgs(turns), tokenize=False, add_generation_prompt=False)


def render_gen(tok, turns) -> str:
    return tok.apply_chat_template(_msgs(turns), tokenize=False, add_generation_prompt=True)


def read_position(tok, turns) -> int:
    """Left-counted index of the last token of the final user turn's own text.

    Computed as the length of the common token prefix between PURE and READOUT, minus one, which is
    exactly the last token before the two renderings diverge.
    """
    a = tok(render_pure(tok, turns), add_special_tokens=False)["input_ids"]
    b = tok(_readout_text(tok, turns, "W1", 0), add_special_tokens=False)["input_ids"]
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n < 3:
        raise ValueError("common prefix unexpectedly short; check the chat template")
    return n - 1


def _question(wording: str, order: int) -> tuple[str, str, str]:
    w = WORDINGS[wording]
    if order == 0:
        a, b, unres, res = w["unresolved"], w["resolved"], "A", "B"
    else:
        a, b, unres, res = w["resolved"], w["unresolved"], "B", "A"
    q = f"{SEP}{w['stem']}\nA) {a}\nB) {b}\n{w['instruction']}"
    return q, unres, res


def _readout_text(tok, turns, wording: str, order: int) -> str:
    q, _, _ = _question(wording, order)
    last = turns[-1]
    turns2 = list(turns[:-1]) + [replace(last, text=last.text + q)]
    return tok.apply_chat_template(_msgs(turns2), tokenize=False, add_generation_prompt=True)


def render_readout(tok, turns, wording: str = "W1", order: int = 0) -> Rendering:
    _, unres, res = _question(wording, order)
    return Rendering(
        text=_readout_text(tok, turns, wording, order),
        read_pos=read_position(tok, turns),
        unresolved_letter=unres,
        resolved_letter=res,
        wording=wording,
        order=order,
    )


def all_renderings(tok, turns, wordings=("W1", "W2")) -> list[Rendering]:
    """2 wordings x 2 option orders. Averaging over orders cancels option-position bias."""
    return [render_readout(tok, turns, w, o) for w in wordings for o in (0, 1)]


# --------------------------------------------------------------------------------------
# context ablations for E1 (shortcut diagnosis)
# --------------------------------------------------------------------------------------

def ctx_full(item, partner=None):
    return list(item.turns)


def ctx_last_only(item, partner=None):
    """Only the final user turn. Accuracy here cannot come from dialogue history."""
    return [item.turns[-1]]


def ctx_shuffled(item, partner=None, seed: int = 0):
    """History permuted within speaker role; final user turn fixed. Keeps the bag of words,
    destroys the order that makes a problem 'still open'."""
    hist = list(item.turns[:-1])
    rng = random.Random(f"{item.item_id}|{seed}")
    for role in ("user", "assistant"):
        idx = [i for i, t in enumerate(hist) if t.speaker == role]
        texts = [hist[i].text for i in idx]
        rng.shuffle(texts)
        for i, txt in zip(idx, texts):
            hist[i] = replace(hist[i], text=txt)
    return hist + [item.turns[-1]]


def ctx_swapped(item, partner):
    """History from the opposite-label partner at the same repetition level, own final user turn.
    Tail anchors are identical within a family, so this isolates the history's contribution."""
    if partner is None:
        return None
    return list(partner.turns[:-1]) + [item.turns[-1]]


CONTEXT_VARIANTS = {
    "full": ctx_full,
    "last_only": ctx_last_only,
    "shuffled": ctx_shuffled,
    "swapped": ctx_swapped,
}
