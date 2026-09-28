"""Surface-cue features, for the shortcut audit that gates Stage 2."""
from __future__ import annotations

import re
import numpy as np

from bridge2026.schema import norm_tokens, repetition_scores

NEG = re.compile(r"\b(not|no|never|none|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|nothing)\b")
POLITE = re.compile(r"\b(please|thanks|thank you|sorry|could you|would you|appreciate|kindly)\b")
REPAIR_MARK = re.compile(r"\b(still|again|already|yet|actually|wait|sorry|repeat|confirm|clarify|mean)\b")
HEDGE = re.compile(r"\b(maybe|perhaps|might|probably|i think|not sure|unsure|possibly)\b")

FEATURE_NAMES = [
    "n_chars", "n_words", "n_turns", "n_user_words", "n_asst_words",
    "n_qmarks", "n_user_qmarks", "rep_any", "rep_user",
    "n_neg", "n_polite", "n_repair_marks", "n_hedge",
    "ttr", "mean_word_len", "n_digits", "last_turn_words",
    "user_asst_word_ratio", "max_sentence_words",
]


class _T:
    __slots__ = ("speaker", "text")

    def __init__(self, speaker, text):
        self.speaker, self.text = speaker, text


def _turns(item):
    return [_T(t["speaker"], t["text"]) if isinstance(t, dict) else t for t in item]


def surface_features(turns) -> dict:
    turns = _turns(turns)
    blob = " ".join(t.text for t in turns)
    utext = " ".join(t.text for t in turns if t.speaker == "user")
    atext = " ".join(t.text for t in turns if t.speaker == "assistant")
    words = norm_tokens(blob)
    uw, aw = len(norm_tokens(utext)), len(norm_tokens(atext))
    rep = repetition_scores(turns)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", blob) if s.strip()]
    low = blob.lower()
    return {
        "n_chars": len(blob),
        "n_words": len(words),
        "n_turns": len(turns),
        "n_user_words": uw,
        "n_asst_words": aw,
        "n_qmarks": blob.count("?"),
        "n_user_qmarks": utext.count("?"),
        "rep_any": rep["rep_any"],
        "rep_user": rep["rep_user"],
        "n_neg": len(NEG.findall(low)),
        "n_polite": len(POLITE.findall(low)),
        "n_repair_marks": len(REPAIR_MARK.findall(low)),
        "n_hedge": len(HEDGE.findall(low)),
        "ttr": len(set(words)) / max(len(words), 1),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "n_digits": sum(c.isdigit() for c in blob),
        "last_turn_words": len(norm_tokens(turns[-1].text)),
        "user_asst_word_ratio": uw / max(aw, 1),
        "max_sentence_words": max((len(norm_tokens(s)) for s in sents), default=0),
    }


def feature_matrix(items) -> np.ndarray:
    return np.array([[surface_features(it["turns"])[k] for k in FEATURE_NAMES] for it in items], dtype=float)


def dialogue_text(item) -> str:
    """The only text a text-only baseline may read."""
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in item["turns"])
