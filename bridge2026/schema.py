"""Data schema for the BRIDGE 2026 conversational-repair dataset."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONDITIONS = ("rep_unres", "rep_res", "norep_unres", "norep_res")
CONDITION_LABEL = {
    "rep_unres": "unresolved",
    "rep_res": "resolved",
    "norep_unres": "unresolved",
    "norep_res": "resolved",
}
CONDITION_REPETITION = {
    "rep_unres": "rep",
    "rep_res": "rep",
    "norep_unres": "norep",
    "norep_res": "norep",
}
POSITIVE_SUBTYPES = ("unanswered_question", "contradiction", "ambiguous_reference")
NEGATIVE_SUBTYPES = ("readback_confirmation", "emphasis_repeat", "clean_progress")
DOMAINS = ("appointment", "shopping", "transit", "cooking", "household")
LABELS = ("unresolved", "resolved", "indeterminate")


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Item:
    """One dialogue = one cell of a family (or one indeterminate item)."""
    item_id: str
    family_id: str
    condition: str
    domain: str
    label: str
    subtype: str
    turns: list[Turn]
    evidence_spans: list[dict]
    confidence: str = "high"
    source: str = "claude-opus-5-authored"

    @property
    def repetition(self) -> str:
        return CONDITION_REPETITION.get(self.condition, "na")

    def text_blob(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def user_turns(self) -> list[int]:
        return [i for i, t in enumerate(self.turns) if t.speaker == "user"]


@dataclass
class Family:
    family_id: str
    domain: str
    task_goal: str
    n_turns: int
    tail_anchor: str
    conditions: dict[str, Item] = field(default_factory=dict)
    notes: str = ""


def _turns_from(raw: list[dict]) -> list[Turn]:
    return [Turn(speaker=t["speaker"].strip().lower(), text=t["text"].strip()) for t in raw]


def load_family(path: str | Path) -> Family:
    raw = json.loads(Path(path).read_text())
    fam = Family(
        family_id=raw["family_id"],
        domain=raw["domain"],
        task_goal=raw["task_goal"],
        n_turns=int(raw["n_turns"]),
        tail_anchor=raw["tail_anchor"].strip(),
        notes=raw.get("notes", ""),
    )
    for cond, body in raw["conditions"].items():
        fam.conditions[cond] = Item(
            item_id=f"{fam.family_id}__{cond}",
            family_id=fam.family_id,
            condition=cond,
            domain=fam.domain,
            label=body.get("label", CONDITION_LABEL.get(cond, "")),
            subtype=body["subtype"],
            turns=_turns_from(body["turns"]),
            evidence_spans=body.get("evidence_spans", []),
            confidence=body.get("confidence", "high"),
            source=body.get("source", "claude-opus-5-authored"),
        )
    return fam


def load_indeterminate(path: str | Path) -> list[Item]:
    raw = json.loads(Path(path).read_text())
    out = []
    for body in raw["items"]:
        out.append(
            Item(
                item_id=body["item_id"],
                family_id=body["item_id"],
                condition="indeterminate",
                domain=body["domain"],
                label="indeterminate",
                subtype=body.get("subtype", "indeterminate"),
                turns=_turns_from(body["turns"]),
                evidence_spans=body.get("evidence_spans", []),
                confidence=body.get("confidence", "low"),
                source=body.get("source", "claude-opus-5-authored"),
            )
        )
    return out


def family_to_dict(fam: Family) -> dict:
    return {
        "family_id": fam.family_id,
        "domain": fam.domain,
        "task_goal": fam.task_goal,
        "n_turns": fam.n_turns,
        "tail_anchor": fam.tail_anchor,
        "notes": fam.notes,
        "conditions": {
            c: {
                "label": it.label,
                "subtype": it.subtype,
                "confidence": it.confidence,
                "source": it.source,
                "evidence_spans": it.evidence_spans,
                "turns": [asdict(t) for t in it.turns],
            }
            for c, it in fam.conditions.items()
        },
    }


# ---- text utilities shared by validator and diagnostics ----

_WORD = re.compile(r"[a-z0-9']+")


def norm_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def longest_common_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous common token run (classic DP, O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def repetition_scores(turns: list[Turn]) -> dict:
    """rep_user: max echo between distinct user turns.
    rep_any: max echo where the repeating turn is a user turn and the source is any earlier turn."""
    toks = [norm_tokens(t.text) for t in turns]
    rep_user = 0
    rep_any = 0
    rep_any_pair = None
    for i, t in enumerate(turns):
        if t.speaker != "user":
            continue
        for j in range(i):
            run = longest_common_run(toks[i], toks[j])
            if turns[j].speaker == "user":
                rep_user = max(rep_user, run)
            if run > rep_any:
                rep_any = run
                rep_any_pair = (j, i)
    return {"rep_user": rep_user, "rep_any": rep_any, "rep_any_pair": rep_any_pair}


def last_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[-1].strip() if parts else text.strip()
