"""Rule-based validator enforcing the Stage-1 design constraints."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bridge2026.schema import (
    CONDITIONS, CONDITION_LABEL, CONDITION_REPETITION, DOMAINS,
    POSITIVE_SUBTYPES, NEGATIVE_SUBTYPES,
    Family, Item, load_family, load_indeterminate,
    norm_tokens, repetition_scores, last_sentence,
)

LEN_TOL = 0.18            # +-18% of family mean characters
QMARK_TOL = 1             # max spread of '?' counts inside a family
REP_MIN = 5               # rep conditions need an echo of >= 5 contiguous tokens
NOREP_MAX = 4             # norep conditions must stay <= 4

FORBIDDEN = [
    r"\bdementia\b", r"\balzheimer", r"\bcognitive\b", r"\bcognition\b",
    r"\bmemory (?:loss|problem|issue|trouble)\b", r"\bforgetful", r"\bdiagnos",
    r"\bdementia-like\b", r"\bsenile\b", r"\bmci\b", r"\bdecline\b",
    r"\bmy age\b", r"\bat my age\b", r"\bgetting old\b", r"\bold age\b",
    r"\bimpair", r"\bconfusion disorder\b", r"\bneurolog",
]
LEAK = [
    r"\brep_unres\b", r"\brep_res\b", r"\bnorep_unres\b", r"\bnorep_res\b",
    r"\bunresolved\b", r"\bresolved\b", r"\bindeterminate\b",
    r"\bunanswered_question\b", r"\bcontradiction\b", r"\bambiguous_reference\b",
    r"\bevidence_span", r"\bfamily_id\b", r"\bcondition\b:",
]


def _check_item(it: Item, errs: list[str], warns: list[str]) -> dict:
    tag = it.item_id
    blob = it.text_blob()
    low = blob.lower()

    if it.label != CONDITION_LABEL.get(it.condition, it.label):
        errs.append(f"{tag}: label {it.label!r} does not match condition {it.condition!r}")
    if it.label == "unresolved" and it.subtype not in POSITIVE_SUBTYPES:
        errs.append(f"{tag}: positive subtype {it.subtype!r} not in {POSITIVE_SUBTYPES}")
    if it.label == "resolved" and it.subtype not in NEGATIVE_SUBTYPES:
        errs.append(f"{tag}: negative subtype {it.subtype!r} not in {NEGATIVE_SUBTYPES}")

    speakers = [t.speaker for t in it.turns]
    if set(speakers) - {"user", "assistant"}:
        errs.append(f"{tag}: bad speaker labels {set(speakers)}")
    if speakers and speakers[0] != "user":
        errs.append(f"{tag}: dialogue must start with user")
    if speakers and speakers[-1] != "user":
        errs.append(f"{tag}: dialogue must end on a user turn")
    for a, b in zip(speakers, speakers[1:]):
        if a == b:
            errs.append(f"{tag}: consecutive turns by same speaker ({a})")
            break
    if not (4 <= len(it.turns) <= 8):
        warns.append(f"{tag}: n_turns={len(it.turns)} outside the 4-8 target band")

    for pat in FORBIDDEN:
        m = re.search(pat, low)
        if m:
            errs.append(f"{tag}: forbidden health/age content {m.group(0)!r}")
    for pat in LEAK:
        m = re.search(pat, low)
        if m:
            errs.append(f"{tag}: metadata leak in text {m.group(0)!r}")

    rep = repetition_scores(it.turns)
    want = CONDITION_REPETITION.get(it.condition)
    if want == "rep" and rep["rep_any"] < REP_MIN:
        errs.append(f"{tag}: rep condition but rep_any={rep['rep_any']} < {REP_MIN}")
    if want == "norep" and rep["rep_any"] > NOREP_MAX:
        errs.append(f"{tag}: norep condition but rep_any={rep['rep_any']} > {NOREP_MAX}")

    if not it.evidence_spans:
        errs.append(f"{tag}: no evidence spans")
    for sp in it.evidence_spans:
        ti = sp.get("turn_index")
        quote = (sp.get("quote") or "").strip()
        if ti is None or not isinstance(ti, int) or not (0 <= ti < len(it.turns)):
            errs.append(f"{tag}: evidence turn_index {ti!r} out of range")
            continue
        if quote and quote.lower() not in it.turns[ti].text.lower():
            errs.append(f"{tag}: evidence quote not found in turn {ti}: {quote[:60]!r}")
    if it.confidence not in ("high", "medium", "low"):
        errs.append(f"{tag}: bad confidence {it.confidence!r}")
    if it.confidence == "low":
        warns.append(f"{tag}: low confidence should be routed to indeterminate")

    return {
        "item_id": tag,
        "family_id": it.family_id,
        "condition": it.condition,
        "domain": it.domain,
        "label": it.label,
        "subtype": it.subtype,
        "n_turns": len(it.turns),
        "n_chars": len(blob),
        "n_words": len(norm_tokens(blob)),
        "n_qmarks": blob.count("?"),
        "rep_user": rep["rep_user"],
        "rep_any": rep["rep_any"],
    }


def check_family(fam: Family) -> tuple[list[str], list[str], list[dict]]:
    errs: list[str] = []
    warns: list[str] = []
    rows: list[dict] = []

    if fam.domain not in DOMAINS:
        errs.append(f"{fam.family_id}: unknown domain {fam.domain!r}")
    missing = set(CONDITIONS) - set(fam.conditions)
    if missing:
        errs.append(f"{fam.family_id}: missing conditions {sorted(missing)}")
        return errs, warns, rows
    extra = set(fam.conditions) - set(CONDITIONS)
    if extra:
        errs.append(f"{fam.family_id}: unexpected conditions {sorted(extra)}")

    for cond in CONDITIONS:
        rows.append(_check_item(fam.conditions[cond], errs, warns))

    nts = {c: len(fam.conditions[c].turns) for c in CONDITIONS}
    if len(set(nts.values())) != 1:
        errs.append(f"{fam.family_id}: turn counts differ across conditions {nts}")
    elif next(iter(nts.values())) != fam.n_turns:
        errs.append(f"{fam.family_id}: declared n_turns={fam.n_turns} but got {nts}")

    seqs = {c: tuple(t.speaker for t in fam.conditions[c].turns) for c in CONDITIONS}
    if len(set(seqs.values())) != 1:
        errs.append(f"{fam.family_id}: speaker sequences differ across conditions")

    anchors = {}
    for c in CONDITIONS:
        final_user = fam.conditions[c].turns[-1].text
        anchors[c] = last_sentence(final_user)
    if len(set(anchors.values())) != 1:
        errs.append(f"{fam.family_id}: tail anchors differ across conditions: {anchors}")
    elif next(iter(anchors.values())) != fam.tail_anchor:
        errs.append(
            f"{fam.family_id}: declared tail_anchor {fam.tail_anchor!r} != actual "
            f"{next(iter(anchors.values()))!r}"
        )

    lens = [r["n_chars"] for r in rows]
    mean = sum(lens) / len(lens)
    for r in rows:
        dev = abs(r["n_chars"] - mean) / mean
        if dev > LEN_TOL:
            errs.append(
                f"{fam.family_id}/{r['condition']}: length {r['n_chars']} deviates "
                f"{dev:.0%} from family mean {mean:.0f} (tol {LEN_TOL:.0%})"
            )

    qs = [r["n_qmarks"] for r in rows]
    if max(qs) - min(qs) > QMARK_TOL:
        errs.append(f"{fam.family_id}: question-mark spread {qs} exceeds {QMARK_TOL}")

    return errs, warns, rows


def check_indeterminate(items: list[Item]) -> tuple[list[str], list[str], list[dict]]:
    errs: list[str] = []
    warns: list[str] = []
    rows: list[dict] = []
    for it in items:
        blob = it.text_blob()
        low = blob.lower()
        for pat in FORBIDDEN:
            m = re.search(pat, low)
            if m:
                errs.append(f"{it.item_id}: forbidden content {m.group(0)!r}")
        for pat in LEAK:
            m = re.search(pat, low)
            if m:
                errs.append(f"{it.item_id}: metadata leak {m.group(0)!r}")
        sp = [t.speaker for t in it.turns]
        if not sp or sp[0] != "user" or sp[-1] != "user":
            errs.append(f"{it.item_id}: must start and end on user")
        rep = repetition_scores(it.turns)
        rows.append({
            "item_id": it.item_id, "family_id": it.item_id, "condition": "indeterminate",
            "domain": it.domain, "label": "indeterminate", "subtype": it.subtype,
            "n_turns": len(it.turns), "n_chars": len(blob), "n_words": len(norm_tokens(blob)),
            "n_qmarks": blob.count("?"), "rep_user": rep["rep_user"], "rep_any": rep["rep_any"],
        })
    return errs, warns, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-dir", default="data/bridge2026/families")
    ap.add_argument("--indeterminate", default="data/bridge2026/indeterminate.json")
    ap.add_argument("--out-rows", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(args.families_dir).glob("*.json"))
    all_errs, all_warns, all_rows = [], [], []
    seen_ids: set[str] = set()
    for p in paths:
        try:
            fam = load_family(p)
        except Exception as exc:  # noqa: BLE001
            all_errs.append(f"{p.name}: cannot parse ({type(exc).__name__}: {exc})")
            continue
        if fam.family_id in seen_ids:
            all_errs.append(f"{p.name}: duplicate family_id {fam.family_id}")
        seen_ids.add(fam.family_id)
        e, w, r = check_family(fam)
        all_errs += e
        all_warns += w
        all_rows += r

    ip = Path(args.indeterminate)
    if ip.exists():
        items = load_indeterminate(ip)
        e, w, r = check_indeterminate(items)
        all_errs += e
        all_warns += w
        all_rows += r

    if args.out_rows:
        Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(args.out_rows, "w", newline="") as fh:
            if all_rows:
                wtr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(all_rows)

    n_fam = len(seen_ids)
    print(f"families={n_fam} items={len(all_rows)} errors={len(all_errs)} warnings={len(all_warns)}")
    if not args.quiet:
        for e in all_errs:
            print("  ERROR  ", e)
        for w in all_warns[:40]:
            print("  warn   ", w)
    return 1 if all_errs else 0


if __name__ == "__main__":
    sys.exit(main())
