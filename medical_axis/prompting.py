from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .concepts import ConceptAxis, ConceptSide, PROMPT_TEMPLATES


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def row_text(row: dict[str, str]) -> str:
    fields = (
        row.get("ICDString", ""),
        row.get("ICDIntegerString", ""),
        row.get("CCSString", ""),
        row.get("CCSParentString", ""),
        row.get("CCSChildString", ""),
    )
    return normalize_text(" ".join(fields))


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def matches_side(row: dict[str, str], side: ConceptSide) -> bool:
    text = row_text(row)
    if side.require_any and not _matches_any(text, side.require_any):
        return False
    if not all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in side.include):
        return False
    return not _matches_any(text, side.exclude)


def score_candidate(row: dict[str, str], side: ConceptSide) -> tuple[int, int, str]:
    text = row_text(row)
    exact_hits = sum(1 for pattern in side.include if re.search(pattern, text, flags=re.IGNORECASE))
    base_code_bonus = 4 if len(row.get("ICD", "")) <= 5 else 0
    icd10_bonus = 2 if row.get("Flag") == "10" else 0
    description_len = len(row.get("ICDString", ""))
    return exact_hits * 10 + base_code_bonus + icd10_bonus, -description_len, row.get("ICD", "")


def select_side_rows(
    rows: list[dict[str, str]],
    side: ConceptSide,
    *,
    max_rows: int,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if matches_side(row, side)]
    candidates = sorted(candidates, key=lambda row: score_candidate(row, side), reverse=True)
    seen_descriptions: set[str] = set()
    selected: list[dict[str, str]] = []
    for row in candidates:
        description = normalize_text(row.get("ICDString", ""))
        if not description or description in seen_descriptions:
            continue
        selected.append(row)
        seen_descriptions.add(description)
        if len(selected) >= max_rows:
            break
    return selected


def split_for_template(template_id: int, heldout_template_ids: set[int]) -> str:
    return "test" if template_id in heldout_template_ids else "train"


def stable_pair_key(axis_id: str, pair_index: int) -> str:
    digest = hashlib.sha1(f"{axis_id}:{pair_index}".encode("utf-8")).hexdigest()[:10]
    return f"{axis_id}_{pair_index:04d}_{digest}"


def generate_prompt_rows(
    icd_rows: list[dict[str, str]],
    axes: Sequence[ConceptAxis],
    *,
    templates: Sequence[str] = PROMPT_TEMPLATES,
    max_pairs_per_axis: int = 40,
    heldout_template_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    heldout_template_ids = heldout_template_ids or {8, 9, 10}
    output: list[dict[str, object]] = []

    for axis in axes:
        positive_rows = select_side_rows(icd_rows, axis.positive, max_rows=max_pairs_per_axis)
        negative_rows = select_side_rows(icd_rows, axis.negative, max_rows=max_pairs_per_axis)
        pair_count = min(len(positive_rows), len(negative_rows), max_pairs_per_axis)
        if pair_count == 0:
            continue

        for pair_idx in range(pair_count):
            pair_id = stable_pair_key(axis.axis_id, pair_idx + 1)
            paired = (
                ("positive", axis.positive, positive_rows[pair_idx]),
                ("negative", axis.negative, negative_rows[pair_idx]),
            )
            for side_name, side, row in paired:
                for template_idx, template in enumerate(templates, start=1):
                    icd_code = row.get("ICD", "")
                    description = row.get("ICDString") or row.get("ICDIntegerString") or ""
                    prompt = template.format(icd_code=icd_code, icd_description=description)
                    output.append(
                        {
                            "axis_id": axis.axis_id,
                            "axis_description": axis.description,
                            "side": side_name,
                            "side_name": side.name,
                            "concept_label": side.label,
                            "opposite_label": axis.negative.label if side_name == "positive" else axis.positive.label,
                            "pair_id": pair_id,
                            "template_id": template_idx,
                            "split": split_for_template(template_idx, heldout_template_ids),
                            "icd_code": icd_code,
                            "icd_description": description,
                            "ccs_string": row.get("CCSString", ""),
                            "prompt": prompt,
                        }
                    )
    return output
