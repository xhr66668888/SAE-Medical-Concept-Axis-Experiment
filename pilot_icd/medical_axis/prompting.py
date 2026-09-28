from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .concepts import ConceptAxis, ConceptSide, PROMPT_TEMPLATES


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def row_ccs_code(row: dict[str, str]) -> str:
    return str(row.get("ccs_code") or row.get("CCSCode") or "").strip()


def row_ccs_label(row: dict[str, str]) -> str:
    return str(row.get("ccs_label") or row.get("CCSString") or "").strip()


def row_ccs_source(row: dict[str, str]) -> str:
    return str(row.get("ccs_source") or row.get("CCSSource") or "").strip()


def row_text(row: dict[str, str]) -> str:
    fields = (
        row.get("ICDString", ""),
        row.get("ICDIntegerString", ""),
        row.get("CCSString", ""),
        row.get("ccs_label", ""),
        row.get("CCSParentString", ""),
        row.get("CCSChildString", ""),
    )
    return normalize_text(" ".join(fields))


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def matches_side(row: dict[str, str], side: ConceptSide) -> bool:
    ccs_code = row_ccs_code(row)
    if side.ccs_codes and ccs_code not in {str(code).strip() for code in side.ccs_codes}:
        return False
    if side.exclude_ccs_codes and ccs_code in {str(code).strip() for code in side.exclude_ccs_codes}:
        return False
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


def pair_split_indices(axis_id: str, pair_count: int, *, test_fraction: float, seed: int) -> set[int]:
    if pair_count <= 1:
        return set()
    target = int(round(pair_count * test_fraction))
    if pair_count >= 4:
        target = max(1, target)
    target = min(target, pair_count - 1)
    ranked = sorted(
        range(pair_count),
        key=lambda index: hashlib.sha1(f"{seed}:{axis_id}:{index}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[:target])


def combined_split(pair_split: str, template_split: str) -> str:
    if pair_split == "train" and template_split == "train":
        return "train"
    if pair_split == "test" and template_split == "test":
        return "test"
    return "calibration"


def stable_pair_key(axis_id: str, pair_index: int) -> str:
    digest = hashlib.sha1(f"{axis_id}:{pair_index}".encode("utf-8")).hexdigest()[:10]
    return f"{axis_id}_{pair_index:04d}_{digest}"


def generate_prompt_rows(
    icd_rows: list[dict[str, str]],
    axes: Sequence[ConceptAxis],
    *,
    templates: Sequence[str] = PROMPT_TEMPLATES,
    max_pairs_per_axis: int = 120,
    min_primary_side_rows: int = 10,
    include_exploratory: bool = False,
    heldout_template_ids: set[int] | None = None,
    heldout_pair_fraction: float = 0.25,
    split_seed: int = 20260528,
) -> list[dict[str, object]]:
    heldout_template_ids = heldout_template_ids or {8, 9, 10}
    if not 0.0 <= heldout_pair_fraction < 1.0:
        raise ValueError("heldout_pair_fraction must be in [0, 1).")
    output: list[dict[str, object]] = []

    for axis in axes:
        if not axis.primary_axis and not include_exploratory:
            continue
        positive_rows = select_side_rows(icd_rows, axis.positive, max_rows=max_pairs_per_axis)
        negative_rows = select_side_rows(icd_rows, axis.negative, max_rows=max_pairs_per_axis)
        side_row_floor = min_primary_side_rows
        if axis.min_side_rows is not None:
            side_row_floor = min(side_row_floor, axis.min_side_rows)
        if axis.primary_axis and min(len(positive_rows), len(negative_rows)) < side_row_floor:
            continue
        pair_count = min(len(positive_rows), len(negative_rows), max_pairs_per_axis)
        if pair_count == 0:
            continue
        test_pair_indices = pair_split_indices(
            axis.axis_id,
            pair_count,
            test_fraction=heldout_pair_fraction,
            seed=split_seed,
        )

        for pair_idx in range(pair_count):
            pair_id = stable_pair_key(axis.axis_id, pair_idx + 1)
            pair_split = "test" if pair_idx in test_pair_indices else "train"
            paired = (
                ("positive", axis.positive, positive_rows[pair_idx]),
                ("negative", axis.negative, negative_rows[pair_idx]),
            )
            for side_name, side, row in paired:
                for template_idx, template in enumerate(templates, start=1):
                    icd_code = row.get("ICD", "")
                    description = row.get("ICDString") or row.get("ICDIntegerString") or ""
                    ccs_code = row_ccs_code(row)
                    ccs_label = row_ccs_label(row)
                    ccs_source = row_ccs_source(row) or axis.ccs_source
                    prompt = template.format(
                        icd_code=icd_code,
                        icd_description=description,
                        ccs_code=ccs_code,
                        ccs_label=ccs_label,
                    )
                    template_split = split_for_template(template_idx, heldout_template_ids)
                    output.append(
                        {
                            "axis_id": axis.axis_id,
                            "axis_description": axis.description,
                            "axis_family": axis.axis_family,
                            "primary_axis": axis.primary_axis,
                            "side": side_name,
                            "side_name": side.name,
                            "concept_label": side.label,
                            "opposite_label": axis.negative.label if side_name == "positive" else axis.positive.label,
                            "pair_id": pair_id,
                            "pair_split": pair_split,
                            "template_id": template_idx,
                            "template_split": template_split,
                            "split": combined_split(pair_split, template_split),
                            "icd_code": icd_code,
                            "icd_description": description,
                            "ccs_code": ccs_code,
                            "ccs_label": ccs_label,
                            "ccs_source": ccs_source,
                            "ccs_string": row.get("CCSString", ccs_label),
                            "prompt": prompt,
                        }
                    )
    return output
