from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


APPENDIX_A_SINGLE_DX_SOURCE = "appendix_a_single_dx"
DEFAULT_APPENDIX_A_SINGLE_DX_PATH = Path(__file__).resolve().parents[1] / "AppendixASingleDX.txt"


@dataclass(frozen=True)
class CCSCategory:
    """Canonical CCS single-level diagnosis category from Appendix A."""

    code: str
    label: str
    icd9_codes: tuple[str, ...]


_CATEGORY_RE = re.compile(r"^(\d+)\s+(.+?)\s*$")
_NON_CODE_CHARS_RE = re.compile(r"[^A-Z0-9]")


def normalize_icd9_code(code: str) -> str:
    """Normalize an ICD-9 code to the no-dot Appendix A representation."""

    return _NON_CODE_CHARS_RE.sub("", (code or "").upper())


def parse_appendix_a_single_dx(path: str | Path = DEFAULT_APPENDIX_A_SINGLE_DX_PATH) -> dict[str, CCSCategory]:
    """Parse HCUP CCS Appendix A single-level diagnosis categories."""

    categories: dict[str, CCSCategory] = {}
    current_code = ""
    current_label = ""
    current_icd9_codes: list[str] = []

    def flush_current() -> None:
        nonlocal current_code, current_label, current_icd9_codes
        if not current_code:
            return
        categories[current_code] = CCSCategory(
            code=current_code,
            label=current_label,
            icd9_codes=tuple(dict.fromkeys(current_icd9_codes)),
        )
        current_code = ""
        current_label = ""
        current_icd9_codes = []

    appendix_path = Path(path)
    for raw_line in appendix_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        header_match = _CATEGORY_RE.match(raw_line)
        if header_match and not raw_line[0].isspace():
            flush_current()
            current_code = header_match.group(1)
            current_label = " ".join(header_match.group(2).split())
            continue
        if current_code and raw_line[0].isspace():
            current_icd9_codes.extend(
                normalized
                for normalized in (normalize_icd9_code(token) for token in raw_line.split())
                if normalized
            )
    flush_current()
    return categories


def build_icd9_ccs_lookup(categories: Mapping[str, CCSCategory] | Iterable[CCSCategory]) -> dict[str, CCSCategory]:
    """Build a normalized ICD-9 code to CCS category lookup."""

    category_values = categories.values() if isinstance(categories, Mapping) else categories
    lookup: dict[str, CCSCategory] = {}
    for category in category_values:
        for icd9_code in category.icd9_codes:
            if icd9_code in lookup:
                raise ValueError(f"ICD-9 code {icd9_code} appears in multiple CCS categories.")
            lookup[icd9_code] = category
    return lookup


def enrich_icd9_ccs_rows(
    rows: Iterable[dict[str, str]],
    appendix_path: str | Path = DEFAULT_APPENDIX_A_SINGLE_DX_PATH,
) -> list[dict[str, str]]:
    """Keep ICD-9 rows that join to Appendix A and attach canonical CCS metadata."""

    categories = parse_appendix_a_single_dx(appendix_path)
    lookup = build_icd9_ccs_lookup(categories)
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("Flag", "")).strip() != "9":
            continue
        normalized_icd9_code = normalize_icd9_code(row.get("ICD", ""))
        category = lookup.get(normalized_icd9_code)
        if category is None:
            continue
        enriched = dict(row)
        enriched["normalized_icd9_code"] = normalized_icd9_code
        enriched["CCSCode"] = category.code
        enriched["CCSString"] = category.label
        enriched["ccs_code"] = category.code
        enriched["ccs_label"] = category.label
        enriched["ccs_source"] = APPENDIX_A_SINGLE_DX_SOURCE
        enriched_rows.append(enriched)
    return enriched_rows
