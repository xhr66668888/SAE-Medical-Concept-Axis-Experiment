#!/usr/bin/env python3
"""Step 2: generate contrastive diabetes prompts from ICD and ATC CSV files."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_med.data_utils import read_csv_dicts, write_csv_dicts


COMPLICATIONS = {
    "none": {
        "phrase": "no documented complications",
        "terms": ("without complications", "without complication"),
    },
    "kidney": {
        "phrase": "kidney complications",
        "terms": ("kidney", "renal", "nephropathy", "chronic kidney"),
    },
    "neurological": {
        "phrase": "neurological complications",
        "terms": ("neurological", "neuropathy", "mononeuropathy", "polyneuropathy", "autonomic"),
    },
    "ophthalmic": {
        "phrase": "ophthalmic complications",
        "terms": ("ophthalmic", "retinopathy", "cataract", "macular"),
    },
    "circulatory": {
        "phrase": "peripheral circulatory complications",
        "terms": ("circulatory", "angiopathy", "peripheral"),
    },
}

TEMPLATE_VARIANTS = [
    "A patient diagnosed with {disease_label} (ICD-10: {icd_code}) presents with {complication_phrase}. The recommended ATC drug is",
    "Clinical note: {disease_label} using ICD-10 code {icd_code}, with {complication_phrase}. A plausible ATC medication class is",
    "For a patient with {disease_label} ({icd_code}) and {complication_phrase}, the ATC drug category most relevant to diabetes treatment is",
    "Medical coding context: diagnosis {icd_code} means {icd_description}. The patient has {complication_phrase}. The recommended ATC drug is",
]

FALLBACK_ICD = {
    "type1": {
        "code": "E10",
        "label": "Type 1 diabetes mellitus",
        "description": "Type 1 diabetes mellitus",
    },
    "type2": {
        "code": "E11",
        "label": "Type 2 diabetes mellitus",
        "description": "Type 2 diabetes mellitus",
    },
}

TARGET_ATC_BY_TYPE = {
    "type1": "A10AB01",  # insulin (human)
    "type2": "A10BA02",  # metformin
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate contrastive diabetes prompts from ICD/ATC CSVs.")
    parser.add_argument("--icd-csv", default="data/icd_diagnosis_ccs.csv")
    parser.add_argument("--atc-csv", default="data/atc_drug_hierarchy.csv")
    parser.add_argument("--output", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument(
        "--complications",
        default="none,kidney,neurological,ophthalmic,circulatory",
        help="Comma-separated complication keys.",
    )
    parser.add_argument("--variants-per-combo", type=int, default=3)
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def diabetes_type_for_row(row: dict[str, str]) -> str | None:
    code = row.get("ICD", "")
    description = normalize(row.get("ICDString", ""))
    integer_description = normalize(row.get("ICDIntegerString", ""))
    if code.startswith("E10") or "type 1 diabetes" in description or "type 1 diabetes" in integer_description:
        return "type1"
    if code.startswith("E11") or "type 2 diabetes" in description or "type 2 diabetes" in integer_description:
        return "type2"
    return None


def score_icd_row(row: dict[str, str], complication_key: str) -> tuple[int, int, str]:
    code = row.get("ICD", "")
    description = normalize(row.get("ICDString", ""))
    terms = COMPLICATIONS[complication_key]["terms"]
    term_score = max((len(term) for term in terms if term in description), default=0)
    exact_base_bonus = 100 if code in {"E10", "E11"} and complication_key == "none" else 0
    shorter_code_bonus = max(0, 20 - len(code))
    return exact_base_bonus + term_score + shorter_code_bonus, -len(description), code


def select_icd_rows(icd_rows: list[dict[str, str]], complication_keys: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in icd_rows:
        if row.get("Flag") != "10":
            continue
        diabetes_type = diabetes_type_for_row(row)
        if diabetes_type:
            grouped[diabetes_type].append(row)

    selected: dict[tuple[str, str], dict[str, str]] = {}
    for diabetes_type in ("type1", "type2"):
        for complication_key in complication_keys:
            candidates = grouped.get(diabetes_type, [])
            terms = COMPLICATIONS[complication_key]["terms"]
            matching = [row for row in candidates if any(term in normalize(row.get("ICDString", "")) for term in terms)]
            if not matching and complication_key == "none":
                matching = [row for row in candidates if row.get("ICD") in {"E10", "E11"}]
            if matching:
                selected[(diabetes_type, complication_key)] = max(
                    matching,
                    key=lambda row: score_icd_row(row, complication_key),
                )
            else:
                fallback = FALLBACK_ICD[diabetes_type]
                selected[(diabetes_type, complication_key)] = {
                    "ICD": fallback["code"],
                    "ICDString": fallback["description"],
                    "ICDIntegerString": fallback["label"],
                }
    return selected


def atc_lookup(atc_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup = {row.get("ATC_Code", ""): row for row in atc_rows}
    return lookup


def make_rows(
    selected_icd: dict[tuple[str, str], dict[str, str]],
    atc_by_code: dict[str, dict[str, str]],
    complication_keys: list[str],
    variants_per_combo: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prompt_index = 0
    for complication_key in complication_keys:
        complication = COMPLICATIONS[complication_key]
        for diabetes_type in ("type1", "type2"):
            icd = selected_icd[(diabetes_type, complication_key)]
            fallback = FALLBACK_ICD[diabetes_type]
            atc_code = TARGET_ATC_BY_TYPE[diabetes_type]
            atc = atc_by_code.get(atc_code, {})
            disease_label = fallback["label"]
            icd_code = icd.get("ICD") or fallback["code"]
            icd_description = icd.get("ICDString") or fallback["description"]
            for variant_idx, template in enumerate(TEMPLATE_VARIANTS[:variants_per_combo], start=1):
                prompt_index += 1
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index:04d}",
                        "diabetes_type": diabetes_type,
                        "diabetes_label": disease_label,
                        "icd_code": icd_code,
                        "icd_description": icd_description,
                        "complication": complication_key,
                        "complication_phrase": complication["phrase"],
                        "target_atc_code": atc_code,
                        "target_atc_name": atc.get("ATC_Name", ""),
                        "target_atc_l2": atc.get("ATC_L2_Name", ""),
                        "template_variant": variant_idx,
                        "prompt": template.format(
                            disease_label=disease_label,
                            icd_code=icd_code,
                            icd_description=icd_description,
                            complication_phrase=complication["phrase"],
                        ),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    complication_keys = [key.strip() for key in args.complications.split(",") if key.strip()]
    unknown = [key for key in complication_keys if key not in COMPLICATIONS]
    if unknown:
        raise SystemExit(f"Unknown complication key(s): {', '.join(unknown)}")
    if args.variants_per_combo < 1 or args.variants_per_combo > len(TEMPLATE_VARIANTS):
        raise SystemExit(f"--variants-per-combo must be between 1 and {len(TEMPLATE_VARIANTS)}")

    icd_rows = read_csv_dicts(args.icd_csv)
    atc_rows = read_csv_dicts(args.atc_csv)
    selected_icd = select_icd_rows(icd_rows, complication_keys)
    rows = make_rows(selected_icd, atc_lookup(atc_rows), complication_keys, args.variants_per_combo)

    fieldnames = [
        "prompt_id",
        "diabetes_type",
        "diabetes_label",
        "icd_code",
        "icd_description",
        "complication",
        "complication_phrase",
        "target_atc_code",
        "target_atc_name",
        "target_atc_l2",
        "template_variant",
        "prompt",
    ]
    write_csv_dicts(args.output, rows, fieldnames)
    print(f"Wrote {len(rows)} prompts to {args.output}")
    print("Example prompts:")
    for row in rows[:3]:
        print(f"- {row['prompt_id']} [{row['diabetes_type']} / {row['complication']}]: {row['prompt']}")


if __name__ == "__main__":
    main()
