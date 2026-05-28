#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis import DEFAULT_AXES
from medical_axis.io import read_csv, write_csv
from medical_axis.prompting import generate_prompt_rows


FIELDNAMES = [
    "axis_id",
    "axis_description",
    "side",
    "side_name",
    "concept_label",
    "opposite_label",
    "pair_id",
    "template_id",
    "split",
    "icd_code",
    "icd_description",
    "ccs_string",
    "prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matched medical concept prompts from ICD/CCS rows.")
    parser.add_argument("--icd-csv", default="data/icd_diagnosis_ccs.csv")
    parser.add_argument("--output", default="outputs/concept_prompts.csv")
    parser.add_argument("--max-pairs-per-axis", type=int, default=40)
    parser.add_argument("--heldout-template-ids", default="8,9,10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout = {int(part.strip()) for part in args.heldout_template_ids.split(",") if part.strip()}
    rows = generate_prompt_rows(
        read_csv(args.icd_csv),
        DEFAULT_AXES,
        max_pairs_per_axis=args.max_pairs_per_axis,
        heldout_template_ids=heldout,
    )
    if not rows:
        raise SystemExit("No prompt rows were generated. Check ICD input and concept patterns.")
    write_csv(args.output, rows, FIELDNAMES)
    by_axis: dict[str, int] = {}
    for row in rows:
        by_axis[str(row["axis_id"])] = by_axis.get(str(row["axis_id"]), 0) + 1
    print(f"Wrote {len(rows)} prompts to {args.output}")
    for axis_id, count in sorted(by_axis.items()):
        print(f"  {axis_id}: {count}")


if __name__ == "__main__":
    main()
