#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis import DEFAULT_AXES
from medical_axis.ccs import enrich_icd9_ccs_rows
from medical_axis.io import read_csv, write_csv
from medical_axis.prompting import generate_prompt_rows


FIELDNAMES = [
    "axis_id",
    "axis_description",
    "axis_family",
    "primary_axis",
    "side",
    "side_name",
    "concept_label",
    "opposite_label",
    "pair_id",
    "pair_split",
    "template_id",
    "template_split",
    "split",
    "icd_code",
    "icd_description",
    "ccs_code",
    "ccs_label",
    "ccs_source",
    "ccs_string",
    "prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matched medical concept prompts from ICD/CCS rows.")
    parser.add_argument("--icd-csv", default="data/icd_diagnosis_ccs.csv")
    parser.add_argument("--ccs-appendix", default="AppendixASingleDX.txt")
    parser.add_argument("--output", default="outputs/concept_prompts.csv")
    parser.add_argument("--max-pairs-per-axis", type=int, default=120)
    parser.add_argument("--min-primary-side-rows", type=int, default=30)
    parser.add_argument("--min-primary-pairs-per-side", type=int, default=None)
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--heldout-template-ids", default="8,9,10")
    parser.add_argument("--heldout-pair-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260528)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout = {int(part.strip()) for part in args.heldout_template_ids.split(",") if part.strip()}
    icd_rows = read_csv(args.icd_csv)
    primary_rows = enrich_icd9_ccs_rows(icd_rows, args.ccs_appendix)
    rows = generate_prompt_rows(
        primary_rows,
        DEFAULT_AXES,
        max_pairs_per_axis=args.max_pairs_per_axis,
        min_primary_side_rows=args.min_primary_pairs_per_side or args.min_primary_side_rows,
        include_exploratory=args.include_exploratory,
        heldout_template_ids=heldout,
        heldout_pair_fraction=args.heldout_pair_fraction,
        split_seed=args.split_seed,
    )
    if not rows:
        raise SystemExit("No prompt rows were generated. Check ICD input and concept patterns.")
    write_csv(args.output, rows, FIELDNAMES)
    by_axis: dict[str, int] = {}
    for row in rows:
        by_axis[str(row["axis_id"])] = by_axis.get(str(row["axis_id"]), 0) + 1
    print(f"Loaded {len(primary_rows)} ICD-9 rows that join to {args.ccs_appendix}")
    print(f"Wrote {len(rows)} prompts to {args.output}")
    for axis_id, count in sorted(by_axis.items()):
        print(f"  {axis_id}: {count}")
    by_split: dict[str, int] = {}
    for row in rows:
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1
    for split, count in sorted(by_split.items()):
        print(f"  split {split}: {count}")


if __name__ == "__main__":
    main()
