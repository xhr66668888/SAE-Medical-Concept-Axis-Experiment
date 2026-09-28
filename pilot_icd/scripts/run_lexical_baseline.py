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
from medical_axis.prompting import matches_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a keyword/ontology lexical baseline for concept labels.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--output", default="outputs/lexical_baseline.csv")
    parser.add_argument("--use-split", choices=["train", "test", "all"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axes = {axis.axis_id: axis for axis in DEFAULT_AXES}
    rows = read_csv(args.prompts)
    if args.use_split != "all":
        rows = [row for row in rows if row.get("split") == args.use_split]

    detailed = []
    for row in rows:
        axis = axes[row["axis_id"]]
        icd_row = {
            "ICDString": row.get("icd_description", ""),
            "ICDIntegerString": row.get("icd_description", ""),
            "CCSString": row.get("ccs_string", ""),
            "ccs_code": row.get("ccs_code", ""),
            "ccs_label": row.get("ccs_label", ""),
            "ccs_source": row.get("ccs_source", ""),
        }
        pos = matches_side(icd_row, axis.positive)
        neg = matches_side(icd_row, axis.negative)
        if pos and not neg:
            prediction = "positive"
        elif neg and not pos:
            prediction = "negative"
        else:
            prediction = "abstain"
        detailed.append(
            {
                "axis_id": row["axis_id"],
                "split": row.get("split", ""),
                "side": row["side"],
                "prediction": prediction,
                "correct": int(prediction == row["side"]),
                "abstain": int(prediction == "abstain"),
            }
        )

    summary = []
    for axis_id in sorted({row["axis_id"] for row in detailed}):
        group = [row for row in detailed if row["axis_id"] == axis_id]
        answered = [row for row in group if row["prediction"] != "abstain"]
        summary.append(
            {
                "axis_id": axis_id,
                "split": args.use_split,
                "rows": len(group),
                "answered": len(answered),
                "coverage": len(answered) / len(group) if group else 0.0,
                "accuracy": sum(row["correct"] for row in answered) / len(answered) if answered else 0.0,
                "accuracy_with_abstain_wrong": sum(row["correct"] for row in group) / len(group) if group else 0.0,
            }
        )

    write_csv(
        Path(args.output),
        summary,
        ["axis_id", "split", "rows", "answered", "coverage", "accuracy", "accuracy_with_abstain_wrong"],
    )
    print(f"Wrote lexical baseline to {args.output}")


if __name__ == "__main__":
    main()
