from __future__ import annotations

import unittest

from medical_axis import DEFAULT_AXES
from medical_axis.prompting import generate_prompt_rows, matches_side


class PromptGenerationTest(unittest.TestCase):
    def test_diabetes_subtype_matching(self) -> None:
        axis = next(axis for axis in DEFAULT_AXES if axis.axis_id == "diabetes_subtype")
        type1 = {"ICDString": "Type 1 diabetes mellitus without complications"}
        type2 = {"ICDString": "Type 2 diabetes mellitus without complications"}
        self.assertTrue(matches_side(type1, axis.positive))
        self.assertFalse(matches_side(type1, axis.negative))
        self.assertTrue(matches_side(type2, axis.negative))
        self.assertFalse(matches_side(type2, axis.positive))

    def test_generated_rows_are_paired_and_split_by_template(self) -> None:
        rows = [
            {"ICD": "E10.9", "Flag": "10", "ICDString": "Type 1 diabetes mellitus without complications"},
            {"ICD": "E11.9", "Flag": "10", "ICDString": "Type 2 diabetes mellitus without complications"},
        ]
        axis = [axis for axis in DEFAULT_AXES if axis.axis_id == "diabetes_subtype"]
        generated = generate_prompt_rows(rows, axis, max_pairs_per_axis=1, heldout_template_ids={2})
        self.assertEqual(len(generated), 20)
        self.assertEqual({row["pair_id"] for row in generated}, {generated[0]["pair_id"]})
        self.assertEqual({row["split"] for row in generated if row["template_id"] == 2}, {"test"})
        self.assertEqual({row["split"] for row in generated if row["template_id"] == 1}, {"train"})


if __name__ == "__main__":
    unittest.main()
