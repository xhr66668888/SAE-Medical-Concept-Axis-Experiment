from __future__ import annotations

from pathlib import Path
import unittest

from medical_axis import DEFAULT_AXES
from medical_axis.ccs import APPENDIX_A_SINGLE_DX_SOURCE, enrich_icd9_ccs_rows, parse_appendix_a_single_dx
from medical_axis.prompting import generate_prompt_rows, matches_side


APPENDIX_PATH = Path(__file__).resolve().parents[1] / "AppendixASingleDX.txt"


class PromptGenerationTest(unittest.TestCase):
    def test_appendix_parser_loads_canonical_diabetes_categories(self) -> None:
        categories = parse_appendix_a_single_dx(APPENDIX_PATH)

        self.assertGreaterEqual(len(categories), 280)
        self.assertEqual(categories["49"].label, "Diabetes mellitus without complication")
        self.assertEqual(categories["50"].label, "Diabetes mellitus with complications")
        self.assertEqual(
            categories["186"].label,
            "Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium",
        )
        self.assertIn("25000", categories["49"].icd9_codes)
        self.assertIn("25040", categories["50"].icd9_codes)
        self.assertIn("64800", categories["186"].icd9_codes)
        self.assertNotIn("250.00", categories["49"].icd9_codes)

    def test_primary_filter_requires_icd9_rows_that_join_appendix(self) -> None:
        rows = [
            {"ICD": "250.00", "Flag": "9", "ICDString": "type II diabetes mellitus without complication"},
            {"ICD": "250", "Flag": "9", "ICDString": "Diabetes mellitus base code"},
            {"ICD": "E11.9", "Flag": "10", "ICDString": "Type 2 diabetes mellitus without complications"},
            {"ICD": "XYZ", "Flag": "9", "ICDString": "Unmatched synthetic diagnosis"},
        ]

        enriched = enrich_icd9_ccs_rows(rows, APPENDIX_PATH)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["normalized_icd9_code"], "25000")
        self.assertEqual(enriched[0]["ccs_code"], "49")
        self.assertEqual(enriched[0]["ccs_label"], "Diabetes mellitus without complication")
        self.assertEqual(enriched[0]["ccs_source"], APPENDIX_A_SINGLE_DX_SOURCE)

    def test_diabetes_complication_axis_matches_ccs_49_50(self) -> None:
        axis = next(axis for axis in DEFAULT_AXES if axis.axis_id == "diabetes_complication_status_ccs")
        complicated = {"ICDString": "Diabetes mellitus with renal manifestations", "ccs_code": "50"}
        uncomplicated = {"ICDString": "Diabetes mellitus without complication", "ccs_code": "49"}

        self.assertTrue(axis.primary_axis)
        self.assertTrue(matches_side(complicated, axis.positive))
        self.assertFalse(matches_side(complicated, axis.negative))
        self.assertTrue(matches_side(uncomplicated, axis.negative))
        self.assertFalse(matches_side(uncomplicated, axis.positive))

    def test_ccs_186_is_excluded_from_diabetes_primary_axis(self) -> None:
        axis = next(axis for axis in DEFAULT_AXES if axis.axis_id == "diabetes_complication_status_ccs")
        pregnancy_diabetes = {
            "ICDString": "Diabetes mellitus of mother complicating pregnancy",
            "CCSCode": "186",
            "CCSString": "Diabetes or abnormal glucose tolerance complicating pregnancy",
        }

        self.assertFalse(matches_side(pregnancy_diabetes, axis.positive))
        self.assertFalse(matches_side(pregnancy_diabetes, axis.negative))

    def test_diabetes_subtype_is_exploratory_icd_derived(self) -> None:
        axis = next(axis for axis in DEFAULT_AXES if axis.axis_id == "exploratory_icd_diabetes_subtype")
        type1 = {"ICDString": "Type 1 diabetes mellitus without complications"}
        type2 = {"ICDString": "Type 2 diabetes mellitus without complications"}

        self.assertFalse(axis.primary_axis)
        self.assertEqual(axis.axis_family, "diabetes_icd_derived")
        self.assertTrue(matches_side(type1, axis.positive))
        self.assertFalse(matches_side(type1, axis.negative))
        self.assertTrue(matches_side(type2, axis.negative))
        self.assertFalse(matches_side(type2, axis.positive))

    def test_generated_rows_are_paired_and_split_by_pair_and_template(self) -> None:
        rows = [
            {
                "ICD": "250.40",
                "Flag": "9",
                "ICDString": "Diabetes with renal manifestations, type II",
                "ccs_code": "50",
                "ccs_label": "Diabetes mellitus with complications",
                "ccs_source": APPENDIX_A_SINGLE_DX_SOURCE,
            },
            {
                "ICD": "250.60",
                "Flag": "9",
                "ICDString": "Diabetes with neurological manifestations, type II",
                "ccs_code": "50",
                "ccs_label": "Diabetes mellitus with complications",
                "ccs_source": APPENDIX_A_SINGLE_DX_SOURCE,
            },
            {
                "ICD": "250.00",
                "Flag": "9",
                "ICDString": "Diabetes mellitus type II without complication",
                "ccs_code": "49",
                "ccs_label": "Diabetes mellitus without complication",
                "ccs_source": APPENDIX_A_SINGLE_DX_SOURCE,
            },
            {
                "ICD": "250.01",
                "Flag": "9",
                "ICDString": "Diabetes mellitus type I without complication",
                "ccs_code": "49",
                "ccs_label": "Diabetes mellitus without complication",
                "ccs_source": APPENDIX_A_SINGLE_DX_SOURCE,
            },
            {
                "ICD": "648.00",
                "Flag": "9",
                "ICDString": "Diabetes mellitus of mother complicating pregnancy",
                "ccs_code": "186",
                "ccs_label": "Diabetes or abnormal glucose tolerance complicating pregnancy",
                "ccs_source": APPENDIX_A_SINGLE_DX_SOURCE,
            },
        ]
        axis = [axis for axis in DEFAULT_AXES if axis.axis_id == "diabetes_complication_status_ccs"]
        generated = generate_prompt_rows(
            rows,
            axis,
            max_pairs_per_axis=2,
            min_primary_side_rows=1,
            heldout_template_ids={2},
            heldout_pair_fraction=0.5,
            split_seed=0,
        )
        self.assertEqual(len(generated), 40)
        self.assertEqual({row["pair_split"] for row in generated}, {"train", "test"})
        self.assertEqual({row["template_split"] for row in generated if row["template_id"] == 2}, {"test"})
        self.assertIn("calibration", {row["split"] for row in generated})
        self.assertEqual({row["axis_family"] for row in generated}, {"diabetes_ccs"})
        self.assertEqual({row["primary_axis"] for row in generated}, {True})
        self.assertEqual({row["ccs_source"] for row in generated}, {APPENDIX_A_SINGLE_DX_SOURCE})
        self.assertEqual({row["ccs_code"] for row in generated}, {"49", "50"})
        for row in generated:
            if row["pair_split"] == "train" and row["template_split"] == "train":
                self.assertEqual(row["split"], "train")
            elif row["pair_split"] == "test" and row["template_split"] == "test":
                self.assertEqual(row["split"], "test")
            else:
                self.assertEqual(row["split"], "calibration")


if __name__ == "__main__":
    unittest.main()
