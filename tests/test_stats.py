from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from medical_axis.io import append_csv_rows, csv_row_key, existing_csv_keys, read_csv, read_json, write_csv, write_json
from medical_axis.runtime import locate_decoder_layers
from medical_axis.stats import benjamini_hochberg, bootstrap_ci, fit_mean_difference_axis, predict_from_axis


class _Node:
    def __init__(self, **children):
        self.__dict__.update(children)


class StatsTest(unittest.TestCase):
    def test_mean_difference_axis_separates_simple_data(self) -> None:
        activations = np.asarray(
            [
                [2.0, 0.0],
                [2.1, 0.2],
                [-2.0, 0.0],
                [-2.1, -0.2],
                [1.9, 0.1],
                [-1.9, -0.1],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([1, 1, 0, 0, 1, 0])
        train_mask = np.asarray([True, True, True, True, False, False])
        test_mask = ~train_mask
        fit = fit_mean_difference_axis(activations, labels, train_mask, test_mask)
        pred = predict_from_axis(activations, fit.axis_unit, fit.threshold)
        self.assertEqual(fit.train_accuracy, 1.0)
        self.assertEqual(fit.test_accuracy, 1.0)
        self.assertTrue(np.all(pred == labels))

    def test_bootstrap_ci_constant_values(self) -> None:
        mean, low, high = bootstrap_ci([1.0, 1.0, 1.0], trials=50, seed=0)
        self.assertEqual(mean, 1.0)
        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_benjamini_hochberg_monotone_adjustment(self) -> None:
        q_values = benjamini_hochberg([0.01, 0.04, 0.03, float("nan")])
        self.assertAlmostEqual(q_values[0], 0.03)
        self.assertAlmostEqual(q_values[1], 0.04)
        self.assertAlmostEqual(q_values[2], 0.04)
        self.assertTrue(np.isnan(q_values[3]))

    def test_decoder_layer_locator_handles_multimodal_gemma3_path(self) -> None:
        layers = [object(), object()]
        model = _Node(model=_Node(language_model=_Node(layers=layers)))
        self.assertIs(locate_decoder_layers(model), layers)

    def test_atomic_json_and_resumable_csv_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            json_path = f"{tmp}/summary.json"
            write_json(json_path, {"b": 2, "a": 1})
            self.assertEqual(read_json(json_path), {"a": 1, "b": 2})

            csv_path = f"{tmp}/results.csv"
            fieldnames = ["axis_id", "layer", "position", "value"]
            append_csv_rows(csv_path, [{"axis_id": "dx", "layer": 3, "position": -1, "value": 0.5}], fieldnames)
            append_csv_rows(csv_path, [{"axis_id": "dx", "layer": 3, "position": -2, "value": 0.25}], fieldnames)
            keys = existing_csv_keys(csv_path, ["axis_id", "layer", "position"])
            self.assertIn(csv_row_key({"axis_id": "dx", "layer": 3, "position": -1}, ["axis_id", "layer", "position"]), keys)
            self.assertEqual(len(read_csv(csv_path)), 2)

    def test_append_csv_extends_existing_header_for_resume_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/results.csv"
            write_csv(csv_path, [{"axis_id": "dx"}], ["axis_id"])
            append_csv_rows(csv_path, [{"axis_id": "dx", "positions": "prompt_all"}], ["axis_id", "positions"])
            rows = read_csv(csv_path)
            self.assertEqual(rows[0]["positions"], "")
            self.assertEqual(rows[1]["positions"], "prompt_all")

    def test_activation_cache_tracks_layer_row_completion(self) -> None:
        from scripts.fit_axes import ActivationCache

        rows = [
            {
                "axis_id": "dx",
                "pair_id": "p0",
                "template_id": "t0",
                "split": "train",
                "side": "positive",
                "prompt": "Example prompt.",
            }
        ]
        with TemporaryDirectory() as tmp:
            cache = ActivationCache(tmp, rows=rows, layers=[0, 2], d_model=3, model_name="model")
            self.assertEqual(cache.done.shape, (2, 1))
            self.assertEqual(cache.missing_layers(0), [0, 2])
            cache.write_layers(0, {0: np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})
            self.assertTrue(cache.is_done(0, 0))
            self.assertEqual(cache.missing_layers(0), [2])

            reopened = ActivationCache(tmp, rows=rows, layers=[0, 2], d_model=3, model_name="model")
            self.assertTrue(reopened.is_done(0, 0))
            self.assertEqual(reopened.layer_values(0)[0].tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
