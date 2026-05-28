from __future__ import annotations

import unittest

import numpy as np

from medical_axis.stats import bootstrap_ci, fit_mean_difference_axis, predict_from_axis


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


if __name__ == "__main__":
    unittest.main()
