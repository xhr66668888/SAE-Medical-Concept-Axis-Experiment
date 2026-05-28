from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisFit:
    axis: np.ndarray
    axis_unit: np.ndarray
    threshold: float
    positive_mean: np.ndarray
    negative_mean: np.ndarray
    train_accuracy: float
    test_accuracy: float
    train_projection: np.ndarray
    test_projection: np.ndarray


def unit_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def fit_mean_difference_axis(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> AxisFit:
    positive_train = activations[train_mask & (labels == 1)]
    negative_train = activations[train_mask & (labels == 0)]
    if len(positive_train) == 0 or len(negative_train) == 0:
        raise ValueError("Training split must contain both positive and negative examples.")

    positive_mean = positive_train.mean(axis=0)
    negative_mean = negative_train.mean(axis=0)
    axis = positive_mean - negative_mean
    axis_unit = unit_vector(axis)

    projections = activations @ axis_unit
    pos_center = projections[train_mask & (labels == 1)].mean()
    neg_center = projections[train_mask & (labels == 0)].mean()
    threshold = float((pos_center + neg_center) / 2.0)

    predictions = (projections >= threshold).astype(int)
    train_accuracy = float((predictions[train_mask] == labels[train_mask]).mean())
    test_accuracy = float((predictions[test_mask] == labels[test_mask]).mean()) if test_mask.any() else math.nan
    return AxisFit(
        axis=axis.astype(np.float32),
        axis_unit=axis_unit,
        threshold=threshold,
        positive_mean=positive_mean.astype(np.float32),
        negative_mean=negative_mean.astype(np.float32),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        train_projection=projections[train_mask].astype(np.float32),
        test_projection=projections[test_mask].astype(np.float32),
    )


def predict_from_axis(activations: np.ndarray, axis_unit: np.ndarray, threshold: float) -> np.ndarray:
    return (activations @ axis_unit >= threshold).astype(int)


def bootstrap_ci(values: list[float] | np.ndarray, *, trials: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if trials <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(trials, dtype=float)
    for idx in range(trials):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = sample.mean()
    return mean, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def permutation_null_accuracy(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = np.empty(trials, dtype=float)
    for idx in range(trials):
        shuffled = labels.copy()
        shuffled[train_mask] = rng.permutation(shuffled[train_mask])
        fit = fit_mean_difference_axis(activations, shuffled, train_mask, test_mask)
        pred = predict_from_axis(activations[test_mask], fit.axis_unit, fit.threshold)
        scores[idx] = (pred == labels[test_mask]).mean() if test_mask.any() else math.nan
    return scores


def random_direction_null_accuracy(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d_model = activations.shape[1]
    scores = np.empty(trials, dtype=float)
    for idx in range(trials):
        direction = rng.normal(size=d_model).astype(np.float32)
        direction = unit_vector(direction)
        projections = activations @ direction
        pos_center = projections[train_mask & (labels == 1)].mean()
        neg_center = projections[train_mask & (labels == 0)].mean()
        threshold = float((pos_center + neg_center) / 2.0)
        sign = 1.0 if pos_center >= neg_center else -1.0
        predictions = (sign * (projections - threshold) >= 0).astype(int)
        scores[idx] = (predictions[test_mask] == labels[test_mask]).mean() if test_mask.any() else math.nan
    return scores


def cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    keys = sorted(vectors)
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    units = {key: unit_vector(value) for key, value in vectors.items()}
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            matrix[i, j] = float(np.dot(units[left], units[right]))
    return keys, matrix
