#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import read_csv, write_csv, write_json
from medical_axis.plots import plot_accuracy_by_layer, plot_cosine_heatmap
from medical_axis.runtime import (
    capture_layer_matrix,
    choose_device,
    choose_dtype,
    configure_runtime,
    label_logprob_diff,
    load_causal_lm,
    locate_decoder_layers,
    require_torch_transformers,
)
from medical_axis.stats import (
    bootstrap_ci,
    cosine_matrix,
    fit_mean_difference_axis,
    permutation_null_accuracy,
    predict_from_axis,
    random_direction_null_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit residual-stream directions for medical concept contrasts.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/axis")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default="google/gemma-3-1b-it")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-prompts-per-axis", type=int, default=None)
    return parser.parse_args()


def parse_layers(raw: str, n_layers: int) -> list[int]:
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return list(range(n_layers))
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise SystemExit(f"Layer(s) out of range for n_layers={n_layers}: {bad}")
    if not layers:
        raise SystemExit("--layers resolved to an empty list.")
    return layers


def balanced_limit(rows: list[dict[str, str]], max_per_axis: int | None) -> list[dict[str, str]]:
    if max_per_axis is None:
        return rows
    kept: list[dict[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["axis_id"], row["side"])
        if counts.get(key, 0) >= max_per_axis:
            continue
        kept.append(row)
        counts[key] = counts.get(key, 0) + 1
    return kept


def main() -> None:
    args = parse_args()
    torch, _, _ = require_torch_transformers()
    configure_runtime(torch, threads=args.threads)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    rows = balanced_limit(read_csv(args.prompts), args.max_prompts_per_axis)
    if not rows:
        raise SystemExit("No prompt rows found.")

    print("Loading model")
    print(f"  model : {args.model_name}")
    print(f"  device: {device}")
    print(f"  dtype : {dtype}")
    model, tokenizer = load_causal_lm(args.model_name, device=device, dtype=dtype)
    n_layers = len(locate_decoder_layers(model))
    layers = parse_layers(args.layers, n_layers)
    print(f"  layers: {layers}")

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    activations: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    readout_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        if idx <= 3 or idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] {row['axis_id']} {row['side']} template={row['template_id']}")
        hidden = capture_layer_matrix(model, tokenizer, row["prompt"], layers=layers, device=device)
        for layer in layers:
            activations[layer].append(hidden[layer].numpy())
        readout = label_logprob_diff(
            model,
            tokenizer,
            row["prompt"],
            row["concept_label"],
            row["opposite_label"],
            device=device,
        )
        readout_rows.append(
            {
                "axis_id": row["axis_id"],
                "side": row["side"],
                "split": row["split"],
                "pair_id": row["pair_id"],
                "template_id": row["template_id"],
                "label_logprob_diff": readout,
                "prompt": row["prompt"],
            }
        )

    layer_arrays = {layer: np.stack(values, axis=0).astype(np.float32) for layer, values in activations.items()}
    all_axis_ids = sorted({row["axis_id"] for row in rows})
    layer_sweep_rows: list[dict[str, object]] = []
    axis_summary_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    best_axis_units: dict[str, np.ndarray] = {}
    best_axes: dict[str, object] = {}

    for axis_id in all_axis_ids:
        indices = [idx for idx, row in enumerate(rows) if row["axis_id"] == axis_id]
        axis_rows = [rows[idx] for idx in indices]
        labels = np.asarray([1 if row["side"] == "positive" else 0 for row in axis_rows], dtype=int)
        train_mask = np.asarray([row["split"] == "train" for row in axis_rows], dtype=bool)
        test_mask = np.asarray([row["split"] == "test" for row in axis_rows], dtype=bool)
        if set(labels[train_mask].tolist()) != {0, 1}:
            raise SystemExit(f"{axis_id}: train split must contain both sides.")
        if test_mask.any() and set(labels[test_mask].tolist()) != {0, 1}:
            raise SystemExit(f"{axis_id}: test split must contain both sides.")

        axis_layer_rows = []
        fits = {}
        for layer in layers:
            acts = layer_arrays[layer][indices]
            fit = fit_mean_difference_axis(acts, labels, train_mask, test_mask)
            fits[layer] = fit
            pred_test = predict_from_axis(acts[test_mask], fit.axis_unit, fit.threshold)
            test_correct = (pred_test == labels[test_mask]).astype(float) if test_mask.any() else np.asarray([])
            _, ci_low, ci_high = bootstrap_ci(test_correct, trials=args.bootstrap_trials, seed=args.seed + layer)
            rand_null = random_direction_null_accuracy(
                acts,
                labels,
                train_mask,
                test_mask,
                trials=args.null_trials,
                seed=args.seed + layer,
            )
            perm_null = permutation_null_accuracy(
                acts,
                labels,
                train_mask,
                test_mask,
                trials=args.null_trials,
                seed=args.seed + 10_000 + layer,
            )
            random_mean = float(np.nanmean(rand_null))
            permutation_p = float(np.nanmean(perm_null >= fit.test_accuracy)) if not math.isnan(fit.test_accuracy) else math.nan
            row = {
                "axis_id": axis_id,
                "layer": layer,
                "train_accuracy": fit.train_accuracy,
                "test_accuracy": fit.test_accuracy,
                "test_ci_low": ci_low,
                "test_ci_high": ci_high,
                "random_null_mean": random_mean,
                "permutation_null_mean": float(np.nanmean(perm_null)),
                "permutation_p": permutation_p,
                "axis_norm": float(np.linalg.norm(fit.axis)),
                "score": fit.test_accuracy - random_mean,
            }
            axis_layer_rows.append(row)
            layer_sweep_rows.append(row)
            saved_arrays[f"axis__{axis_id}__layer_{layer}"] = fit.axis
            saved_arrays[f"axis_unit__{axis_id}__layer_{layer}"] = fit.axis_unit
            saved_arrays[f"positive_mean__{axis_id}__layer_{layer}"] = fit.positive_mean
            saved_arrays[f"negative_mean__{axis_id}__layer_{layer}"] = fit.negative_mean

        best_row = max(axis_layer_rows, key=lambda item: (float(item["score"]), float(item["test_accuracy"])))
        best_layer = int(best_row["layer"])
        best_fit = fits[best_layer]
        best_axis_units[axis_id] = best_fit.axis_unit
        axis_summary_rows.append(
            {
                "axis_id": axis_id,
                "best_layer": best_layer,
                "train_accuracy": best_row["train_accuracy"],
                "test_accuracy": best_row["test_accuracy"],
                "test_ci_low": best_row["test_ci_low"],
                "test_ci_high": best_row["test_ci_high"],
                "random_null_mean": best_row["random_null_mean"],
                "permutation_null_mean": best_row["permutation_null_mean"],
                "permutation_p": best_row["permutation_p"],
                "axis_norm": best_row["axis_norm"],
            }
        )
        best_axes[axis_id] = {
            "best_layer": best_layer,
            "model_name": args.model_name,
            "threshold": best_fit.threshold,
            "axis_key": f"axis__{axis_id}__layer_{best_layer}",
            "axis_unit_key": f"axis_unit__{axis_id}__layer_{best_layer}",
        }

        acts = layer_arrays[best_layer][indices]
        projections = acts @ best_fit.axis_unit
        predictions = (projections >= best_fit.threshold).astype(int)
        for local_idx, row in enumerate(axis_rows):
            projection_rows.append(
                {
                    "axis_id": axis_id,
                    "pair_id": row["pair_id"],
                    "template_id": row["template_id"],
                    "split": row["split"],
                    "side": row["side"],
                    "projection": float(projections[local_idx]),
                    "predicted_side": "positive" if predictions[local_idx] == 1 else "negative",
                    "correct": int(predictions[local_idx] == labels[local_idx]),
                    "prompt": row["prompt"],
                }
            )

    write_csv(
        output_dir / "layer_sweep.csv",
        layer_sweep_rows,
        [
            "axis_id",
            "layer",
            "train_accuracy",
            "test_accuracy",
            "test_ci_low",
            "test_ci_high",
            "random_null_mean",
            "permutation_null_mean",
            "permutation_p",
            "axis_norm",
            "score",
        ],
    )
    write_csv(
        output_dir / "axis_summary.csv",
        axis_summary_rows,
        [
            "axis_id",
            "best_layer",
            "train_accuracy",
            "test_accuracy",
            "test_ci_low",
            "test_ci_high",
            "random_null_mean",
            "permutation_null_mean",
            "permutation_p",
            "axis_norm",
        ],
    )
    write_csv(
        output_dir / "axis_projections.csv",
        projection_rows,
        ["axis_id", "pair_id", "template_id", "split", "side", "projection", "predicted_side", "correct", "prompt"],
    )
    write_csv(
        output_dir.parent / "readout_baseline.csv",
        readout_rows,
        ["axis_id", "side", "split", "pair_id", "template_id", "label_logprob_diff", "prompt"],
    )
    np.savez_compressed(output_dir / "axis_vectors.npz", **saved_arrays)
    write_json(output_dir / "best_axes.json", {"model_name": args.model_name, "n_layers": n_layers, "axes": best_axes})
    labels_for_heatmap, cosine = cosine_matrix(best_axis_units)
    write_csv(
        output_dir / "axis_cosine.csv",
        [
            {"axis_id": left, **{right: float(cosine[i, j]) for j, right in enumerate(labels_for_heatmap)}}
            for i, left in enumerate(labels_for_heatmap)
        ],
        ["axis_id", *labels_for_heatmap],
    )
    plot_accuracy_by_layer(layer_sweep_rows, figure_dir / "accuracy_by_layer.png")
    plot_cosine_heatmap(labels_for_heatmap, cosine, figure_dir / "axis_cosine_heatmap.png")
    print(f"Wrote axis outputs to {output_dir}")


if __name__ == "__main__":
    main()
