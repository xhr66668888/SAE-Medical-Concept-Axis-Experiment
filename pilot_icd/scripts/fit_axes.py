#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import append_csv_rows, atomic_output_path, csv_row_key, existing_csv_keys, read_csv, read_json, write_csv, write_json
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
    benjamini_hochberg,
    bootstrap_ci,
    cosine_matrix,
    fit_mean_difference_axis,
    permutation_null_accuracy,
    predict_from_axis,
    random_direction_null_accuracy,
)

FIXED_MODEL_NAME = "google/gemma-3-4b-it"
READOUT_FIELDNAMES = ["axis_id", "side", "split", "pair_id", "template_id", "label_logprob_diff", "prompt"]
READOUT_KEY_FIELDS = ["axis_id", "side", "split", "pair_id", "template_id", "prompt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit residual-stream directions for medical concept contrasts.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/axis")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default=FIXED_MODEL_NAME)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-prompts-per-axis", type=int, default=None)
    parser.add_argument("--fit-splits", default="train,test")
    parser.add_argument(
        "--activation-cache-dir",
        default=None,
        help="Directory for resumable float32 residual activation memmaps. Defaults to OUTPUT_DIR/activation_cache.",
    )
    parser.add_argument("--activation-cache", dest="activation_cache_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--refresh-activation-cache", action="store_true", help="Overwrite any existing activation cache.")
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


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def activation_row_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            row.get("axis_id", ""),
            row.get("pair_id", ""),
            row.get("template_id", ""),
            row.get("split", ""),
            row.get("side", ""),
            prompt_hash(row.get("prompt", "")),
        ]
    )


def infer_hidden_size(model: Any) -> int | None:
    candidates = []
    config = getattr(model, "config", None)
    if config is not None:
        candidates.append(config)
        for name in ("text_config", "language_config"):
            nested = getattr(config, name, None)
            if nested is not None:
                candidates.append(nested)
    for obj in candidates:
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    return None


def as_float32_vector(vector: Any, *, d_model: int) -> np.ndarray:
    if hasattr(vector, "detach"):
        arr = vector.detach().float().cpu().numpy()
    else:
        arr = np.asarray(vector, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape != (d_model,):
        raise ValueError(f"Activation vector shape {arr.shape} does not match expected {(d_model,)}.")
    return arr


class ActivationCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        rows: list[dict[str, str]],
        layers: list[int],
        d_model: int,
        model_name: str,
        refresh: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.layer_to_index = {layer: idx for idx, layer in enumerate(self.layers)}
        self.n_rows = len(rows)
        self.d_model = int(d_model)
        self.done_path = self.cache_dir / "done.npy"
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = {
            "version": 1,
            "model_name": model_name,
            "dtype": "float32",
            "n_rows": self.n_rows,
            "d_model": self.d_model,
            "layers": self.layers,
            "row_keys": [activation_row_key(row) for row in rows],
        }
        if refresh or not self.manifest_path.exists():
            write_json(self.manifest_path, self.manifest)
        else:
            existing = read_json(self.manifest_path)
            expected = {key: self.manifest[key] for key in ("model_name", "dtype", "n_rows", "d_model", "layers", "row_keys")}
            actual = {key: existing.get(key) for key in expected}
            if actual != expected:
                raise SystemExit(
                    f"Activation cache manifest does not match this run: {self.manifest_path}. "
                    "Use --activation-cache-dir for a separate cache or --refresh-activation-cache to overwrite it."
                )

        if refresh or not self.done_path.exists():
            self.done = np.zeros((len(self.layers), self.n_rows), dtype=bool)
            self._write_done()
        else:
            self.done = np.load(self.done_path).astype(bool, copy=False)
            if self.done.shape != (len(self.layers), self.n_rows):
                raise SystemExit(f"Activation cache done mask has shape {self.done.shape}, expected {(len(self.layers), self.n_rows)}.")

        self.maps: dict[int, np.memmap] = {}
        expected_bytes = self.n_rows * self.d_model * np.dtype(np.float32).itemsize
        done_changed = False
        for layer in self.layers:
            path = self.layer_path(layer)
            if path.exists() and not refresh and path.stat().st_size != expected_bytes:
                raise SystemExit(f"Activation cache file has unexpected size: {path}")
            if not path.exists() and not refresh:
                self.done[self.layer_to_index[layer], :] = False
                done_changed = True
            mode = "w+" if refresh or not path.exists() else "r+"
            self.maps[layer] = np.memmap(path, dtype=np.float32, mode=mode, shape=(self.n_rows, self.d_model))
        if done_changed:
            self._write_done()

    def layer_path(self, layer: int) -> Path:
        return self.cache_dir / f"layer_{layer}.float32.mmap"

    def _write_done(self) -> None:
        with atomic_output_path(self.done_path) as tmp_path:
            with tmp_path.open("wb") as handle:
                np.save(handle, self.done)

    def is_done(self, row_index: int, layer: int) -> bool:
        return bool(self.done[self.layer_to_index[layer], row_index])

    def missing_layers(self, row_index: int) -> list[int]:
        return [layer for layer in self.layers if not self.is_done(row_index, layer)]

    def write_layers(self, row_index: int, values: dict[int, Any]) -> None:
        changed = False
        for layer, vector in values.items():
            if layer not in self.layer_to_index or self.is_done(row_index, layer):
                continue
            self.maps[layer][row_index, :] = as_float32_vector(vector, d_model=self.d_model)
            self.maps[layer].flush()
            self.done[self.layer_to_index[layer], row_index] = True
            changed = True
        if changed:
            self._write_done()

    def layer_values(self, layer: int) -> np.memmap:
        return self.maps[layer]


def main() -> None:
    args = parse_args()
    if args.model_name != FIXED_MODEL_NAME:
        raise SystemExit(f"This experiment is configured for {FIXED_MODEL_NAME}.")
    torch, _, _ = require_torch_transformers()
    configure_runtime(torch, threads=args.threads)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    allowed_splits = {part.strip() for part in args.fit_splits.split(",") if part.strip()}
    rows = [row for row in read_csv(args.prompts) if row.get("split", "train") in allowed_splits]
    rows = balanced_limit(rows, args.max_prompts_per_axis)
    if not rows:
        raise SystemExit("No prompt rows found.")
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    print("Loading model")
    print(f"  model : {args.model_name}")
    print(f"  device: {device}")
    print(f"  dtype : {dtype}")
    model, tokenizer = load_causal_lm(args.model_name, device=device, dtype=dtype)
    n_layers = len(locate_decoder_layers(model))
    layers = parse_layers(args.layers, n_layers)
    print(f"  layers: {layers}")
    print(f"  splits: {split_counts}")

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    d_model = infer_hidden_size(model)
    if d_model is None:
        print("Inferring hidden size from first prompt")
        first_hidden = capture_layer_matrix(model, tokenizer, rows[0]["prompt"], layers=[layers[0]], device=device)
        d_model = int(first_hidden[layers[0]].numel())
    cache_dir = Path(args.activation_cache_dir) if args.activation_cache_dir else output_dir / "activation_cache"
    activation_cache = ActivationCache(
        cache_dir,
        rows=rows,
        layers=layers,
        d_model=d_model,
        model_name=args.model_name,
        refresh=args.refresh_activation_cache,
    )
    readout_path = output_dir.parent / "readout_baseline.csv"
    readout_keys = existing_csv_keys(readout_path, READOUT_KEY_FIELDS)

    for idx, row in enumerate(rows, start=1):
        row_index = idx - 1
        missing_layers = activation_cache.missing_layers(row_index)
        readout_key = csv_row_key(row, READOUT_KEY_FIELDS)
        needs_readout = readout_key not in readout_keys
        if idx <= 3 or idx % 25 == 0 or idx == len(rows) or missing_layers or needs_readout:
            status = "cached" if not missing_layers and not needs_readout else "compute"
            print(f"[{idx}/{len(rows)}] {status} {row['axis_id']} {row['side']} template={row['template_id']}")
        if missing_layers:
            hidden = capture_layer_matrix(model, tokenizer, row["prompt"], layers=missing_layers, device=device)
            activation_cache.write_layers(row_index, hidden)
        if needs_readout:
            readout = label_logprob_diff(
                model,
                tokenizer,
                row["prompt"],
                row["concept_label"],
                row["opposite_label"],
                device=device,
            )
            append_csv_rows(
                readout_path,
                [
                    {
                        "axis_id": row["axis_id"],
                        "side": row["side"],
                        "split": row["split"],
                        "pair_id": row["pair_id"],
                        "template_id": row["template_id"],
                        "label_logprob_diff": readout,
                        "prompt": row["prompt"],
                    }
                ],
                READOUT_FIELDNAMES,
            )
            readout_keys.add(readout_key)

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
            acts = np.asarray(activation_cache.layer_values(layer)[indices], dtype=np.float32)
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

        acts = np.asarray(activation_cache.layer_values(best_layer)[indices], dtype=np.float32)
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

    q_values = benjamini_hochberg([float(row["permutation_p"]) for row in layer_sweep_rows])
    for row, q_value in zip(layer_sweep_rows, q_values):
        row["permutation_q_bh"] = float(q_value)
    q_by_axis_layer = {
        (str(row["axis_id"]), int(row["layer"])): float(row["permutation_q_bh"])
        for row in layer_sweep_rows
    }
    for row in axis_summary_rows:
        row["permutation_q_bh"] = q_by_axis_layer[(str(row["axis_id"]), int(row["best_layer"]))]

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
            "permutation_q_bh",
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
            "permutation_q_bh",
            "axis_norm",
        ],
    )
    write_csv(
        output_dir / "axis_projections.csv",
        projection_rows,
        ["axis_id", "pair_id", "template_id", "split", "side", "projection", "predicted_side", "correct", "prompt"],
    )
    current_readout_keys = {csv_row_key(row, READOUT_KEY_FIELDS) for row in rows}
    current_readout_rows = [
        row for row in read_csv(readout_path) if csv_row_key(row, READOUT_KEY_FIELDS) in current_readout_keys
    ]
    write_csv(
        readout_path,
        current_readout_rows,
        READOUT_FIELDNAMES,
    )
    with atomic_output_path(output_dir / "axis_vectors.npz") as tmp_path:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **saved_arrays)
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
