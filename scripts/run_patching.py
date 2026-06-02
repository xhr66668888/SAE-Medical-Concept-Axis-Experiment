#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import append_csv_rows, csv_row_key, existing_csv_keys, read_csv, read_json, write_csv
from medical_axis.plots import plot_patching_heatmap
from medical_axis.runtime import (
    ResidualPatchHook,
    capture_layer_position_vectors,
    choose_device,
    choose_dtype,
    configure_runtime,
    label_logprob_diff,
    load_causal_lm,
    locate_decoder_layers,
    prompt_token_length,
    require_torch_transformers,
)
from medical_axis.stats import bootstrap_ci

FIXED_MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_POSITIONS = "-1,-2,-3,-4"
RESULT_FIELDNAMES = [
    "axis_id",
    "pair_index",
    "pair_id",
    "layer",
    "position",
    "clean_logprob_diff",
    "corrupt_logprob_diff",
    "patched_logprob_diff",
    "normalized_score",
    "positive_prompt",
    "negative_prompt",
]
RESULT_KEY_FIELDS = ["axis_id", "pair_index", "pair_id", "layer", "position"]
SUMMARY_FIELDNAMES = [
    "axis_id",
    "layer",
    "position",
    "count",
    "mean_normalized_score",
    "median_normalized_score",
    "std_normalized_score",
    "ci_low",
    "ci_high",
]


def normalize_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--positions":
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                normalized.append(f"--positions={argv[idx + 1]}")
                idx += 2
            else:
                normalized.append(f"--positions={DEFAULT_POSITIONS}")
                idx += 1
        else:
            normalized.append(token)
            idx += 1
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch matched negative-side residuals into positive-side prompts.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--axis-dir", default="outputs/axis")
    parser.add_argument("--output-dir", default="outputs/patching")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--use-split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS)
    parser.add_argument(
        "--layers",
        default="window:2",
        help="'auto' uses best layers; 'window:N' uses best layer +/- N; 'all' uses all layers; otherwise comma-separated layers.",
    )
    parser.add_argument("--max-pairs-per-axis", type=int, default=60)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))


def parse_positions(raw: str) -> list[int]:
    positions = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not positions:
        raise SystemExit("--positions must contain at least one integer.")
    return positions


def build_pairs(rows: list[dict[str, str]], axis_id: str, use_split: str, max_pairs: int) -> list[tuple[dict[str, str], dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["axis_id"] != axis_id:
            continue
        if use_split != "all" and row["split"] != use_split:
            continue
        grouped[row["pair_id"]][row["side"]] = row
    pairs = []
    for pair_id in sorted(grouped):
        group = grouped[pair_id]
        if "positive" in group and "negative" in group:
            pairs.append((group["positive"], group["negative"]))
        if len(pairs) >= max_pairs:
            break
    return pairs


def layer_list(raw: str, best_axes: dict[str, object], *, n_layers: int) -> dict[str, list[int]]:
    normalized = raw.strip().lower()
    if normalized == "auto":
        return {axis_id: [int(meta["best_layer"])] for axis_id, meta in best_axes.items()}
    if normalized == "all":
        return {axis_id: list(range(n_layers)) for axis_id in best_axes}
    if normalized.startswith("window:"):
        width = int(normalized.split(":", 1)[1])
        return {
            axis_id: list(range(max(0, int(meta["best_layer"]) - width), min(n_layers, int(meta["best_layer"]) + width + 1)))
            for axis_id, meta in best_axes.items()
        }
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return {axis_id: layers for axis_id in best_axes}


def summarize_rows(rows: list[dict[str, object]], *, bootstrap_trials: int, seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("normalized_score")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        grouped[(str(row["axis_id"]), int(row["layer"]), int(row["position"]))].append(numeric)
    summary = []
    for idx, ((axis_id, layer, position), values) in enumerate(sorted(grouped.items())):
        arr = np.asarray(values, dtype=float)
        mean, ci_low, ci_high = bootstrap_ci(values, trials=bootstrap_trials, seed=seed + idx)
        summary.append(
            {
                "axis_id": axis_id,
                "layer": layer,
                "position": position,
                "count": len(values),
                "mean_normalized_score": mean,
                "median_normalized_score": float(np.median(arr)),
                "std_normalized_score": float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    torch, _, _ = require_torch_transformers()
    configure_runtime(torch, threads=args.threads)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    axis_dir = Path(args.axis_dir)
    best = read_json(axis_dir / "best_axes.json")
    best_axes = dict(best["axes"])
    model_name = args.model_name or str(best["model_name"])
    if model_name != FIXED_MODEL_NAME:
        raise SystemExit(f"This experiment is configured for {FIXED_MODEL_NAME}.")
    prompt_rows = read_csv(args.prompts)
    positions = parse_positions(args.positions)
    if "n_layers" in best:
        n_layers_hint = int(best["n_layers"])
    else:
        n_layers_hint = max(int(meta["best_layer"]) for meta in best_axes.values()) + 1
    layers_by_axis = layer_list(args.layers, best_axes, n_layers=n_layers_hint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "patching_results.csv"
    completed_keys = existing_csv_keys(result_path, RESULT_KEY_FIELDS)
    task_keys: set[tuple[str, ...]] = set()
    pending_by_axis: dict[str, list[dict[str, object]]] = defaultdict(list)

    for axis_id in sorted(best_axes):
        pairs = build_pairs(prompt_rows, axis_id, args.use_split, args.max_pairs_per_axis)
        if not pairs:
            continue
        for pair_index, (positive, negative) in enumerate(pairs, start=1):
            for layer in layers_by_axis[axis_id]:
                for position in positions:
                    task = {
                        "axis_id": axis_id,
                        "pair_index": pair_index,
                        "pair_id": positive["pair_id"],
                        "layer": layer,
                        "position": position,
                        "positive": positive,
                        "negative": negative,
                    }
                    key = csv_row_key(task, RESULT_KEY_FIELDS)
                    task_keys.add(key)
                    if key not in completed_keys:
                        pending_by_axis[axis_id].append(task)

    pending_count = sum(len(tasks) for tasks in pending_by_axis.values())
    if pending_count:
        model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
        decoder_layers = locate_decoder_layers(model)
        for axis_id, tasks in sorted(pending_by_axis.items()):
            grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
            for task in tasks:
                grouped[(int(task["pair_index"]), str(task["pair_id"]))].append(task)
            print(f"Patching {axis_id}: {len(tasks)} missing rows across {len(grouped)} pairs")
            for (_pair_index, _pair_id), pair_tasks in sorted(grouped.items()):
                positive = pair_tasks[0]["positive"]
                negative = pair_tasks[0]["negative"]
                assert isinstance(positive, dict)
                assert isinstance(negative, dict)
                positive_label = positive["concept_label"]
                negative_label = negative["concept_label"]
                clean_score = label_logprob_diff(
                    model,
                    tokenizer,
                    positive["prompt"],
                    positive_label,
                    negative_label,
                    device=device,
                )
                corrupt_score = label_logprob_diff(
                    model,
                    tokenizer,
                    negative["prompt"],
                    positive_label,
                    negative_label,
                    device=device,
                )
                denominator = corrupt_score - clean_score
                positive_prompt_length = prompt_token_length(tokenizer, positive["prompt"])
                missing_layers = sorted({int(task["layer"]) for task in pair_tasks})
                missing_positions = sorted({int(task["position"]) for task in pair_tasks})
                replacements = capture_layer_position_vectors(
                    model,
                    tokenizer,
                    negative["prompt"],
                    layers=missing_layers,
                    positions=missing_positions,
                    device=device,
                )
                for task in pair_tasks:
                    layer = int(task["layer"])
                    position = int(task["position"])
                    replacement = replacements[(layer, position)]
                    handle = decoder_layers[layer].register_forward_hook(
                        ResidualPatchHook(replacement, position=position, prompt_length=positive_prompt_length)
                    )
                    try:
                        patched_score = label_logprob_diff(
                            model,
                            tokenizer,
                            positive["prompt"],
                            positive_label,
                            negative_label,
                            device=device,
                        )
                    finally:
                        handle.remove()
                    normalized = (patched_score - clean_score) / denominator if abs(denominator) > 1e-8 else math.nan
                    output_row = {
                        "axis_id": axis_id,
                        "pair_index": task["pair_index"],
                        "pair_id": positive["pair_id"],
                        "layer": layer,
                        "position": position,
                        "clean_logprob_diff": clean_score,
                        "corrupt_logprob_diff": corrupt_score,
                        "patched_logprob_diff": patched_score,
                        "normalized_score": normalized,
                        "positive_prompt": positive["prompt"],
                        "negative_prompt": negative["prompt"],
                    }
                    append_csv_rows(result_path, [output_row], RESULT_FIELDNAMES)
                    completed_keys.add(csv_row_key(output_row, RESULT_KEY_FIELDS))
    else:
        print("Patching results already complete; rebuilding summary from CSV.")

    all_result_rows = read_csv(result_path) if result_path.exists() else []
    result_by_key = {
        csv_row_key(row, RESULT_KEY_FIELDS): row
        for row in all_result_rows
        if csv_row_key(row, RESULT_KEY_FIELDS) in task_keys
    }
    result_rows = list(result_by_key.values())

    write_csv(
        result_path,
        result_rows,
        RESULT_FIELDNAMES,
    )
    summary_rows = summarize_rows(result_rows, bootstrap_trials=args.bootstrap_trials, seed=args.seed)
    write_csv(
        output_dir / "patching_summary.csv",
        summary_rows,
        SUMMARY_FIELDNAMES,
    )
    if result_rows:
        plot_patching_heatmap(result_rows, Path(args.figure_dir) / "patching_heatmap.png")
    print(f"Wrote patching outputs to {output_dir}")


if __name__ == "__main__":
    main()
