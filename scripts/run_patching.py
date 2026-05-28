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

from medical_axis.io import read_csv, read_json, write_csv
from medical_axis.plots import plot_patching_heatmap
from medical_axis.runtime import (
    ResidualPatchHook,
    capture_hidden_vector,
    choose_device,
    choose_dtype,
    configure_runtime,
    label_logprob_diff,
    load_causal_lm,
    locate_decoder_layers,
    require_torch_transformers,
)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--positions", default="-1,-2")
    parser.add_argument("--layers", default="auto", help="'auto' uses best layers; otherwise comma-separated layer list.")
    parser.add_argument("--max-pairs-per-axis", type=int, default=12)
    return parser.parse_args()


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


def layer_list(raw: str, best_axes: dict[str, object]) -> dict[str, list[int]]:
    if raw.strip().lower() == "auto":
        return {axis_id: [int(meta["best_layer"])] for axis_id, meta in best_axes.items()}
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return {axis_id: layers for axis_id in best_axes}


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
    model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
    decoder_layers = locate_decoder_layers(model)
    prompt_rows = read_csv(args.prompts)
    positions = parse_positions(args.positions)
    layers_by_axis = layer_list(args.layers, best_axes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []

    for axis_id in sorted(best_axes):
        pairs = build_pairs(prompt_rows, axis_id, args.use_split, args.max_pairs_per_axis)
        if not pairs:
            continue
        print(f"Patching {axis_id}: {len(pairs)} pairs")
        for pair_index, (positive, negative) in enumerate(pairs, start=1):
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
            for layer in layers_by_axis[axis_id]:
                for position in positions:
                    replacement = capture_hidden_vector(
                        model,
                        tokenizer,
                        negative["prompt"],
                        layer=layer,
                        device=device,
                        position=position,
                    )
                    handle = decoder_layers[layer].register_forward_hook(ResidualPatchHook(replacement, position=position))
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
                    result_rows.append(
                        {
                            "axis_id": axis_id,
                            "pair_index": pair_index,
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
                    )

    write_csv(
        output_dir / "patching_results.csv",
        result_rows,
        [
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
        ],
    )
    if result_rows:
        plot_patching_heatmap(result_rows, Path(args.figure_dir) / "patching_heatmap.png")
    print(f"Wrote patching outputs to {output_dir}")


if __name__ == "__main__":
    main()
