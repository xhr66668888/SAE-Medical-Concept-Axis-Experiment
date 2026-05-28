#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import read_csv, read_json, write_csv
from medical_axis.plots import plot_steering
from medical_axis.runtime import (
    ResidualSteeringHook,
    choose_device,
    choose_dtype,
    configure_runtime,
    label_logprob_diff,
    load_causal_lm,
    locate_decoder_layers,
    require_torch_transformers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steer residual activations along fitted medical concept axes.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--axis-dir", default="outputs/axis")
    parser.add_argument("--output-dir", default="outputs/steering")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--positions", choices=["all", "last"], default="all")
    parser.add_argument("--use-split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--max-prompts-per-axis", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch, _, _ = require_torch_transformers()
    configure_runtime(torch, threads=args.threads)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    axis_dir = Path(args.axis_dir)
    best = read_json(axis_dir / "best_axes.json")
    arrays = np.load(axis_dir / "axis_vectors.npz")
    model_name = args.model_name or str(best["model_name"])
    model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
    decoder_layers = locate_decoder_layers(model)
    alphas = sorted({float(part.strip()) for part in args.alphas.split(",") if part.strip()})

    prompt_rows = read_csv(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for axis_id, meta in sorted(dict(best["axes"]).items()):
        layer = int(meta["best_layer"])
        direction = torch.tensor(arrays[str(meta["axis_unit_key"])], dtype=torch.float32)
        axis_all_rows = [row for row in prompt_rows if row["axis_id"] == axis_id]
        positive_label = next(row["concept_label"] for row in axis_all_rows if row["side"] == "positive")
        negative_label = next(row["concept_label"] for row in axis_all_rows if row["side"] == "negative")
        rows = list(axis_all_rows)
        if args.use_split != "all":
            rows = [row for row in rows if row["split"] == args.use_split]
        rows = rows[: args.max_prompts_per_axis]
        if not rows:
            continue
        print(f"Steering {axis_id} at layer {layer} on {len(rows)} prompts")
        per_alpha: dict[float, list[float]] = {alpha: [] for alpha in alphas}
        baseline_by_prompt: dict[int, float] = {}
        for prompt_idx, row in enumerate(rows):
            for alpha in alphas:
                handle = decoder_layers[layer].register_forward_hook(
                    ResidualSteeringHook(direction, alpha, positions=args.positions)
                )
                try:
                    score = label_logprob_diff(
                        model,
                        tokenizer,
                        row["prompt"],
                        positive_label,
                        negative_label,
                        device=device,
                    )
                finally:
                    handle.remove()
                if alpha == 0.0:
                    baseline_by_prompt[prompt_idx] = score
                per_alpha[alpha].append(score)
                detailed_rows.append(
                    {
                        "axis_id": axis_id,
                        "layer": layer,
                        "prompt_index": prompt_idx,
                        "pair_id": row["pair_id"],
                        "template_id": row["template_id"],
                        "side": row["side"],
                        "alpha": alpha,
                        "logprob_diff": score,
                        "prompt": row["prompt"],
                    }
                )
        baseline_mean = float(np.mean(per_alpha.get(0.0, [0.0])))
        for alpha in alphas:
            mean_score = float(np.mean(per_alpha[alpha]))
            deltas = [
                value - baseline_by_prompt[idx]
                for idx, value in enumerate(per_alpha[alpha])
                if idx in baseline_by_prompt
            ]
            summary_rows.append(
                {
                    "axis_id": axis_id,
                    "layer": layer,
                    "alpha": alpha,
                    "mean_logprob_diff": mean_score,
                    "delta_logprob_diff": float(np.mean(deltas)) if deltas else mean_score - baseline_mean,
                    "prompts": len(per_alpha[alpha]),
                    "positions": args.positions,
                }
            )

    write_csv(
        output_dir / "steering_prompt_results.csv",
        detailed_rows,
        ["axis_id", "layer", "prompt_index", "pair_id", "template_id", "side", "alpha", "logprob_diff", "prompt"],
    )
    write_csv(
        output_dir / "steering_results.csv",
        summary_rows,
        ["axis_id", "layer", "alpha", "mean_logprob_diff", "delta_logprob_diff", "prompts", "positions"],
    )
    plot_steering(summary_rows, Path(args.figure_dir) / "steering_curves.png")
    print(f"Wrote steering outputs to {output_dir}")


if __name__ == "__main__":
    main()
