#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import append_csv_rows, csv_row_key, existing_csv_keys, read_csv, read_json, write_csv
from medical_axis.plots import plot_steering
from medical_axis.runtime import (
    ResidualSteeringHook,
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
DEFAULT_ALPHAS = "-6,-4,-2,-1,0,1,2,4,6"
DETAIL_FIELDNAMES = [
    "axis_id",
    "layer",
    "prompt_index",
    "pair_id",
    "template_id",
    "split",
    "side",
    "alpha",
    "positions",
    "logprob_diff",
    "prompt",
]
DETAIL_KEY_FIELDS = ["axis_id", "layer", "prompt_index", "pair_id", "template_id", "split", "side", "alpha", "positions"]
SUMMARY_FIELDNAMES = [
    "axis_id",
    "layer",
    "alpha",
    "mean_logprob_diff",
    "delta_logprob_diff",
    "delta_ci_low",
    "delta_ci_high",
    "prompts",
    "positions",
]
DOSE_FIELDNAMES = [
    "axis_id",
    "layer",
    "mean_prompt_slope",
    "slope_ci_low",
    "slope_ci_high",
    "positive_slope_fraction",
    "prompts",
    "positions",
]


def summarize_detailed_rows(
    rows: list[dict[str, object]],
    *,
    bootstrap_trials: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, object]]] = defaultdict(list)
    baseline_by_prompt: dict[tuple[str, int, str, str], float] = {}
    scores_by_prompt: dict[tuple[str, int, str, str], dict[float, float]] = defaultdict(dict)
    for row in rows:
        axis_id = str(row["axis_id"])
        layer = int(row["layer"])
        alpha = float(row["alpha"])
        positions = str(row.get("positions", ""))
        prompt_index = str(row["prompt_index"])
        score = float(row["logprob_diff"])
        grouped[(axis_id, layer, alpha, positions)].append(row)
        prompt_key = (axis_id, layer, prompt_index, positions)
        scores_by_prompt[prompt_key][alpha] = score
        if alpha == 0.0:
            baseline_by_prompt[prompt_key] = score

    summary_rows = []
    for idx, ((axis_id, layer, alpha, positions), group) in enumerate(sorted(grouped.items())):
        scores = [float(row["logprob_diff"]) for row in group]
        deltas = []
        for row in group:
            prompt_key = (axis_id, layer, str(row["prompt_index"]), positions)
            if prompt_key in baseline_by_prompt:
                deltas.append(float(row["logprob_diff"]) - baseline_by_prompt[prompt_key])
        mean_score = float(np.mean(scores))
        baseline_scores = [score for key, score in baseline_by_prompt.items() if key[0] == axis_id and key[1] == layer and key[3] == positions]
        baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0
        mean_delta, ci_low, ci_high = bootstrap_ci(
            deltas,
            trials=bootstrap_trials,
            seed=seed + idx,
        )
        summary_rows.append(
            {
                "axis_id": axis_id,
                "layer": layer,
                "alpha": alpha,
                "mean_logprob_diff": mean_score,
                "delta_logprob_diff": mean_delta if deltas else mean_score - baseline_mean,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "prompts": len(scores),
                "positions": positions,
            }
        )

    slopes_by_axis: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (axis_id, layer, _prompt_index, positions), alpha_scores in scores_by_prompt.items():
        if 0.0 not in alpha_scores or len(alpha_scores) < 2:
            continue
        prompt_alphas = sorted(alpha_scores)
        x = np.asarray(prompt_alphas, dtype=float)
        y = np.asarray([alpha_scores[alpha] - alpha_scores[0.0] for alpha in prompt_alphas], dtype=float)
        slopes_by_axis[(axis_id, layer, positions)].append(float(np.polyfit(x, y, deg=1)[0]))

    dose_rows = []
    for idx, ((axis_id, layer, positions), slopes) in enumerate(sorted(slopes_by_axis.items())):
        mean_slope, slope_low, slope_high = bootstrap_ci(
            slopes,
            trials=bootstrap_trials,
            seed=seed + len(summary_rows) + idx,
        )
        dose_rows.append(
            {
                "axis_id": axis_id,
                "layer": layer,
                "mean_prompt_slope": mean_slope,
                "slope_ci_low": slope_low,
                "slope_ci_high": slope_high,
                "positive_slope_fraction": float(np.mean(np.asarray(slopes) > 0)),
                "prompts": len(slopes),
                "positions": positions,
            }
        )
    return summary_rows, dose_rows


def normalize_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--alphas":
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                normalized.append(f"--alphas={argv[idx + 1]}")
                idx += 2
            else:
                normalized.append(f"--alphas={DEFAULT_ALPHAS}")
                idx += 1
        else:
            normalized.append(token)
            idx += 1
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steer residual activations along fitted medical concept axes.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--axis-dir", default="outputs/axis")
    parser.add_argument("--output-dir", default="outputs/steering")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--alphas",
        default=DEFAULT_ALPHAS,
        help="Comma-separated steering coefficients. A bare --alphas uses the default grid.",
    )
    parser.add_argument("--positions", choices=["prompt_all", "prompt_last", "all", "last"], default="prompt_all")
    parser.add_argument("--use-split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--max-prompts-per-axis", type=int, default=120)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))


def parse_alphas(raw: str | None) -> list[float]:
    value = (raw or DEFAULT_ALPHAS).strip()
    if not value:
        value = DEFAULT_ALPHAS
    try:
        alphas = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit(f"--alphas must be a comma-separated list of numbers: {raw!r}") from exc
    if not alphas:
        raise SystemExit("--alphas resolved to an empty list.")
    return alphas


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
    if model_name != FIXED_MODEL_NAME:
        raise SystemExit(f"This experiment is configured for {FIXED_MODEL_NAME}.")
    alphas = parse_alphas(args.alphas)

    prompt_rows = read_csv(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "steering_prompt_results.csv"
    completed_keys = existing_csv_keys(detailed_path, DETAIL_KEY_FIELDS)
    task_keys: set[tuple[str, ...]] = set()
    pending_by_axis: dict[str, list[dict[str, object]]] = {}

    for axis_id, meta in sorted(dict(best["axes"]).items()):
        layer = int(meta["best_layer"])
        axis_all_rows = [row for row in prompt_rows if row["axis_id"] == axis_id]
        if not axis_all_rows:
            continue
        positive_label = next(row["concept_label"] for row in axis_all_rows if row["side"] == "positive")
        negative_label = next(row["concept_label"] for row in axis_all_rows if row["side"] == "negative")
        rows = list(axis_all_rows)
        if args.use_split != "all":
            rows = [row for row in rows if row["split"] == args.use_split]
        rows = rows[: args.max_prompts_per_axis]
        if not rows:
            continue
        for prompt_idx, row in enumerate(rows):
            for alpha in alphas:
                task = {
                    "axis_id": axis_id,
                    "layer": layer,
                    "prompt_index": prompt_idx,
                    "pair_id": row["pair_id"],
                    "template_id": row["template_id"],
                    "split": row["split"],
                    "side": row["side"],
                    "alpha": alpha,
                    "positions": args.positions,
                    "prompt": row["prompt"],
                    "positive_label": positive_label,
                    "negative_label": negative_label,
                    "axis_unit_key": str(meta["axis_unit_key"]),
                }
                key = csv_row_key(task, DETAIL_KEY_FIELDS)
                task_keys.add(key)
                if key not in completed_keys:
                    pending_by_axis.setdefault(axis_id, []).append(task)

    pending_count = sum(len(tasks) for tasks in pending_by_axis.values())
    if pending_count:
        model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
        decoder_layers = locate_decoder_layers(model)
        for axis_id, tasks in sorted(pending_by_axis.items()):
            layer = int(tasks[0]["layer"])
            direction = torch.tensor(arrays[str(tasks[0]["axis_unit_key"])], dtype=torch.float32)
            prompt_lengths: dict[str, int] = {}
            print(f"Steering {axis_id} at layer {layer}: {len(tasks)} missing rows")
            for task in tasks:
                prompt = str(task["prompt"])
                prompt_length = prompt_lengths.get(prompt)
                if prompt_length is None:
                    prompt_length = prompt_token_length(tokenizer, prompt)
                    prompt_lengths[prompt] = prompt_length
                alpha = float(task["alpha"])
                handle = decoder_layers[int(task["layer"])].register_forward_hook(
                    ResidualSteeringHook(direction, alpha, positions=args.positions, prompt_length=prompt_length)
                )
                try:
                    score = label_logprob_diff(
                        model,
                        tokenizer,
                        prompt,
                        str(task["positive_label"]),
                        str(task["negative_label"]),
                        device=device,
                    )
                finally:
                    handle.remove()
                output_row = {
                    "axis_id": task["axis_id"],
                    "layer": task["layer"],
                    "prompt_index": task["prompt_index"],
                    "pair_id": task["pair_id"],
                    "template_id": task["template_id"],
                    "split": task["split"],
                    "side": task["side"],
                    "alpha": alpha,
                    "positions": args.positions,
                    "logprob_diff": score,
                    "prompt": prompt,
                }
                append_csv_rows(detailed_path, [output_row], DETAIL_FIELDNAMES)
                completed_keys.add(csv_row_key(output_row, DETAIL_KEY_FIELDS))
    else:
        print("Steering prompt results already complete; rebuilding summaries from CSV.")

    all_detailed = read_csv(detailed_path) if detailed_path.exists() else []
    detailed_by_key = {
        csv_row_key(row, DETAIL_KEY_FIELDS): row
        for row in all_detailed
        if csv_row_key(row, DETAIL_KEY_FIELDS) in task_keys
    }
    detailed_rows = list(detailed_by_key.values())
    summary_rows, dose_rows = summarize_detailed_rows(
        detailed_rows,
        bootstrap_trials=args.bootstrap_trials,
        seed=args.seed,
    )
    write_csv(
        detailed_path,
        detailed_rows,
        DETAIL_FIELDNAMES,
    )
    write_csv(
        output_dir / "steering_results.csv",
        summary_rows,
        SUMMARY_FIELDNAMES,
    )
    write_csv(
        output_dir / "steering_dose_response.csv",
        dose_rows,
        DOSE_FIELDNAMES,
    )
    plot_steering(summary_rows, Path(args.figure_dir) / "steering_curves.png")
    print(f"Wrote steering outputs to {output_dir}")


if __name__ == "__main__":
    main()
