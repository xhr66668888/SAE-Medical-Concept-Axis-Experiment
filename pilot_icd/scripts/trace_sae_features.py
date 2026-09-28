#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.io import append_csv_rows, csv_row_key, read_csv, read_json, write_csv
from medical_axis.plots import plot_sae_features
from medical_axis.runtime import (
    capture_hidden_vector,
    choose_device,
    choose_dtype,
    configure_runtime,
    load_causal_lm,
    require_torch_transformers,
)

FIXED_MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
FEATURE_FIELDNAMES = [
    "axis_id",
    "layer",
    "rank",
    "feature_id",
    "axis_contribution",
    "activation_diff",
    "decoder_axis_dot",
    "pairs",
    "sae_release",
    "sae_id",
]
AXIS_KEY_FIELDS = ["axis_id", "layer", "sae_release", "sae_id"]
FEATURE_KEY_FIELDS = ["axis_id", "layer", "rank", "sae_release", "sae_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Gemma Scope SAE features aligned with fitted medical axes.")
    parser.add_argument("--prompts", default="outputs/concept_prompts.csv")
    parser.add_argument("--axis-dir", default="outputs/axis")
    parser.add_argument("--output-dir", default="outputs/sae")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--sae-id-format", default="layer_{layer}_width_16k_l0_small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--use-split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-pairs-per-axis", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def require_sae_lens():
    try:
        from sae_lens import SAE
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: sae_lens. Install requirements.txt first.") from exc
    return SAE


def default_sae_release(model_name: str) -> str:
    if model_name != FIXED_MODEL_NAME:
        raise SystemExit(f"This experiment is configured for {FIXED_MODEL_NAME}.")
    return DEFAULT_SAE_RELEASE


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


def completed_axis_keys(rows: list[dict[str, str]], *, top_k: int) -> set[tuple[str, ...]]:
    ranks_by_axis: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for row in rows:
        try:
            rank = int(row["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= rank <= top_k:
            ranks_by_axis[csv_row_key(row, AXIS_KEY_FIELDS)].add(rank)
    required = set(range(1, top_k + 1))
    return {key for key, ranks in ranks_by_axis.items() if required.issubset(ranks)}


def dedupe_feature_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        by_key[csv_row_key(row, FEATURE_KEY_FIELDS)] = row
    return list(by_key.values())


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
    sae_release = args.sae_release
    if sae_release is None:
        sae_release = default_sae_release(model_name)

    prompt_rows = read_csv(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "sae_features.csv"
    existing_rows = read_csv(feature_path) if feature_path.exists() else []
    complete_axes = completed_axis_keys(existing_rows, top_k=args.top_k)
    axis_keys: set[tuple[str, ...]] = set()
    pending_axes: list[dict[str, object]] = []

    for axis_id, meta in sorted(dict(best["axes"]).items()):
        layer = int(meta["best_layer"])
        pairs = build_pairs(prompt_rows, axis_id, args.use_split, args.max_pairs_per_axis)
        if not pairs:
            continue
        sae_id = args.sae_id_format.format(layer=layer)
        axis_row = {"axis_id": axis_id, "layer": layer, "sae_release": sae_release, "sae_id": sae_id}
        axis_key = csv_row_key(axis_row, AXIS_KEY_FIELDS)
        axis_keys.add(axis_key)
        if axis_key not in complete_axes:
            pending_axes.append(
                {
                    "axis_id": axis_id,
                    "layer": layer,
                    "pairs": pairs,
                    "sae_id": sae_id,
                    "axis_unit_key": str(meta["axis_unit_key"]),
                }
            )

    if pending_axes:
        SAE = require_sae_lens()
        model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
        tasks_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
        for task in pending_axes:
            tasks_by_layer[int(task["layer"])].append(task)

        for layer, layer_tasks in sorted(tasks_by_layer.items()):
            sae_id = str(layer_tasks[0]["sae_id"])
            print(f"Loading SAE {sae_release} / {sae_id}")
            loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            sae.eval()
            try:
                for task in layer_tasks:
                    axis_id = str(task["axis_id"])
                    pairs = task["pairs"]
                    assert isinstance(pairs, list)
                    axis_unit = torch.tensor(arrays[str(task["axis_unit_key"])], dtype=torch.float32)
                    pos_acts = []
                    neg_acts = []
                    print(f"Tracing SAE features for {axis_id} layer {layer}: {len(pairs)} pairs")
                    for positive, negative in pairs:
                        pos_vec = capture_hidden_vector(model, tokenizer, positive["prompt"], layer=layer, device=device)
                        neg_vec = capture_hidden_vector(model, tokenizer, negative["prompt"], layer=layer, device=device)
                        with torch.inference_mode():
                            sae_dtype = next(sae.parameters()).dtype
                            pos_act = sae.encode(pos_vec.to(device=device, dtype=sae_dtype).unsqueeze(0)).squeeze(0).float().cpu()
                            neg_act = sae.encode(neg_vec.to(device=device, dtype=sae_dtype).unsqueeze(0)).squeeze(0).float().cpu()
                        pos_acts.append(pos_act)
                        neg_acts.append(neg_act)
                    pos_stack = torch.stack(pos_acts)
                    neg_stack = torch.stack(neg_acts)
                    mean_diff = (pos_stack - neg_stack).mean(dim=0)
                    decoder = sae.W_dec.detach().float().cpu()
                    decoder_axis_dot = decoder @ axis_unit
                    contribution = mean_diff * decoder_axis_dot
                    top_values, top_indices = torch.topk(contribution, k=min(args.top_k, contribution.numel()))
                    axis_rows = []
                    for rank, (feature_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                        axis_rows.append(
                            {
                                "axis_id": axis_id,
                                "layer": layer,
                                "rank": rank,
                                "feature_id": feature_id,
                                "axis_contribution": float(value),
                                "activation_diff": float(mean_diff[feature_id].item()),
                                "decoder_axis_dot": float(decoder_axis_dot[feature_id].item()),
                                "pairs": len(pairs),
                                "sae_release": sae_release,
                                "sae_id": sae_id,
                            }
                        )
                    append_csv_rows(feature_path, axis_rows, FEATURE_FIELDNAMES)
            finally:
                del sae
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
                    torch.xpu.empty_cache()
    else:
        print("SAE feature rows already complete; rebuilding sorted output from CSV.")

    all_feature_rows = read_csv(feature_path) if feature_path.exists() else []
    feature_rows = []
    for row in dedupe_feature_rows(all_feature_rows):
        try:
            rank = int(row["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        if csv_row_key(row, AXIS_KEY_FIELDS) in axis_keys and rank <= args.top_k:
            feature_rows.append(row)
    feature_rows.sort(key=lambda row: float(row["axis_contribution"]), reverse=True)
    write_csv(
        feature_path,
        feature_rows,
        FEATURE_FIELDNAMES,
    )
    if feature_rows:
        plot_sae_features(feature_rows, Path(args.figure_dir) / "sae_top_features.png", top_k=args.top_k)
    print(f"Wrote SAE outputs to {output_dir}")


if __name__ == "__main__":
    main()
