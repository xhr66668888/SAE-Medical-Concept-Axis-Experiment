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

from medical_axis.io import read_csv, read_json, write_csv
from medical_axis.plots import plot_sae_features
from medical_axis.runtime import (
    capture_hidden_vector,
    choose_device,
    choose_dtype,
    configure_runtime,
    load_causal_lm,
    require_torch_transformers,
)


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


def main() -> None:
    args = parse_args()
    torch, _, _ = require_torch_transformers()
    SAE = require_sae_lens()
    configure_runtime(torch, threads=args.threads)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    axis_dir = Path(args.axis_dir)
    best = read_json(axis_dir / "best_axes.json")
    arrays = np.load(axis_dir / "axis_vectors.npz")
    model_name = args.model_name or str(best["model_name"])
    sae_release = args.sae_release
    if sae_release is None:
        sae_release = "gemma-scope-2-270m-it-res-all" if "270m" in model_name else "gemma-scope-2-1b-it-res-all"

    model, tokenizer = load_causal_lm(model_name, device=device, dtype=dtype)
    prompt_rows = read_csv(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []

    loaded_sae_by_layer: dict[int, object] = {}
    for axis_id, meta in sorted(dict(best["axes"]).items()):
        layer = int(meta["best_layer"])
        pairs = build_pairs(prompt_rows, axis_id, args.use_split, args.max_pairs_per_axis)
        if not pairs:
            continue
        if layer not in loaded_sae_by_layer:
            sae_id = args.sae_id_format.format(layer=layer)
            print(f"Loading SAE {sae_release} / {sae_id}")
            loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            sae.eval()
            loaded_sae_by_layer[layer] = sae
        sae = loaded_sae_by_layer[layer]
        axis_unit = torch.tensor(arrays[str(meta["axis_unit_key"])], dtype=torch.float32)
        pos_acts = []
        neg_acts = []
        for positive, negative in pairs:
            pos_vec = capture_hidden_vector(model, tokenizer, positive["prompt"], layer=layer, device=device)
            neg_vec = capture_hidden_vector(model, tokenizer, negative["prompt"], layer=layer, device=device)
            with torch.no_grad():
                pos_act = sae.encode(pos_vec.to(device=device, dtype=next(sae.parameters()).dtype).unsqueeze(0)).squeeze(0).float().cpu()
                neg_act = sae.encode(neg_vec.to(device=device, dtype=next(sae.parameters()).dtype).unsqueeze(0)).squeeze(0).float().cpu()
            pos_acts.append(pos_act)
            neg_acts.append(neg_act)
        pos_stack = torch.stack(pos_acts)
        neg_stack = torch.stack(neg_acts)
        mean_diff = (pos_stack - neg_stack).mean(dim=0)
        decoder = sae.W_dec.detach().float().cpu()
        decoder_axis_dot = decoder @ axis_unit
        contribution = mean_diff * decoder_axis_dot
        top_values, top_indices = torch.topk(contribution, k=min(args.top_k, contribution.numel()))
        for rank, (feature_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
            feature_rows.append(
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
                    "sae_id": args.sae_id_format.format(layer=layer),
                }
            )

    feature_rows.sort(key=lambda row: float(row["axis_contribution"]), reverse=True)
    write_csv(
        output_dir / "sae_features.csv",
        feature_rows,
        [
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
        ],
    )
    if feature_rows:
        plot_sae_features(feature_rows, Path(args.figure_dir) / "sae_top_features.png", top_k=args.top_k)
    print(f"Wrote SAE outputs to {output_dir}")


if __name__ == "__main__":
    main()
