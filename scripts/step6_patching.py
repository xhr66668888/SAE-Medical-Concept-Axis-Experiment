#!/usr/bin/env python3
"""Step 6: activation patching on matched Type 1 / Type 2 prompt pairs.

For each matched pair with the same complication, surface form, and template:

  clean  = Type 1 prompt
  corrupt = Type 2 prompt

we cache corrupt residual-stream activations at the candidate layers chosen by
Step 3, then patch one residual vector into the clean run:

  clean resid_post[layer, position] <- corrupt resid_post[layer, position]

The main score is normalized so that 0 means "clean baseline" and 1 means
"fully moved to the corrupt baseline" for logit(insulin) - logit(metformin).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_med.data_utils import read_prompts, write_csv_dicts
from sae_med.model_utils import (
    PRESETS,
    bootstrap_ci,
    cache_all_layer_residuals,
    choose_device,
    choose_dtype,
    configure_torch,
    load_model,
    print_run_header,
    require_runtime_deps,
    resolve_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation patching over candidate layers and final positions.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--best-layers-json", default="outputs/axis/best_layers.json")
    parser.add_argument("--output-dir", default="outputs/patching")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset / best_layers.json.")
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated layers or 'auto' to read candidate_layers from --best-layers-json.",
    )
    parser.add_argument(
        "--complications",
        default="kidney,neurological",
        help="Comma-separated complications to patch, or 'all'. Default matches the Step 4 contrast.",
    )
    parser.add_argument("--use-split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--max-pairs-per-complication", type=int, default=None)
    parser.add_argument(
        "--positions",
        default="-1,-2,-3,-4",
        help="Comma-separated token positions to patch. Negative values are relative to the end.",
    )
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--type1-token", default=" insulin")
    parser.add_argument("--type2-token", default=" metformin")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--no-prepend-bos", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def require_plot_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name or str(exc)}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return np, plt


def parse_positions(raw: str) -> list[int]:
    positions = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not positions:
        raise SystemExit("--positions must contain at least one integer position.")
    return positions


def parse_layers(raw: str, best_layers_path: Path) -> tuple[list[int], dict[str, object]]:
    raw = (raw or "auto").strip().lower()
    blob: dict[str, object] = {}
    if raw == "auto":
        if not best_layers_path.exists():
            raise SystemExit(f"--layers auto requested but {best_layers_path} does not exist. Run Step 3 first.")
        blob = json.loads(best_layers_path.read_text(encoding="utf-8"))
        layers = blob.get("candidate_layers") or [blob.get("best_layer")]
        return sorted({int(layer) for layer in layers}), blob
    return [int(part.strip()) for part in raw.split(",") if part.strip()], blob


def parse_complications(raw: str, rows: list[dict[str, str]]) -> list[str]:
    raw = (raw or "").strip()
    if raw.lower() == "all":
        return sorted({row.get("complication", "") for row in rows if row.get("complication", "")})
    complications = [part.strip() for part in raw.split(",") if part.strip()]
    if not complications:
        raise SystemExit("--complications must name at least one complication, or use 'all'.")
    return complications


def numeric_key(value: str) -> tuple[int, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return 0, str(value)


def build_pairs(
    rows: list[dict[str, str]],
    *,
    complications: list[str],
    use_split: str,
    max_pairs_per_complication: int | None,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    rows = [row for row in rows if row.get("diabetes_type") in {"type1", "type2"}]
    if use_split != "all":
        rows = [row for row in rows if row.get("split", "train") == use_split]

    for complication in complications:
        type1: dict[tuple[str, str], dict[str, str]] = {}
        type2: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            if row.get("complication") != complication:
                continue
            key = (row.get("surface_form_index", ""), row.get("template_variant", ""))
            if row.get("diabetes_type") == "type1":
                type1[key] = row
            elif row.get("diabetes_type") == "type2":
                type2[key] = row
        shared_keys = sorted(
            set(type1) & set(type2),
            key=lambda item: (numeric_key(item[0]), numeric_key(item[1])),
        )
        if max_pairs_per_complication is not None:
            shared_keys = shared_keys[:max_pairs_per_complication]
        for key in shared_keys:
            pairs.append(
                {
                    "complication": complication,
                    "surface_form_index": key[0],
                    "template_variant": key[1],
                    "clean": type1[key],
                    "corrupt": type2[key],
                }
            )
    return pairs


def relative_position(seq_len: int, position: int) -> int | None:
    idx = seq_len + position if position < 0 else position
    if idx < 0 or idx >= seq_len:
        return None
    return idx


def logit_diff_from_logits(logits, type1_id: int, type2_id: int) -> float:
    last_logits = logits[0, -1, :].float().detach().cpu()
    return float(last_logits[type1_id].item() - last_logits[type2_id].item())


def logit_diff(model, tokens, type1_id: int, type2_id: int) -> float:
    logits = model(tokens, return_type="logits")
    return logit_diff_from_logits(logits, type1_id, type2_id)


def hook_name_for_layer(layer: int, best_layers_blob: dict[str, object], preset) -> str:
    hook_format = best_layers_blob.get("hook_name_format")
    if isinstance(hook_format, str) and "{layer}" in hook_format:
        return hook_format.format(layer=layer)
    return preset.hook_name(layer)


def main() -> None:
    args = parse_args()
    np, plt = require_plot_deps()
    torch, _, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    rows = read_prompts(args.prompts)
    complications = parse_complications(args.complications, rows)
    positions = parse_positions(args.positions)
    layers, best_layers_blob = parse_layers(args.layers, Path(args.best_layers_json))
    if not layers:
        raise SystemExit("No layers selected for patching.")

    pairs = build_pairs(
        rows,
        complications=complications,
        use_split=args.use_split,
        max_pairs_per_complication=args.max_pairs_per_complication,
    )
    if not pairs:
        raise SystemExit("No matched Type 1 / Type 2 prompt pairs found for patching.")

    preset = PRESETS[args.preset]
    model_name = args.model_name or str(best_layers_blob.get("model_name") or preset.model_name)
    prepend_bos = not args.no_prepend_bos
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)

    print_run_header(
        model_name=model_name,
        sae_release=None,
        device=device,
        dtype=dtype,
        extra={
            "layers": layers,
            "positions": positions,
            "complications": complications,
            "pairs": len(pairs),
            "split": args.use_split,
        },
    )
    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)
    type1_id = resolve_token_id(model.tokenizer, args.type1_token)
    type2_id = resolve_token_id(model.tokenizer, args.type2_token)
    print(f"type1 token id ({args.type1_token!r}): {type1_id}")
    print(f"type2 token id ({args.type2_token!r}): {type2_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    cell_scores: dict[tuple[int, int], list[float]] = defaultdict(list)

    for pair_index, pair in enumerate(pairs, start=1):
        clean_row = pair["clean"]
        corrupt_row = pair["corrupt"]
        clean_prompt = clean_row["prompt"]
        corrupt_prompt = corrupt_row["prompt"]

        clean_tokens = model.to_tokens(clean_prompt, prepend_bos=prepend_bos).to(device)
        clean_baseline = logit_diff(model, clean_tokens, type1_id, type2_id)
        corrupt_resids, _, _, _, corrupt_logits = cache_all_layer_residuals(
            model=model,
            prompt=corrupt_prompt,
            device=device,
            prepend_bos=prepend_bos,
            position="last",
            keyword=None,
            layers=layers,
            keep_full_sequence=True,
            return_logits=True,
        )
        corrupt_baseline = logit_diff_from_logits(corrupt_logits, type1_id, type2_id)
        denominator = corrupt_baseline - clean_baseline

        if pair_index <= 3 or pair_index % 10 == 0 or pair_index == len(pairs):
            print(
                f"[{pair_index}/{len(pairs)}] patching {pair['complication']} "
                f"surface={pair['surface_form_index']} template={pair['template_variant']}"
            )

        for layer in layers:
            hook_name = hook_name_for_layer(layer, best_layers_blob, preset)
            corrupt_layer_resid = corrupt_resids[layer]
            for position in positions:
                clean_idx = relative_position(int(clean_tokens.shape[1]), position)
                corrupt_idx = relative_position(int(corrupt_layer_resid.shape[0]), position)
                if clean_idx is None or corrupt_idx is None:
                    continue
                replacement = corrupt_layer_resid[corrupt_idx].to(device=device, dtype=dtype)

                def patch_hook(activation, hook, *, idx=clean_idx, resid=replacement):
                    patched = activation.clone()
                    patched[:, idx, :] = resid.to(device=activation.device, dtype=activation.dtype)
                    return patched

                with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                    patched_diff = logit_diff(model, clean_tokens, type1_id, type2_id)

                delta_from_clean = patched_diff - clean_baseline
                if abs(denominator) > 1e-8:
                    normalized = delta_from_clean / denominator
                else:
                    normalized = math.nan
                if math.isfinite(normalized):
                    cell_scores[(layer, position)].append(normalized)

                result_rows.append(
                    {
                        "pair_index": pair_index,
                        "complication": pair["complication"],
                        "surface_form_index": pair["surface_form_index"],
                        "template_variant": pair["template_variant"],
                        "clean_prompt_id": clean_row.get("prompt_id", ""),
                        "corrupt_prompt_id": corrupt_row.get("prompt_id", ""),
                        "layer": layer,
                        "hook_name": hook_name,
                        "position": position,
                        "clean_position_index": clean_idx,
                        "corrupt_position_index": corrupt_idx,
                        "clean_logit_diff_type1_minus_type2": clean_baseline,
                        "corrupt_logit_diff_type1_minus_type2": corrupt_baseline,
                        "patched_logit_diff_type1_minus_type2": patched_diff,
                        "delta_from_clean": delta_from_clean,
                        "normalized_score": normalized,
                        "clean_prompt": clean_prompt,
                        "corrupt_prompt": corrupt_prompt,
                    }
                )

    fieldnames = [
        "pair_index",
        "complication",
        "surface_form_index",
        "template_variant",
        "clean_prompt_id",
        "corrupt_prompt_id",
        "layer",
        "hook_name",
        "position",
        "clean_position_index",
        "corrupt_position_index",
        "clean_logit_diff_type1_minus_type2",
        "corrupt_logit_diff_type1_minus_type2",
        "patched_logit_diff_type1_minus_type2",
        "delta_from_clean",
        "normalized_score",
        "clean_prompt",
        "corrupt_prompt",
    ]
    write_csv_dicts(output_dir / "patching_results.csv", result_rows, fieldnames)

    finite_cells = []
    for cell, values in cell_scores.items():
        if values:
            finite_cells.append((cell, float(sum(values) / len(values)), values))
    best_cell = max(finite_cells, key=lambda item: item[1]) if finite_cells else None

    plot_positions = sorted(positions)
    plot_layers = sorted(layers)
    matrix = np.full((len(plot_layers), len(plot_positions)), np.nan, dtype=float)
    for i, layer in enumerate(plot_layers):
        for j, position in enumerate(plot_positions):
            values = cell_scores.get((layer, position), [])
            if values:
                matrix[i, j] = float(sum(values) / len(values))

    fig, ax = plt.subplots(figsize=(max(6, len(plot_positions) * 1.4), max(4, len(plot_layers) * 0.55)))
    if np.isfinite(matrix).any():
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r")
        fig.colorbar(im, ax=ax, label="mean normalized patching score")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite patching scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(range(len(plot_positions)))
    ax.set_xticklabels([str(pos) for pos in plot_positions])
    ax.set_yticks(range(len(plot_layers)))
    ax.set_yticklabels([str(layer) for layer in plot_layers])
    ax.set_xlabel("patched token position")
    ax.set_ylabel("layer")
    ax.set_title("Activation patching: corrupt Type 2 residual into clean Type 1 run")
    fig.tight_layout()
    fig.savefig(output_dir / "patching_heatmap.png", dpi=180)
    plt.close(fig)

    lines = [
        "Activation Patching Summary",
        f"model_name: {model_name}",
        f"pairs: {len(pairs)}  split={args.use_split}  complications={complications}",
        f"layers: {layers}",
        f"positions: {positions}",
        "score: (patched - clean_baseline) / (corrupt_baseline - clean_baseline)",
    ]
    if best_cell is not None:
        (best_layer, best_position), best_mean, best_values = best_cell
        if args.bootstrap_trials > 0:
            _, ci_lower, ci_upper = bootstrap_ci(
                best_values,
                n=args.bootstrap_trials,
                seed=20260425 + best_layer * 17 + abs(best_position),
            )
        else:
            ci_lower = ci_upper = best_mean
        lines.extend(
            [
                f"best_cell: layer={best_layer} position={best_position}",
                f"best_mean_normalized_score: {best_mean:.4f}",
                f"best_95pct_bootstrap_ci: [{ci_lower:.4f}, {ci_upper:.4f}]",
                f"best_cell_pairs: {len(best_values)}",
            ]
        )
    else:
        lines.append("best_cell: none (no finite normalized scores)")

    summary_text = "\n".join(lines) + "\n"
    (output_dir / "patching_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
