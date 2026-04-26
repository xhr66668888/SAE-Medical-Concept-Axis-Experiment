#!/usr/bin/env python3
"""Step 4a: axis-aligned SAE feature tracing.

This script explains the same Type1-vs-Type2 residual direction found in
Step 3. The older Step 4 traces complication-vs-complication differences;
that is useful for context routes, but it does not directly explain the main
concept axis. Here we use matched Type1/Type2 prompt pairs and rank SAE
features by how much their paired activation difference writes along the
concept axis and the insulin-vs-metformin logit direction.
"""

from __future__ import annotations

import argparse
import json
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
    capture_resid_activation,
    choose_device,
    choose_dtype,
    configure_torch,
    empty_device_cache,
    first_parameter,
    infer_hook_name,
    load_model,
    load_sae_with_metadata,
    parse_layers,
    print_run_header,
    require_runtime_deps,
    resolve_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace SAE features aligned with the Type1-vs-Type2 concept axis.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/circuit_axis")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument("--sae-release", help="Override SAE release from preset.")
    parser.add_argument("--sae-id-format", help="Override preset SAE id format.")
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated layers, 'auto' for candidate layers, or 'best' for only the Step 3 best layer.",
    )
    parser.add_argument("--best-layers-json", default="outputs/axis/best_layers.json")
    parser.add_argument("--axes-path", default="outputs/axis/concept_axes_all_layers.pt")
    parser.add_argument("--axis-path", default="outputs/axis/concept_axis.pt")
    parser.add_argument("--use-split", default="train", choices=["train", "test", "all"])
    parser.add_argument(
        "--complications",
        default="all",
        help="Comma-separated complication subset or 'all'. Matched Type1/Type2 pairs are built within each complication.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-k-per-group", type=int, default=20)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--position", choices=["last", "keyword"], default="last")
    parser.add_argument("--keyword", default="diabetes")
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
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return plt


def load_axis_blob(torch, axes_path: Path, axis_path: Path, layer: int) -> dict[str, object]:
    if axes_path.exists():
        all_layers = torch.load(axes_path, map_location="cpu")
        if layer in all_layers:
            return all_layers[layer]
        if str(layer) in all_layers:
            return all_layers[str(layer)]
    if axis_path.exists():
        blob = torch.load(axis_path, map_location="cpu")
        if int(blob.get("layer", -1)) == layer:
            return blob
    raise SystemExit(
        f"Could not find concept axis for layer {layer}. "
        f"Run step3 first or pass --axes-path/--axis-path."
    )


def parse_layer_arg(raw: str, best_layers_json: Path, default_layers: tuple[int, ...]) -> list[int]:
    raw_norm = (raw or "").strip().lower()
    if raw_norm in {"auto", "best"}:
        if not best_layers_json.exists():
            raise SystemExit(f"{raw!r} requested but {best_layers_json} does not exist. Run step3 first.")
        blob = json.loads(best_layers_json.read_text(encoding="utf-8"))
        if raw_norm == "best":
            return [int(blob["best_layer"])]
        return [int(layer) for layer in blob.get("candidate_layers") or [blob["best_layer"]]]
    return parse_layers(raw, default_layers)


def parse_complications(raw: str, rows: list[dict[str, str]]) -> set[str]:
    if raw.strip().lower() == "all":
        return {row["complication"] for row in rows if row.get("complication")}
    return {part.strip() for part in raw.split(",") if part.strip()}


def build_matched_pairs(
    rows: list[dict[str, str]],
    *,
    use_split: str,
    complications: set[str],
    max_pairs: int | None,
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        complication = row.get("complication", "")
        if complication not in complications:
            continue
        diabetes_type = row.get("diabetes_type")
        if diabetes_type not in {"type1", "type2"}:
            continue
        key = (
            complication,
            row.get("surface_form_index", ""),
            row.get("template_variant", ""),
        )
        by_key[key][diabetes_type] = row

    pairs: list[dict[str, object]] = []
    for key in sorted(by_key):
        group = by_key[key]
        if "type1" not in group or "type2" not in group:
            continue
        type1 = group["type1"]
        type2 = group["type2"]
        if use_split != "all" and (type1.get("split") != use_split or type2.get("split") != use_split):
            continue
        pairs.append(
            {
                "pair_index": len(pairs) + 1,
                "complication": key[0],
                "surface_form_index": key[1],
                "template_variant": key[2],
                "type1": type1,
                "type2": type2,
            }
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def encode_prompt_features(
    *,
    torch,
    model,
    sae,
    hook_name: str,
    prompt: str,
    device: str,
    prepend_bos: bool,
    position: str,
    keyword: str,
    debug: bool,
):
    sae_param = first_parameter(sae)
    resid, token, token_idx = capture_resid_activation(
        model=model,
        prompt=prompt,
        hook_name=hook_name,
        device=device,
        prepend_bos=prepend_bos,
        position=position,
        keyword=keyword,
        debug=debug,
    )
    resid = resid.to(device=sae_param.device, dtype=sae_param.dtype)
    acts = sae.encode(resid).squeeze(0).detach().float().cpu()
    return acts, token, token_idx


def grouped_topk_stability(torch, pair_diffs, groups: list[str], decoder_axis_dot, feature_id: int, top_k: int):
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)
    hits = 0
    total = 0
    for indices in by_group.values():
        if not indices:
            continue
        mean_diff = pair_diffs[indices].mean(dim=0)
        contribution = mean_diff * decoder_axis_dot
        positive = contribution > 0
        if positive.any():
            masked = contribution.clone()
            masked[~positive] = float("-inf")
        else:
            masked = contribution
        k = min(top_k, masked.numel())
        top_ids = set(torch.topk(masked, k=k).indices.tolist())
        total += 1
        if feature_id in top_ids:
            hits += 1
    if total == 0:
        return 0.0, 0, 0
    return hits / total, hits, total


def side_name(value: float) -> str:
    if value > 0:
        return "type1"
    if value < 0:
        return "type2"
    return "zero"


def make_layer_plots(plt, output_dir: Path, layer_rows: list[dict[str, object]], feature_rows: list[dict[str, object]]) -> None:
    if layer_rows:
        layers = [int(row["layer"]) for row in layer_rows]
        top = [float(row["top_axis_projection_contribution"]) for row in layer_rows]
        top_sum = [float(row["sum_topk_positive_axis_contribution"]) for row in layer_rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(layers, top, marker="o", label="top feature")
        ax.plot(layers, top_sum, marker="s", label="sum top-k positive")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Axis projection contribution")
        ax.set_title("Axis-aligned SAE contribution by layer")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "axis_sae_contributions_by_layer.png", dpi=180)
        plt.close(fig)

    if feature_rows:
        best_layer = int(max(layer_rows, key=lambda row: float(row["top_axis_projection_contribution"]))["layer"])
        rows = [row for row in feature_rows if int(row["layer"]) == best_layer][:12]
        rows = list(reversed(rows))
        labels = [f"L{row['layer']} F{row['feature_id']}" for row in rows]
        values = [float(row["axis_projection_contribution"]) for row in rows]
        colors = ["#2f855a" if row["activation_side"] == row["decoder_side"] else "#c53030" for row in rows]
        fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * 0.35)))
        ax.barh(labels, values, color=colors)
        ax.set_xlabel("Mean paired feature diff x decoder-axis dot")
        ax.set_title(f"Top axis-aligned SAE features at layer {best_layer}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "axis_sae_top_features.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = require_plot_deps()
    torch, SAE, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    preset = PRESETS[args.preset]
    model_name = args.model_name or preset.model_name
    sae_release = args.sae_release or preset.sae_release
    layers = parse_layer_arg(args.layers, Path(args.best_layers_json), preset.default_layers)
    prepend_bos = not args.no_prepend_bos

    rows = read_prompts(args.prompts)
    complications = parse_complications(args.complications, rows)
    pairs = build_matched_pairs(
        rows,
        use_split=args.use_split,
        complications=complications,
        max_pairs=args.max_pairs,
    )
    if not pairs:
        raise SystemExit("No matched Type1/Type2 pairs found. Check --use-split and --complications.")

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=sae_release,
        device=device,
        dtype=dtype,
        extra={
            "layers": ",".join(str(layer) for layer in layers),
            "pairs": len(pairs),
            "split": args.use_split,
            "complications": sorted(complications),
            "position": args.position,
        },
    )

    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)
    type1_token_id = resolve_token_id(model.tokenizer, args.type1_token)
    type2_token_id = resolve_token_id(model.tokenizer, args.type2_token)
    logit_diff_vec = (model.W_U[:, type1_token_id] - model.W_U[:, type2_token_id]).detach().float().cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_feature_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    def make_sae_id(layer: int) -> str:
        if args.sae_id_format:
            return args.sae_id_format.format(layer=layer)
        return preset.sae_id(layer)

    for layer in layers:
        axis_blob = load_axis_blob(torch, Path(args.axes_path), Path(args.axis_path), layer)
        axis_unit = axis_blob["axis_unit"].detach().float().cpu()
        axis_norm = float(torch.linalg.vector_norm(axis_blob["axis"].detach().float().cpu()).item())
        sae_id = make_sae_id(layer)

        print(f"\n=== Layer {layer}: loading SAE {sae_id} ===")
        sae, cfg_dict, _ = load_sae_with_metadata(SAE, sae_release, sae_id, device)
        sae.eval()
        hook_name = infer_hook_name(sae, cfg_dict, preset.hook_name(layer))
        print(f"hook_name: {hook_name}")

        type1_acts = []
        type2_acts = []
        pair_records = []
        for pair in pairs:
            type1 = pair["type1"]
            type2 = pair["type2"]
            t1_acts, t1_token, t1_pos = encode_prompt_features(
                torch=torch,
                model=model,
                sae=sae,
                hook_name=hook_name,
                prompt=type1["prompt"],
                device=device,
                prepend_bos=prepend_bos,
                position=args.position,
                keyword=args.keyword,
                debug=args.debug,
            )
            t2_acts, t2_token, t2_pos = encode_prompt_features(
                torch=torch,
                model=model,
                sae=sae,
                hook_name=hook_name,
                prompt=type2["prompt"],
                device=device,
                prepend_bos=prepend_bos,
                position=args.position,
                keyword=args.keyword,
                debug=args.debug,
            )
            type1_acts.append(t1_acts)
            type2_acts.append(t2_acts)
            pair_records.append(
                {
                    "pair_index": pair["pair_index"],
                    "complication": pair["complication"],
                    "surface_form_index": pair["surface_form_index"],
                    "template_variant": pair["template_variant"],
                    "type1_prompt_id": type1["prompt_id"],
                    "type2_prompt_id": type2["prompt_id"],
                    "type1_token": t1_token,
                    "type2_token": t2_token,
                    "type1_position_index": t1_pos,
                    "type2_position_index": t2_pos,
                }
            )
            if len(type1_acts) % 25 == 0:
                print(f"  encoded {len(type1_acts)}/{len(pairs)} matched pairs")

        type1_stack = torch.stack(type1_acts)
        type2_stack = torch.stack(type2_acts)
        pair_diffs = type1_stack - type2_stack
        mean_type1 = type1_stack.mean(dim=0)
        mean_type2 = type2_stack.mean(dim=0)
        mean_diff = pair_diffs.mean(dim=0)
        diff_std = pair_diffs.std(dim=0, unbiased=False)
        cohen_d = mean_diff / (diff_std + 1e-6)

        w_dec = sae.W_dec.detach().float().cpu()
        decoder_axis_dot = w_dec @ axis_unit
        decoder_norm = torch.linalg.vector_norm(w_dec, dim=1).clamp_min(1e-8)
        decoder_axis_cosine = decoder_axis_dot / decoder_norm
        decoder_logit_dot = w_dec @ logit_diff_vec

        axis_contribution = mean_diff * decoder_axis_dot
        logit_contribution = mean_diff * decoder_logit_dot
        positive_mask = axis_contribution > 0
        rank_values = axis_contribution.clone()
        if positive_mask.any():
            rank_values[~positive_mask] = float("-inf")
        k = min(args.top_k, int(positive_mask.sum().item()) if positive_mask.any() else rank_values.numel())
        top_values, top_ids = torch.topk(rank_values, k=k)

        complication_groups = [str(record["complication"]) for record in pair_records]
        template_groups = [str(record["template_variant"]) for record in pair_records]
        rows_for_layer: list[dict[str, object]] = []
        for rank, (feature_id, rank_value) in enumerate(zip(top_ids.tolist(), top_values.tolist()), start=1):
            pair_contrib_values = (pair_diffs[:, feature_id] * decoder_axis_dot[feature_id]).tolist()
            contrib_mean, contrib_low, contrib_high = bootstrap_ci(
                pair_contrib_values,
                n=args.bootstrap_trials,
                seed=layer * 100_000 + feature_id,
            )
            comp_stability, comp_hits, comp_total = grouped_topk_stability(
                torch,
                pair_diffs,
                complication_groups,
                decoder_axis_dot,
                feature_id,
                args.top_k_per_group,
            )
            template_stability, template_hits, template_total = grouped_topk_stability(
                torch,
                pair_diffs,
                template_groups,
                decoder_axis_dot,
                feature_id,
                args.top_k_per_group,
            )
            sign_consistency = sum(1 for value in pair_contrib_values if value > 0) / max(1, len(pair_contrib_values))
            diff_value = float(mean_diff[feature_id].item())
            axis_dot_value = float(decoder_axis_dot[feature_id].item())
            activation_side = side_name(diff_value)
            decoder_side = side_name(axis_dot_value)
            row = {
                "node_id": f"L{layer}_F{feature_id}_type1_vs_type2",
                "layer": layer,
                "rank": rank,
                "feature_id": feature_id,
                "mean_type1_activation": float(mean_type1[feature_id].item()),
                "mean_type2_activation": float(mean_type2[feature_id].item()),
                "mean_pair_diff_type1_minus_type2": diff_value,
                "pair_diff_std": float(diff_std[feature_id].item()),
                "cohen_d": float(cohen_d[feature_id].item()),
                "decoder_axis_dot": axis_dot_value,
                "decoder_axis_cosine": float(decoder_axis_cosine[feature_id].item()),
                "axis_projection_contribution": float(axis_contribution[feature_id].item()),
                "axis_projection_ci_low": contrib_low,
                "axis_projection_ci_high": contrib_high,
                "decoder_logit_diff_dot": float(decoder_logit_dot[feature_id].item()),
                "logit_diff_contribution": float(logit_contribution[feature_id].item()),
                "activation_side": activation_side,
                "decoder_side": decoder_side,
                "sign_consistency": float(sign_consistency),
                "complication_stability": float(comp_stability),
                "complication_hits": comp_hits,
                "complication_total": comp_total,
                "template_stability": float(template_stability),
                "template_hits": template_hits,
                "template_total": template_total,
                "pairs": len(pairs),
                "split": args.use_split,
                "position": args.position,
                "axis_norm": axis_norm,
                "sae_release": sae_release,
                "sae_id": sae_id,
                "hook_name": hook_name,
            }
            rows_for_layer.append(row)
            all_feature_rows.append(row)
            print(
                f"    top {rank}: F{feature_id} "
                f"contrib={rank_value:.4f} diff={diff_value:.3f} "
                f"axis_dot={axis_dot_value:.4f} "
                f"sign={sign_consistency:.2f} comp_stab={comp_stability:.2f}"
            )

        positive_contrib = axis_contribution[axis_contribution > 0]
        layer_rows.append(
            {
                "layer": layer,
                "pairs": len(pairs),
                "axis_norm": axis_norm,
                "top_axis_projection_contribution": float(rows_for_layer[0]["axis_projection_contribution"])
                if rows_for_layer
                else float("nan"),
                "top_feature_id": int(rows_for_layer[0]["feature_id"]) if rows_for_layer else "",
                "sum_topk_positive_axis_contribution": float(sum(float(row["axis_projection_contribution"]) for row in rows_for_layer)),
                "positive_feature_count": int(positive_contrib.numel()),
                "mean_positive_axis_contribution": float(positive_contrib.mean().item()) if positive_contrib.numel() else 0.0,
                "sae_release": sae_release,
                "sae_id": sae_id,
                "hook_name": hook_name,
            }
        )

        del sae
        empty_device_cache(torch, device)

    all_feature_rows.sort(key=lambda row: float(row["axis_projection_contribution"]), reverse=True)
    feature_columns = [
        "node_id",
        "layer",
        "rank",
        "feature_id",
        "mean_type1_activation",
        "mean_type2_activation",
        "mean_pair_diff_type1_minus_type2",
        "pair_diff_std",
        "cohen_d",
        "decoder_axis_dot",
        "decoder_axis_cosine",
        "axis_projection_contribution",
        "axis_projection_ci_low",
        "axis_projection_ci_high",
        "decoder_logit_diff_dot",
        "logit_diff_contribution",
        "activation_side",
        "decoder_side",
        "sign_consistency",
        "complication_stability",
        "complication_hits",
        "complication_total",
        "template_stability",
        "template_hits",
        "template_total",
        "pairs",
        "split",
        "position",
        "axis_norm",
        "sae_release",
        "sae_id",
        "hook_name",
    ]
    write_csv_dicts(output_dir / "axis_sae_features.csv", all_feature_rows, feature_columns)
    write_csv_dicts(
        output_dir / "axis_sae_layer_summary.csv",
        layer_rows,
        [
            "layer",
            "pairs",
            "axis_norm",
            "top_axis_projection_contribution",
            "top_feature_id",
            "sum_topk_positive_axis_contribution",
            "positive_feature_count",
            "mean_positive_axis_contribution",
            "sae_release",
            "sae_id",
            "hook_name",
        ],
    )
    make_layer_plots(plt, output_dir, layer_rows, all_feature_rows)

    best_layer_row = max(layer_rows, key=lambda row: float(row["top_axis_projection_contribution"]))
    lines = [
        "Axis-Aligned SAE Feature Summary",
        f"model_name: {model_name}",
        f"sae_release: {sae_release}",
        f"layers: {layers}",
        f"matched_pairs: {len(pairs)} split={args.use_split} complications={sorted(complications)}",
        f"position: {args.position} keyword={args.keyword!r}",
        f"type1_token: {args.type1_token!r} (id={type1_token_id})  type2_token: {args.type2_token!r} (id={type2_token_id})",
        f"best_layer_by_top_feature_contribution: {best_layer_row['layer']}",
        "",
        "layer  pairs  top_feature  top_contrib  sum_topk_contrib  positive_features",
    ]
    for row in layer_rows:
        lines.append(
            f"{int(row['layer']):5d} {int(row['pairs']):6d} "
            f"{str(row['top_feature_id']):>11} "
            f"{float(row['top_axis_projection_contribution']):11.4f} "
            f"{float(row['sum_topk_positive_axis_contribution']):16.4f} "
            f"{int(row['positive_feature_count']):17d}"
        )
    lines.append("")
    lines.append("Top axis-aligned features:")
    for row in all_feature_rows[: min(20, len(all_feature_rows))]:
        lines.append(
            f"L{row['layer']} F{row['feature_id']} rank={row['rank']} "
            f"contrib={float(row['axis_projection_contribution']):.4f} "
            f"ci=[{float(row['axis_projection_ci_low']):.4f},{float(row['axis_projection_ci_high']):.4f}] "
            f"diff={float(row['mean_pair_diff_type1_minus_type2']):.3f} "
            f"axis_dot={float(row['decoder_axis_dot']):.4f} "
            f"sign={float(row['sign_consistency']):.2f} "
            f"comp_stability={float(row['complication_stability']):.2f} "
            f"template_stability={float(row['template_stability']):.2f}"
        )
    (output_dir / "axis_sae_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:12]))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
