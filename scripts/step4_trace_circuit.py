#!/usr/bin/env python3
"""Step 4: trace differential SAE features across layers and measure their
robustness to surface-form / template paraphrase.

Differences from the original draft:

1. Prompt selection now uses ALL train-split prompts that match a (diabetes,
   complication) condition, not just the first 3, so the differential signal
   is averaged over many surface forms and templates.
2. For every layer we compute the top-K differential features per template
   variant separately and report a Jaccard-style stability score across
   variants. Stable features (those that show up across paraphrases) are
   trustworthy candidates; volatile ones are likely surface-form artefacts.
3. The route still draws a heuristic decoder-similarity graph, but the CSVs
   include the stability score so downstream analysis can filter on it.
"""

from __future__ import annotations

import argparse
import itertools
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
    capture_resid_activation,
    choose_device,
    choose_dtype,
    configure_torch,
    first_parameter,
    infer_hook_name,
    empty_device_cache,
    load_model,
    load_sae_with_metadata,
    parse_layers,
    print_run_header,
    require_runtime_deps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cross-layer differential SAE feature routes with stability scoring.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/circuit")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument("--sae-release", help="Override SAE release from preset.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layers, 'auto' (read outputs/axis/best_layers.json), or omitted for preset early/mid/late.",
    )
    parser.add_argument(
        "--best-layers-json",
        default="outputs/axis/best_layers.json",
        help="Path used when --layers auto.",
    )
    parser.add_argument("--diabetes-type", default="type1", choices=["type1", "type2", "all"])
    parser.add_argument("--target-complication", default="kidney")
    parser.add_argument("--baseline-complication", default="neurological")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run every C(complications,2) pair at the single best layer (matrix mode).",
    )
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw the heuristic decoder-similarity graph. Off by default; the patching heatmap supersedes it.",
    )
    parser.add_argument(
        "--sae-id-format",
        default=None,
        help="Override the preset SAE id format (e.g. layer_{layer}_width_65k_l0_medium for the 65k-width check).",
    )
    parser.add_argument(
        "--max-prompts-per-group",
        type=int,
        default=None,
        help="Cap the number of train prompts per group. None = use all train prompts.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top differential features per layer (overall).")
    parser.add_argument(
        "--top-k-per-variant",
        type=int,
        default=20,
        help="Top features used per template variant for the stability score.",
    )
    parser.add_argument("--common-top-n", type=int, default=50)
    parser.add_argument("--edges-per-transition", type=int, default=10)
    parser.add_argument(
        "--use-split",
        default="train",
        choices=["train", "test", "all"],
        help="Which prompt split to use. Defaults to train.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--position", choices=["last", "keyword"], default="last")
    parser.add_argument("--keyword", default="diabetes")
    parser.add_argument("--no-prepend-bos", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def require_graph_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return nx, plt


def select_prompt_group(
    rows: list[dict[str, str]],
    *,
    diabetes_type: str,
    complication: str,
    use_split: str,
    max_prompts: int | None,
) -> list[dict[str, str]]:
    group = [row for row in rows if row.get("complication") == complication]
    if diabetes_type != "all":
        group = [row for row in group if row.get("diabetes_type") == diabetes_type]
    if use_split != "all":
        group = [row for row in group if row.get("split", "train") == use_split]
    if max_prompts is not None:
        group = group[:max_prompts]
    return group


def count_prompt_group(
    rows: list[dict[str, str]],
    *,
    diabetes_type: str,
    complication: str,
    use_split: str,
) -> int:
    return len(
        select_prompt_group(
            rows,
            diabetes_type=diabetes_type,
            complication=complication,
            use_split=use_split,
            max_prompts=None,
        )
    )


def encode_group_features(
    *,
    torch,
    model,
    sae,
    hook_name: str,
    rows: list[dict[str, str]],
    device: str,
    prepend_bos: bool,
    position: str,
    keyword: str | None,
    debug: bool,
):
    sae_param = first_parameter(sae)
    feature_rows = []
    token_records = []
    for row in rows:
        resid, token, token_idx = capture_resid_activation(
            model=model,
            prompt=row["prompt"],
            hook_name=hook_name,
            device=device,
            prepend_bos=prepend_bos,
            position=position,
            keyword=keyword,
            debug=debug,
        )
        resid = resid.to(device=sae_param.device, dtype=sae_param.dtype)
        feature_acts = sae.encode(resid).squeeze(0).detach().float().cpu()
        feature_rows.append(feature_acts)
        token_records.append((token, token_idx))
    return torch.stack(feature_rows), token_records


def top_differential_features(torch, target_mean, baseline_mean, common_mean, top_k: int, common_top_n: int):
    diff = target_mean - baseline_mean
    if common_top_n > 0:
        common_n = min(common_top_n, common_mean.numel())
        common_ids = torch.topk(common_mean, k=common_n).indices
        diff = diff.clone()
        diff[common_ids] = float("-inf")
    k = min(top_k, diff.numel())
    values, indices = torch.topk(diff, k=k)
    return indices.tolist(), values.tolist()


def stability_score(
    torch,
    target_acts,
    baseline_acts,
    target_variants: list[int],
    baseline_variants: list[int],
    feature_id: int,
    top_k_per_variant: int,
    common_mean,
    common_top_n: int,
) -> tuple[float, int, int]:
    """For each shared template variant, recompute top-K diff features and count
    how often `feature_id` makes the list. Returns (fraction, hits, total)."""
    common_ids = set()
    if common_top_n > 0:
        common_n = min(common_top_n, common_mean.numel())
        common_ids = set(torch.topk(common_mean, k=common_n).indices.tolist())

    target_by_variant: dict[int, list[int]] = defaultdict(list)
    baseline_by_variant: dict[int, list[int]] = defaultdict(list)
    for idx, variant in enumerate(target_variants):
        target_by_variant[variant].append(idx)
    for idx, variant in enumerate(baseline_variants):
        baseline_by_variant[variant].append(idx)

    shared_variants = sorted(set(target_by_variant) & set(baseline_by_variant))
    hits = 0
    total = 0
    for variant in shared_variants:
        t_idx = target_by_variant[variant]
        b_idx = baseline_by_variant[variant]
        if not t_idx or not b_idx:
            continue
        t_mean = target_acts[t_idx].mean(dim=0)
        b_mean = baseline_acts[b_idx].mean(dim=0)
        diff = (t_mean - b_mean).clone()
        if common_ids:
            for cid in common_ids:
                if cid < diff.numel():
                    diff[cid] = float("-inf")
        k = min(top_k_per_variant, diff.numel())
        top_ids = set(torch.topk(diff, k=k).indices.tolist())
        total += 1
        if feature_id in top_ids:
            hits += 1
    if total == 0:
        return 0.0, 0, 0
    return hits / total, hits, total


def decoder_vector(sae, feature_id: int):
    weight = getattr(sae, "W_dec", None)
    if weight is None:
        return None
    if weight.shape[0] <= feature_id:
        return None
    return weight[feature_id].detach().float().cpu()


def edge_candidates(torch, prev_layer, next_layer, prev_vectors, next_vectors):
    candidates = []
    for prev in prev_layer:
        prev_vec = prev_vectors.get(prev["node_id"])
        if prev_vec is None:
            continue
        for nxt in next_layer:
            next_vec = next_vectors.get(nxt["node_id"])
            if next_vec is None:
                continue
            sim = torch.nn.functional.cosine_similarity(prev_vec, next_vec, dim=0).item()
            candidates.append((prev["node_id"], nxt["node_id"], sim))
    if not candidates:
        for prev in prev_layer:
            for nxt in next_layer:
                candidates.append((prev["node_id"], nxt["node_id"], 0.0))
    return sorted(candidates, key=lambda item: item[2], reverse=True)


def save_route_text(path: Path, feature_rows: list[dict[str, object]], layers: list[int]) -> None:
    lines = [
        "Differential SAE Feature Route (with stability scores)",
        "Note: edges are decoder-direction similarity heuristics, not causal proof.",
        "Stability = fraction of template variants where this feature ranks in top-K diff features.",
        "",
    ]
    by_layer = {layer: [row for row in feature_rows if row["layer"] == layer] for layer in layers}
    for layer_index, layer in enumerate(layers):
        parts = []
        for row in by_layer[layer]:
            parts.append(
                f"F{row['feature_id']} "
                f"(diff={float(row['diff_activation']):.3f}, "
                f"stability={float(row['stability_score']):.2f} "
                f"[{int(row['stability_hits'])}/{int(row['stability_total'])}])"
            )
        lines.append(f"[Layer {layer}] " + " + ".join(parts))
        if layer_index < len(layers) - 1:
            lines.append("   ↓")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_graph(nx, plt, graph, output_path: Path) -> None:
    pos = {}
    layers = sorted({data["layer"] for _, data in graph.nodes(data=True)})
    for layer_idx, layer in enumerate(layers):
        layer_nodes = [node for node, data in graph.nodes(data=True) if data["layer"] == layer]
        layer_nodes = sorted(layer_nodes, key=lambda node: graph.nodes[node]["rank"])
        for rank, node in enumerate(layer_nodes):
            pos[node] = (layer_idx, -rank)

    plt.figure(figsize=(max(8, len(layers) * 3), 6))
    node_colors = [graph.nodes[node].get("color", "#4a5568") for node in graph.nodes]
    node_labels = {node: graph.nodes[node]["label"] for node in graph.nodes}
    edge_widths = [0.7 + max(0.0, graph.edges[edge].get("similarity", 0.0)) * 2.5 for edge in graph.edges]
    nx.draw_networkx_nodes(graph, pos, node_size=1700, node_color=node_colors, alpha=0.92)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=7, font_color="white")
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=edge_widths,
        edge_color="#718096",
        alpha=0.72,
    )
    edge_labels = {edge: f"{graph.edges[edge].get('similarity', 0.0):.2f}" for edge in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    nx, plt = require_graph_deps()
    torch, SAE, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    preset = PRESETS[args.preset]
    raw_layers = (args.layers or "").strip().lower()
    if raw_layers == "auto":
        best_path = Path(args.best_layers_json)
        if not best_path.exists():
            raise SystemExit(f"--layers auto requested but {best_path} does not exist. Run step3 first.")
        blob = json.loads(best_path.read_text(encoding="utf-8"))
        layers = list(blob.get("candidate_layers") or [blob.get("best_layer")])
        if args.all_pairs:
            layers = [int(blob["best_layer"])]
        print(f"Using --layers auto from {best_path}: {layers}")
    else:
        layers = parse_layers(args.layers, preset.default_layers)
        if args.all_pairs:
            layers = layers[:1]
    model_name = args.model_name or preset.model_name
    sae_release = args.sae_release or preset.sae_release
    sae_id_format_override = args.sae_id_format
    prepend_bos = not args.no_prepend_bos

    def make_sae_id(layer: int) -> str:
        if sae_id_format_override:
            return sae_id_format_override.format(layer=layer)
        return preset.sae_id(layer)

    rows = read_prompts(args.prompts)

    if args.all_pairs:
        complication_set = sorted({row.get("complication", "") for row in rows if row.get("complication")})
        pairs = list(itertools.combinations(complication_set, 2))
        print(f"all-pairs mode: complications={complication_set}, pairs={len(pairs)}")
    else:
        pairs = [(args.target_complication, args.baseline_complication)]

    viable_pairs = [
        (target_comp, baseline_comp)
        for target_comp, baseline_comp in pairs
        if count_prompt_group(rows, diabetes_type=args.diabetes_type, complication=target_comp, use_split=args.use_split) > 0
        and count_prompt_group(rows, diabetes_type=args.diabetes_type, complication=baseline_comp, use_split=args.use_split) > 0
    ]
    if not viable_pairs:
        raise SystemExit(
            "No target/baseline prompt groups are available for Step 4. "
            "Check --target-complication/--baseline-complication/--diabetes-type/--use-split."
        )

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=sae_release,
        device=device,
        dtype=dtype,
        extra={
            "layers": ",".join(str(layer) for layer in layers),
            "pairs": pairs,
            "diabetes": args.diabetes_type,
            "split": args.use_split,
            "all_pairs": args.all_pairs,
        },
    )
    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_feature_rows: list[dict[str, object]] = []
    pair_to_layer_features: dict[tuple[str, str], dict[int, list[dict[str, object]]]] = {}
    decoder_vectors: dict[str, object] = {}

    for layer in layers:
        sae_id = make_sae_id(layer)
        print(f"\n=== Layer {layer}: loading SAE {sae_id} ===")
        sae, cfg_dict, _ = load_sae_with_metadata(SAE, sae_release, sae_id, device)
        sae.eval()
        hook_name = infer_hook_name(sae, cfg_dict, preset.hook_name(layer))
        print(f"hook_name: {hook_name}")

        for target_comp, baseline_comp in pairs:
            target_rows = select_prompt_group(
                rows, diabetes_type=args.diabetes_type, complication=target_comp,
                use_split=args.use_split, max_prompts=args.max_prompts_per_group,
            )
            baseline_rows = select_prompt_group(
                rows, diabetes_type=args.diabetes_type, complication=baseline_comp,
                use_split=args.use_split, max_prompts=args.max_prompts_per_group,
            )
            if not target_rows or not baseline_rows:
                print(f"  pair ({target_comp} vs {baseline_comp}): missing prompts, skipping.")
                continue
            print(f"  pair ({target_comp} vs {baseline_comp}): target={len(target_rows)} baseline={len(baseline_rows)}")
            target_variants = [int(row.get("template_variant", 1) or 1) for row in target_rows]
            baseline_variants = [int(row.get("template_variant", 1) or 1) for row in baseline_rows]

            target_acts, _ = encode_group_features(
                torch=torch, model=model, sae=sae, hook_name=hook_name, rows=target_rows,
                device=device, prepend_bos=prepend_bos, position=args.position,
                keyword=args.keyword, debug=args.debug,
            )
            baseline_acts, _ = encode_group_features(
                torch=torch, model=model, sae=sae, hook_name=hook_name, rows=baseline_rows,
                device=device, prepend_bos=prepend_bos, position=args.position,
                keyword=args.keyword, debug=args.debug,
            )

            target_mean = target_acts.mean(dim=0)
            baseline_mean = baseline_acts.mean(dim=0)
            common_mean = torch.cat([target_acts, baseline_acts], dim=0).mean(dim=0)
            top_ids, top_diffs = top_differential_features(
                torch, target_mean, baseline_mean, common_mean, args.top_k, args.common_top_n,
            )

            rows_for_layer: list[dict[str, object]] = []
            for rank, (feature_id, diff_value) in enumerate(zip(top_ids, top_diffs), start=1):
                stability, hits, total = stability_score(
                    torch, target_acts, baseline_acts, target_variants, baseline_variants,
                    feature_id, args.top_k_per_variant, common_mean, args.common_top_n,
                )
                node_id = f"L{layer}_F{feature_id}_{target_comp}_vs_{baseline_comp}"
                row = {
                    "node_id": node_id,
                    "layer": layer,
                    "pair_target": target_comp,
                    "pair_baseline": baseline_comp,
                    "rank": rank,
                    "feature_id": feature_id,
                    "diff_activation": float(diff_value),
                    "target_activation": float(target_mean[feature_id].item()),
                    "baseline_activation": float(baseline_mean[feature_id].item()),
                    "common_activation": float(common_mean[feature_id].item()),
                    "stability_score": float(stability),
                    "stability_hits": hits,
                    "stability_total": total,
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                }
                rows_for_layer.append(row)
                all_feature_rows.append(row)
                decoder_vectors[node_id] = decoder_vector(sae, feature_id)
                print(
                    f"    top {rank}: F{feature_id} diff={diff_value:.3f} "
                    f"target={target_mean[feature_id].item():.3f} "
                    f"baseline={baseline_mean[feature_id].item():.3f} "
                    f"stability={stability:.2f} ({hits}/{total})"
                )
            pair_to_layer_features.setdefault((target_comp, baseline_comp), {})[layer] = rows_for_layer
        del sae
        empty_device_cache(torch, device)

    write_csv_dicts(
        output_dir / "circuit_features.csv",
        all_feature_rows,
        [
            "node_id", "layer", "pair_target", "pair_baseline", "rank", "feature_id",
            "diff_activation", "target_activation", "baseline_activation", "common_activation",
            "stability_score", "stability_hits", "stability_total",
            "sae_release", "sae_id", "hook_name",
        ],
    )

    # Per-pair route texts.
    for (target_comp, baseline_comp), layer_features in pair_to_layer_features.items():
        pair_rows = [r for r in all_feature_rows if r["pair_target"] == target_comp and r["pair_baseline"] == baseline_comp]
        ordered_layers = sorted(layer_features.keys())
        suffix = f"_{target_comp}_vs_{baseline_comp}" if args.all_pairs else ""
        save_route_text(output_dir / f"circuit_route{suffix}.txt", pair_rows, ordered_layers)

    # All-pairs Jaccard overlap matrix.
    if args.all_pairs and len(pairs) >= 2:
        overlap_path = output_dir / "all_pairs_jaccard.csv"
        feature_sets: dict[tuple[str, str], set[int]] = {}
        for (target_comp, baseline_comp), layer_features in pair_to_layer_features.items():
            ids: set[int] = set()
            for layer in layer_features:
                for row in layer_features[layer]:
                    ids.add(int(row["feature_id"]))
            feature_sets[(target_comp, baseline_comp)] = ids
        keys = list(feature_sets.keys())
        matrix_rows: list[dict[str, object]] = []
        for k1 in keys:
            row = {"pair": f"{k1[0]}_vs_{k1[1]}"}
            for k2 in keys:
                a, b = feature_sets[k1], feature_sets[k2]
                jacc = len(a & b) / max(1, len(a | b))
                row[f"{k2[0]}_vs_{k2[1]}"] = round(jacc, 3)
            matrix_rows.append(row)
        write_csv_dicts(overlap_path, matrix_rows, ["pair"] + [f"{k[0]}_vs_{k[1]}" for k in keys])
        print(f"\nWrote all-pairs Jaccard matrix to {overlap_path}")

    if args.draw_graph and not args.all_pairs:
        graph = nx.DiGraph()
        palette = ["#2b6cb0", "#2f855a", "#b7791f", "#c53030", "#6b46c1"]
        layer_features = pair_to_layer_features.get(pairs[0], {})
        ordered_layers = sorted(layer_features.keys())
        for layer_idx, layer in enumerate(ordered_layers):
            for row in layer_features[layer]:
                graph.add_node(
                    row["node_id"], layer=layer, rank=row["rank"], feature_id=row["feature_id"],
                    label=f"L{layer}\nF{row['feature_id']}\ns={row['stability_score']:.2f}",
                    color=palette[layer_idx % len(palette)],
                    diff_activation=row["diff_activation"], stability_score=row["stability_score"],
                )
        edge_rows = []
        for prev_layer, next_layer in zip(ordered_layers[:-1], ordered_layers[1:]):
            candidates = edge_candidates(
                torch, layer_features[prev_layer], layer_features[next_layer],
                decoder_vectors, decoder_vectors,
            )
            for source, target, similarity in candidates[: args.edges_per_transition]:
                graph.add_edge(source, target, similarity=float(similarity))
                edge_rows.append({
                    "source": source, "target": target,
                    "source_layer": prev_layer, "target_layer": next_layer,
                    "similarity": float(similarity),
                })
        write_csv_dicts(
            output_dir / "circuit_edges.csv", edge_rows,
            ["source", "target", "source_layer", "target_layer", "similarity"],
        )
        nx.write_graphml(graph, output_dir / "circuit_graph.graphml")
        draw_graph(nx, plt, graph, output_dir / "circuit_graph.png")

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
