#!/usr/bin/env python3
"""Step 4: trace differential SAE features across layers and draw a graph."""

from __future__ import annotations

import argparse
import sys
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
    parser = argparse.ArgumentParser(description="Extract cross-layer differential SAE feature routes.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/circuit")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument("--sae-release", help="Override SAE release from preset.")
    parser.add_argument("--layers", default=None, help="Comma-separated layers. Defaults to preset early/mid/late.")
    parser.add_argument("--diabetes-type", default="type1", choices=["type1", "type2", "all"])
    parser.add_argument("--target-complication", default="kidney")
    parser.add_argument("--baseline-complication", default="neurological")
    parser.add_argument("--max-prompts-per-group", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--common-top-n", type=int, default=50)
    parser.add_argument("--edges-per-transition", type=int, default=10)
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
    max_prompts: int,
) -> list[dict[str, str]]:
    group = [row for row in rows if row.get("complication") == complication]
    if diabetes_type != "all":
        group = [row for row in group if row.get("diabetes_type") == diabetes_type]
    return group[:max_prompts]


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
        print(f"  encoded {row.get('prompt_id', '')}: acts shape {tuple(feature_acts.shape)}")
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
        "Differential SAE Feature Route",
        "Note: edges are decoder-direction similarity heuristics, not causal proof.",
        "",
    ]
    by_layer = {layer: [row for row in feature_rows if row["layer"] == layer] for layer in layers}
    for layer_index, layer in enumerate(layers):
        parts = []
        for row in by_layer[layer]:
            parts.append(
                f"Feature {row['feature_id']} "
                f"(diff={float(row['diff_activation']):.4f}, "
                f"target={float(row['target_activation']):.4f}, "
                f"baseline={float(row['baseline_activation']):.4f})"
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
    nx.draw_networkx_nodes(graph, pos, node_size=1500, node_color=node_colors, alpha=0.92)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_color="white")
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
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
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
    layers = parse_layers(args.layers, preset.default_layers)
    model_name = args.model_name or preset.model_name
    sae_release = args.sae_release or preset.sae_release
    prepend_bos = not args.no_prepend_bos

    rows = read_prompts(args.prompts)
    target_rows = select_prompt_group(
        rows,
        diabetes_type=args.diabetes_type,
        complication=args.target_complication,
        max_prompts=args.max_prompts_per_group,
    )
    baseline_rows = select_prompt_group(
        rows,
        diabetes_type=args.diabetes_type,
        complication=args.baseline_complication,
        max_prompts=args.max_prompts_per_group,
    )
    if not target_rows or not baseline_rows:
        raise SystemExit("Could not find both target and baseline prompt groups. Run Step 2 or adjust filters.")

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=sae_release,
        device=device,
        dtype=dtype,
        extra={
            "layers": ",".join(str(layer) for layer in layers),
            "target": args.target_complication,
            "baseline": args.baseline_complication,
            "diabetes": args.diabetes_type,
        },
    )
    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    layer_feature_rows: dict[int, list[dict[str, object]]] = {}
    decoder_vectors: dict[str, object] = {}

    for layer in layers:
        sae_id = preset.sae_id(layer)
        print(f"\n=== Layer {layer}: loading SAE {sae_id} ===")
        sae, cfg_dict, _ = load_sae_with_metadata(SAE, sae_release, sae_id, device)
        sae.eval()
        hook_name = infer_hook_name(sae, cfg_dict, preset.hook_name(layer))
        print(f"hook_name: {hook_name}")

        print(f"Encoding target prompts ({args.target_complication})...")
        target_acts, _ = encode_group_features(
            torch=torch,
            model=model,
            sae=sae,
            hook_name=hook_name,
            rows=target_rows,
            device=device,
            prepend_bos=prepend_bos,
            position=args.position,
            keyword=args.keyword,
            debug=args.debug,
        )
        print(f"Encoding baseline prompts ({args.baseline_complication})...")
        baseline_acts, _ = encode_group_features(
            torch=torch,
            model=model,
            sae=sae,
            hook_name=hook_name,
            rows=baseline_rows,
            device=device,
            prepend_bos=prepend_bos,
            position=args.position,
            keyword=args.keyword,
            debug=args.debug,
        )

        target_mean = target_acts.mean(dim=0)
        baseline_mean = baseline_acts.mean(dim=0)
        common_mean = torch.cat([target_acts, baseline_acts], dim=0).mean(dim=0)
        top_ids, top_diffs = top_differential_features(
            torch,
            target_mean,
            baseline_mean,
            common_mean,
            args.top_k,
            args.common_top_n,
        )

        rows_for_layer = []
        for rank, (feature_id, diff_value) in enumerate(zip(top_ids, top_diffs), start=1):
            node_id = f"L{layer}_F{feature_id}"
            row = {
                "node_id": node_id,
                "layer": layer,
                "rank": rank,
                "feature_id": feature_id,
                "diff_activation": float(diff_value),
                "target_activation": float(target_mean[feature_id].item()),
                "baseline_activation": float(baseline_mean[feature_id].item()),
                "common_activation": float(common_mean[feature_id].item()),
                "sae_release": sae_release,
                "sae_id": sae_id,
                "hook_name": hook_name,
            }
            rows_for_layer.append(row)
            feature_rows.append(row)
            decoder_vectors[node_id] = decoder_vector(sae, feature_id)
            print(
                f"  top {rank}: feature {feature_id} "
                f"diff={float(diff_value):.6f} "
                f"target={float(target_mean[feature_id].item()):.6f} "
                f"baseline={float(baseline_mean[feature_id].item()):.6f}"
            )
        layer_feature_rows[layer] = rows_for_layer
        del sae
        empty_device_cache(torch, device)

    graph = nx.DiGraph()
    palette = ["#2b6cb0", "#2f855a", "#b7791f", "#c53030", "#6b46c1"]
    for layer_idx, layer in enumerate(layers):
        for row in layer_feature_rows[layer]:
            graph.add_node(
                row["node_id"],
                layer=layer,
                rank=row["rank"],
                feature_id=row["feature_id"],
                label=f"L{layer}\nF{row['feature_id']}",
                color=palette[layer_idx % len(palette)],
                diff_activation=row["diff_activation"],
            )

    edge_rows = []
    for prev_layer, next_layer in zip(layers[:-1], layers[1:]):
        candidates = edge_candidates(
            torch,
            layer_feature_rows[prev_layer],
            layer_feature_rows[next_layer],
            decoder_vectors,
            decoder_vectors,
        )
        for source, target, similarity in candidates[: args.edges_per_transition]:
            graph.add_edge(source, target, similarity=float(similarity))
            edge_rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_layer": prev_layer,
                    "target_layer": next_layer,
                    "similarity": float(similarity),
                }
            )

    write_csv_dicts(
        output_dir / "circuit_features.csv",
        feature_rows,
        [
            "node_id",
            "layer",
            "rank",
            "feature_id",
            "diff_activation",
            "target_activation",
            "baseline_activation",
            "common_activation",
            "sae_release",
            "sae_id",
            "hook_name",
        ],
    )
    write_csv_dicts(
        output_dir / "circuit_edges.csv",
        edge_rows,
        ["source", "target", "source_layer", "target_layer", "similarity"],
    )
    nx.write_graphml(graph, output_dir / "circuit_graph.graphml")
    draw_graph(nx, plt, graph, output_dir / "circuit_graph.png")
    save_route_text(output_dir / "circuit_route.txt", feature_rows, layers)

    print("\n=== Differential route ===")
    print((output_dir / "circuit_route.txt").read_text(encoding="utf-8"))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
