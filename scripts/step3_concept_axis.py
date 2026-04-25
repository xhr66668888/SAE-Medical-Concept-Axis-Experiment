#!/usr/bin/env python3
"""Step 3: compute a Type 1 vs Type 2 diabetes concept axis and PCA plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_med.data_utils import balanced_take, read_prompts, write_csv_dicts
from sae_med.model_utils import (
    PRESETS,
    capture_resid_activation,
    choose_device,
    choose_dtype,
    configure_torch,
    load_model,
    parse_layers,
    print_run_header,
    require_runtime_deps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a Type 1 - Type 2 diabetes activation axis.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/axis")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument("--layer", type=int, default=None, help="Layer to capture. Defaults to preset middle layer.")
    parser.add_argument("--hook-name", help="Override hook name.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--position", choices=["last", "keyword"], default="last")
    parser.add_argument("--keyword", default="diabetes")
    parser.add_argument("--filter-complication", default=None, help="Optional complication value, e.g. kidney.")
    parser.add_argument("--max-prompts-per-class", type=int, default=None)
    parser.add_argument("--no-prepend-bos", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def require_plot_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return np, PCA, plt


def save_pca_plot(plt, points, labels, projections, output_path: Path) -> None:
    color_by_label = {"type1": "#2b6cb0", "type2": "#c53030"}
    marker_by_label = {"type1": "o", "type2": "^"}
    plt.figure(figsize=(8, 6))
    for label in ("type1", "type2"):
        idx = [i for i, value in enumerate(labels) if value == label]
        if not idx:
            continue
        xs = [points[i, 0] for i in idx]
        ys = [points[i, 1] for i in idx]
        plt.scatter(
            xs,
            ys,
            c=color_by_label[label],
            marker=marker_by_label[label],
            s=64,
            alpha=0.82,
            label=label,
            edgecolors="white",
            linewidths=0.6,
        )
    for idx, projection in enumerate(projections):
        plt.annotate(f"{projection:.2f}", (points[idx, 0], points[idx, 1]), fontsize=7, alpha=0.7)
    plt.axhline(0, color="#9ca3af", linewidth=0.8)
    plt.axvline(0, color="#9ca3af", linewidth=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Type 1 vs Type 2 Diabetes Residual Activations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    np, PCA, plt = require_plot_deps()
    torch, _, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    preset = PRESETS[args.preset]
    layer = args.layer if args.layer is not None else parse_layers(None, preset.default_layers)[1]
    model_name = args.model_name or preset.model_name
    hook_name = args.hook_name or preset.hook_name(layer)
    prepend_bos = not args.no_prepend_bos

    rows = read_prompts(args.prompts)
    if args.filter_complication:
        rows = [row for row in rows if row.get("complication") == args.filter_complication]
    rows = [row for row in rows if row.get("diabetes_type") in {"type1", "type2"}]
    rows = balanced_take(rows, "diabetes_type", ("type1", "type2"), args.max_prompts_per_class)
    if len(rows) < 4:
        raise SystemExit("Need at least two prompts per class. Run Step 2 or relax filters.")

    labels = [row["diabetes_type"] for row in rows]
    prompts = [row["prompt"] for row in rows]

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=None,
        device=device,
        dtype=dtype,
        extra={"hook_name": hook_name, "layer": layer, "prompts": len(prompts)},
    )
    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)

    activations = []
    token_records = []
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] Capturing {labels[idx - 1]} activation")
        resid, token, token_idx = capture_resid_activation(
            model=model,
            prompt=prompt,
            hook_name=hook_name,
            device=device,
            prepend_bos=prepend_bos,
            position=args.position,
            keyword=args.keyword,
            debug=args.debug,
        )
        activations.append(resid.squeeze(0).float().cpu())
        token_records.append((token, token_idx))

    acts = torch.stack(activations)
    mask_type1 = torch.tensor([label == "type1" for label in labels], dtype=torch.bool)
    mask_type2 = torch.tensor([label == "type2" for label in labels], dtype=torch.bool)
    mean_type1 = acts[mask_type1].mean(dim=0)
    mean_type2 = acts[mask_type2].mean(dim=0)
    axis = mean_type1 - mean_type2
    axis_unit = axis / axis.norm().clamp_min(1e-8)
    projections = (acts @ axis_unit).numpy()
    threshold = (projections[mask_type1.numpy()].mean() + projections[mask_type2.numpy()].mean()) / 2
    predicted = ["type1" if score >= threshold else "type2" for score in projections]
    accuracy = sum(pred == label for pred, label in zip(predicted, labels)) / len(labels)

    pca = PCA(n_components=2)
    points = pca.fit_transform(acts.numpy())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "axis": axis,
            "axis_unit": axis_unit,
            "mean_type1": mean_type1,
            "mean_type2": mean_type2,
            "hook_name": hook_name,
            "layer": layer,
            "model_name": model_name,
        },
        output_dir / "concept_axis.pt",
    )

    result_rows = []
    for row, label, point, projection, pred, token_record in zip(rows, labels, points, projections, predicted, token_records):
        result_rows.append(
            {
                "prompt_id": row.get("prompt_id", ""),
                "diabetes_type": label,
                "complication": row.get("complication", ""),
                "pc1": float(point[0]),
                "pc2": float(point[1]),
                "axis_projection": float(projection),
                "predicted_type": pred,
                "target_token": token_record[0],
                "target_token_index": token_record[1],
                "prompt": row["prompt"],
            }
        )
    write_csv_dicts(
        output_dir / "axis_results.csv",
        result_rows,
        [
            "prompt_id",
            "diabetes_type",
            "complication",
            "pc1",
            "pc2",
            "axis_projection",
            "predicted_type",
            "target_token",
            "target_token_index",
            "prompt",
        ],
    )
    save_pca_plot(plt, points, labels, projections, output_dir / "concept_axis_pca.png")

    summary = [
        "Concept Axis Summary",
        f"model_name: {model_name}",
        f"hook_name: {hook_name}",
        f"layer: {layer}",
        f"prompts: {len(prompts)}",
        f"type1_mean_projection: {float(projections[mask_type1.numpy()].mean()):.6f}",
        f"type2_mean_projection: {float(projections[mask_type2.numpy()].mean()):.6f}",
        f"threshold: {float(threshold):.6f}",
        f"axis_classifier_accuracy: {accuracy:.3f}",
        f"pca_explained_variance: {pca.explained_variance_ratio_.tolist()}",
    ]
    (output_dir / "axis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
