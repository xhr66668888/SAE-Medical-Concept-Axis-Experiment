#!/usr/bin/env python3
"""Step 3: per-layer concept axis sweep with held-out validation, null
distribution, direct logit attribution, logit lens, optional k-fold CV, and
bootstrap CIs.

For every prompt, we capture residual activations at every transformer layer in
a single forward pass. Then per layer we compute:

- mean-difference axis on the train split, classification accuracy on train and
  test splits, and a random-direction null distribution,
- direct logit attribution: ``axis_unit · (W_U[:, type1_token] − W_U[:, type2_token])``,
- "logit lens" difference: ``(mean_train_type1_resid − mean_train_type2_resid) ·
  (W_U[:, type1_token] − W_U[:, type2_token])`` -- i.e. whether the natural
  type difference at this layer's residual already aligns with the answer head.

Outputs:

- ``concept_axes_all_layers.pt`` -- every layer's axis tensor and metadata
- ``best_layers.json`` -- best layer (max test_acc - null_test_mean) plus
  ±2 neighbours, an early layer (≈n_layers // 5), and a late layer
  (≈n_layers - 3). Step 4/5/6 read this.
- ``concept_axis.pt`` -- the best layer's axis (back-compat for step5)
- ``axis_summary.txt`` -- per-layer table
- ``axis_results.csv`` -- per-prompt projection at the best layer
- ``concept_axis_pca.png``, ``accuracy_by_layer.png``, ``dla_and_logit_lens.png``,
  ``null_distribution_train.png``, ``null_distribution_test.png``
"""

from __future__ import annotations

import argparse
import json
import sys
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
    direct_logit_attribution,
    load_model,
    print_run_header,
    require_runtime_deps,
    resolve_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-layer concept-axis sweep + DLA + logit lens + null + CI.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/axis")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layers, or 'all' (default). 'auto' is not valid here.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--position", choices=["last", "keyword"], default="last")
    parser.add_argument("--keyword", default="diabetes")
    parser.add_argument("--filter-complication", default=None)
    parser.add_argument("--max-prompts-per-class", type=int, default=None)
    parser.add_argument("--null-trials", type=int, default=2000)
    parser.add_argument("--null-seed", type=int, default=20260425)
    parser.add_argument("--folds", type=int, default=1, help="K-fold CV. 1 disables (use the CSV split column).")
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--type1-token", default=" insulin")
    parser.add_argument("--type2-token", default=" metformin")
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
            f"Missing dependency: {missing}\nInstall dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return np, PCA, plt


def parse_layers_arg(raw: str, n_layers: int) -> list[int]:
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return list(range(n_layers))
    layers = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    invalid = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if invalid:
        raise SystemExit(f"Layer(s) out of range for n_layers={n_layers}: {invalid}")
    if not layers:
        raise SystemExit("--layers resolved to an empty list.")
    return layers


def kfold_split(prompts: list[dict[str, str]], k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Stratified k-fold over diabetes_type. Returns list of (train_idx, test_idx)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {"type1": [], "type2": []}
    for idx, row in enumerate(prompts):
        label = row.get("diabetes_type", "")
        if label in by_label:
            by_label[label].append(idx)
    min_label_count = min(len(indices) for indices in by_label.values())
    if k < 2:
        raise SystemExit("--folds must be >= 2 when k-fold mode is enabled.")
    if min_label_count < k:
        raise SystemExit(
            f"--folds={k} is too high for the class counts "
            f"(type1={len(by_label['type1'])}, type2={len(by_label['type2'])})."
        )
    for label in by_label:
        rng.shuffle(by_label[label])
    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(k):
        test_idx: list[int] = []
        train_idx: list[int] = []
        for label, indices in by_label.items():
            chunks = np.array_split(indices, k)
            test_idx.extend(int(x) for x in chunks[fold])
            for j, chunk in enumerate(chunks):
                if j != fold:
                    train_idx.extend(int(x) for x in chunk)
        folds.append((sorted(train_idx), sorted(test_idx)))
    return folds


def validate_split(train_idx: list[int], test_idx: list[int], labels: list[str], context: str) -> None:
    if not train_idx:
        raise SystemExit(f"{context}: train split is empty.")
    train_labels = {labels[i] for i in train_idx}
    if train_labels != {"type1", "type2"}:
        raise SystemExit(f"{context}: train split must contain both type1 and type2; found {sorted(train_labels)}.")
    if test_idx:
        test_labels = {labels[i] for i in test_idx}
        if test_labels != {"type1", "type2"}:
            raise SystemExit(f"{context}: test split must contain both type1 and type2; found {sorted(test_labels)}.")


def per_layer_metrics(
    np,
    layer_acts,
    train_idx: list[int],
    test_idx: list[int],
    labels: list[str],
    null_trials: int,
    null_seed: int,
):
    """layer_acts is a CPU numpy array [n_prompts, d_model]."""
    train_idx_np = np.asarray(train_idx)
    test_idx_np = np.asarray(test_idx)
    train_labels = np.asarray([labels[i] for i in train_idx])
    test_labels = np.asarray([labels[i] for i in test_idx]) if len(test_idx) else np.array([])

    train_acts = layer_acts[train_idx_np]
    mean_t1 = train_acts[train_labels == "type1"].mean(axis=0)
    mean_t2 = train_acts[train_labels == "type2"].mean(axis=0)
    axis = mean_t1 - mean_t2
    axis_norm = float(np.linalg.norm(axis))
    axis_unit = axis / max(1e-8, axis_norm)

    proj_all = layer_acts @ axis_unit
    proj_train = proj_all[train_idx_np]
    t1_proj_train = proj_train[train_labels == "type1"].mean() if (train_labels == "type1").any() else 0.0
    t2_proj_train = proj_train[train_labels == "type2"].mean() if (train_labels == "type2").any() else 0.0
    threshold = (t1_proj_train + t2_proj_train) / 2.0

    pred = np.where(proj_all >= threshold, "type1", "type2")
    train_acc = float((pred[train_idx_np] == train_labels).mean())
    test_acc = float((pred[test_idx_np] == test_labels).mean()) if len(test_idx_np) else float("nan")

    null_train_scores: list[float] = []
    null_test_scores: list[float] = []
    if null_trials > 0:
        rng = np.random.default_rng(null_seed)
        train_is_type1 = train_labels == "type1"
        test_is_type1 = test_labels == "type1" if len(test_idx_np) else None
        batch_size = 512
        d_model = layer_acts.shape[1]
        remaining = null_trials
        while remaining > 0:
            batch = min(batch_size, remaining)
            directions = rng.normal(size=(d_model, batch)).astype(np.float32)
            directions /= np.maximum(1e-8, np.linalg.norm(directions, axis=0, keepdims=True))
            proj = layer_acts @ directions
            tp = proj[train_idx_np]
            t1m = tp[train_is_type1].mean(axis=0)
            t2m = tp[~train_is_type1].mean(axis=0)
            thr = (t1m + t2m) / 2.0
            sign = np.where(t1m >= t2m, 1.0, -1.0)
            train_pred_is_type1 = sign * (tp - thr) >= 0
            null_train_scores.extend((train_pred_is_type1 == train_is_type1[:, None]).mean(axis=0).astype(float).tolist())
            if len(test_idx_np):
                te = proj[test_idx_np]
                test_pred_is_type1 = sign * (te - thr) >= 0
                null_test_scores.extend((test_pred_is_type1 == test_is_type1[:, None]).mean(axis=0).astype(float).tolist())
            remaining -= batch

    return {
        "axis": axis,
        "axis_unit": axis_unit,
        "axis_norm": axis_norm,
        "mean_type1": mean_t1,
        "mean_type2": mean_t2,
        "threshold": float(threshold),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "null_train_scores": null_train_scores,
        "null_test_scores": null_test_scores,
        "projections": proj_all,
        "predicted": pred,
        "train_t1_mean": float(t1_proj_train),
        "train_t2_mean": float(t2_proj_train),
    }


def main() -> None:
    args = parse_args()
    np, PCA, plt = require_plot_deps()
    torch, _, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    preset = PRESETS[args.preset]
    model_name = args.model_name or preset.model_name
    prepend_bos = not args.no_prepend_bos

    rows = read_prompts(args.prompts)
    if args.filter_complication:
        rows = [row for row in rows if row.get("complication") == args.filter_complication]
    rows = [row for row in rows if row.get("diabetes_type") in {"type1", "type2"}]
    if args.max_prompts_per_class is not None:
        kept: list[dict[str, str]] = []
        per_label: dict[str, int] = {"type1": 0, "type2": 0}
        for row in rows:
            lab = row["diabetes_type"]
            if per_label[lab] < args.max_prompts_per_class:
                kept.append(row)
                per_label[lab] += 1
        rows = kept
    if not rows:
        raise SystemExit("No prompts after filtering. Loosen --filter-complication / --max-prompts-per-class.")

    has_split_column = "split" in rows[0]
    if not has_split_column:
        for idx, row in enumerate(rows):
            row["split"] = "test" if idx % 4 == 0 else "train"

    labels = [row["diabetes_type"] for row in rows]

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=None,
        device=device,
        dtype=dtype,
        extra={"prompts": len(rows), "layers": args.layers, "folds": args.folds},
    )
    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)

    n_layers = int(model.cfg.n_layers)
    layers = parse_layers_arg(args.layers, n_layers)
    type1_id = resolve_token_id(model.tokenizer, args.type1_token)
    type2_id = resolve_token_id(model.tokenizer, args.type2_token)
    print(f"n_layers={n_layers}  layers={layers}  type1_token_id={type1_id}  type2_token_id={type2_id}")

    # Capture all chosen layers in one forward pass per prompt.
    per_layer_acts: dict[int, list] = {layer: [] for layer in layers}
    token_records: list[tuple[str, int]] = []
    for idx, row in enumerate(rows, start=1):
        if idx <= 3 or idx % 50 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] capturing all-layer residuals for {row['diabetes_type']} ({row.get('split','?')})")
        layer_to_tensor, token, token_idx = cache_all_layer_residuals(
            model=model,
            prompt=row["prompt"],
            device=device,
            prepend_bos=prepend_bos,
            position=args.position,
            keyword=args.keyword,
            layers=layers,
        )[:3]
        for layer in layers:
            per_layer_acts[layer].append(layer_to_tensor[layer])
        token_records.append((token, token_idx))

    layer_acts_np: dict[int, object] = {
        layer: torch.stack(per_layer_acts[layer]).numpy() for layer in layers
    }

    # Splits.
    if args.folds and args.folds > 1:
        folds = kfold_split(rows, args.folds, seed=args.null_seed)
    else:
        train_idx = [i for i, row in enumerate(rows) if row.get("split") == "train"]
        test_idx = [i for i, row in enumerate(rows) if row.get("split") == "test"]
        folds = [(train_idx, test_idx)]
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        validate_split(train_idx, test_idx, labels, f"fold {fold_idx}")

    # W_U direction for DLA / logit lens.
    w_u = model.W_U.detach().float().cpu().numpy()
    diff_dir = w_u[:, type1_id] - w_u[:, type2_id]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics per layer averaged across folds.
    per_layer_summary: list[dict[str, object]] = []
    per_layer_axis_blob: dict[int, dict[str, object]] = {}
    for layer in layers:
        fold_train: list[float] = []
        fold_test: list[float] = []
        fold_null_test_means: list[float] = []
        fold_null_test_p: list[float] = []
        last_metrics = None
        for train_idx, test_idx in folds:
            metrics = per_layer_metrics(
                np,
                layer_acts_np[layer],
                train_idx,
                test_idx,
                labels,
                null_trials=args.null_trials,
                null_seed=args.null_seed + layer,
            )
            fold_train.append(metrics["train_accuracy"])
            if not np.isnan(metrics["test_accuracy"]):
                fold_test.append(metrics["test_accuracy"])
            if metrics["null_test_scores"]:
                fold_null_test_means.append(float(np.mean(metrics["null_test_scores"])))
                fold_null_test_p.append(float((np.asarray(metrics["null_test_scores"]) >= metrics["test_accuracy"]).mean()))
            last_metrics = metrics

        train_mean = float(np.mean(fold_train))
        train_std = float(np.std(fold_train))
        test_mean = float(np.mean(fold_test)) if fold_test else float("nan")
        test_std = float(np.std(fold_test)) if fold_test else float("nan")
        null_test_mean = float(np.mean(fold_null_test_means)) if fold_null_test_means else float("nan")
        null_test_p = float(np.mean(fold_null_test_p)) if fold_null_test_p else float("nan")

        # Bootstrap CI on the per-prompt correctness vector at the LAST fold (cheap).
        train_idx, test_idx = folds[-1]
        pred = last_metrics["predicted"]
        if len(test_idx):
            test_correct = [int(pred[i] == labels[i]) for i in test_idx]
            ci_mean, ci_lower, ci_upper = bootstrap_ci(test_correct, n=args.bootstrap_trials, seed=args.null_seed + layer)
        else:
            ci_mean, ci_lower, ci_upper = float("nan"), float("nan"), float("nan")

        # DLA on the unit axis from the last fold.
        axis_unit = last_metrics["axis_unit"]
        axis_norm = last_metrics["axis_norm"]
        dla = float(np.dot(axis_unit, diff_dir))

        # Logit lens: (mean_t1 - mean_t2) at this layer projected onto diff_dir.
        # We use the train means of the last fold so it's directly comparable.
        mean_diff = last_metrics["mean_type1"] - last_metrics["mean_type2"]
        logit_lens = float(np.dot(mean_diff, diff_dir))

        per_layer_summary.append(
            {
                "layer": layer,
                "train_acc_mean": train_mean,
                "train_acc_std": train_std,
                "test_acc_mean": test_mean,
                "test_acc_std": test_std,
                "test_acc_ci_lower": ci_lower,
                "test_acc_ci_upper": ci_upper,
                "null_test_mean": null_test_mean,
                "null_p_test": null_test_p,
                "axis_norm": axis_norm,
                "dla": dla,
                "logit_lens": logit_lens,
            }
        )
        per_layer_axis_blob[layer] = {
            "axis": torch.tensor(last_metrics["axis"], dtype=torch.float32),
            "axis_unit": torch.tensor(last_metrics["axis_unit"], dtype=torch.float32),
            "mean_type1": torch.tensor(last_metrics["mean_type1"], dtype=torch.float32),
            "mean_type2": torch.tensor(last_metrics["mean_type2"], dtype=torch.float32),
            "threshold": last_metrics["threshold"],
            "train_accuracy": train_mean,
            "test_accuracy": test_mean,
            "test_accuracy_ci": (ci_lower, ci_upper),
            "null_test_mean": null_test_mean,
            "null_p_test": null_test_p,
            "dla": dla,
            "logit_lens": logit_lens,
            "hook_name": preset.hook_name(layer),
            "layer": layer,
            "model_name": model_name,
        }

    # Choose best layer: maximise (test_acc - null_test_mean), break ties by |dla|.
    def score_layer(row: dict[str, object]) -> float:
        gap = (row["test_acc_mean"] if not np.isnan(row["test_acc_mean"]) else 0.0) - (
            row["null_test_mean"] if not np.isnan(row["null_test_mean"]) else 0.0
        )
        return float(gap) + 1e-3 * abs(float(row["dla"]))

    best_row = max(per_layer_summary, key=score_layer)
    best_layer = int(best_row["layer"])
    print(
        f"Best layer = {best_layer} (test_acc={best_row['test_acc_mean']:.3f}, "
        f"null_mean={best_row['null_test_mean']:.3f}, dla={best_row['dla']:+.3f})"
    )

    # Pick neighbours, an early layer, and a late layer.
    neighbours = sorted({best_layer + d for d in (-2, -1, 0, 1, 2) if 0 <= best_layer + d < n_layers})
    early_layer = max(0, n_layers // 5)
    late_layer = max(0, n_layers - 3)
    candidate_layers = sorted(set(neighbours + [early_layer, late_layer]))
    candidate_layers = [layer for layer in candidate_layers if layer in layers]

    best_layers_blob = {
        "best_layer": best_layer,
        "candidate_layers": candidate_layers,
        "neighbours": neighbours,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "n_layers": n_layers,
        "model_name": model_name,
        "hook_name_format": preset.hook_name_format,
        "type1_token": args.type1_token,
        "type2_token": args.type2_token,
        "type1_token_id": type1_id,
        "type2_token_id": type2_id,
    }
    (output_dir / "best_layers.json").write_text(json.dumps(best_layers_blob, indent=2), encoding="utf-8")

    torch.save(per_layer_axis_blob, output_dir / "concept_axes_all_layers.pt")
    best_blob = dict(per_layer_axis_blob[best_layer])
    torch.save(best_blob, output_dir / "concept_axis.pt")

    # Per-prompt CSV at the best layer.
    train_idx, test_idx = folds[-1]
    splits_resolved = ["train" if i in set(train_idx) else "test" for i in range(len(rows))]
    proj_at_best = layer_acts_np[best_layer] @ best_blob["axis_unit"].numpy()
    pca = PCA(n_components=2)
    points = pca.fit_transform(layer_acts_np[best_layer])
    pred_at_best = np.where(proj_at_best >= best_blob["threshold"], "type1", "type2")
    rows_csv = []
    for i, row in enumerate(rows):
        rows_csv.append(
            {
                "prompt_id": row.get("prompt_id", ""),
                "diabetes_type": labels[i],
                "complication": row.get("complication", ""),
                "split": splits_resolved[i],
                "pc1": float(points[i, 0]),
                "pc2": float(points[i, 1]),
                "axis_projection": float(proj_at_best[i]),
                "predicted_type": pred_at_best[i],
                "correct": int(pred_at_best[i] == labels[i]),
                "target_token": token_records[i][0],
                "target_token_index": token_records[i][1],
                "prompt": row["prompt"],
            }
        )
    write_csv_dicts(
        output_dir / "axis_results.csv",
        rows_csv,
        [
            "prompt_id", "diabetes_type", "complication", "split",
            "pc1", "pc2", "axis_projection", "predicted_type", "correct",
            "target_token", "target_token_index", "prompt",
        ],
    )

    # PCA plot.
    color = {"type1": "#2b6cb0", "type2": "#c53030"}
    plt.figure(figsize=(8, 6))
    for label in ("type1", "type2"):
        for split, marker, alpha, size in (("train", "o", 0.85, 64), ("test", "^", 0.95, 90)):
            idx = [i for i in range(len(rows)) if labels[i] == label and splits_resolved[i] == split]
            if not idx:
                continue
            plt.scatter(
                [points[i, 0] for i in idx], [points[i, 1] for i in idx],
                c=color[label], marker=marker, s=size, alpha=alpha,
                label=f"{label}/{split}", edgecolors="white" if split == "train" else "black",
                linewidths=0.6 if split == "train" else 1.0,
            )
    plt.axhline(0, color="#9ca3af", linewidth=0.8)
    plt.axvline(0, color="#9ca3af", linewidth=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Type 1 vs Type 2 residual activations (layer {best_layer})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "concept_axis_pca.png", dpi=180)
    plt.close()

    # Accuracy by layer.
    layers_arr = np.array([row["layer"] for row in per_layer_summary])
    train_arr = np.array([row["train_acc_mean"] for row in per_layer_summary])
    test_arr = np.array([row["test_acc_mean"] for row in per_layer_summary])
    null_arr = np.array([row["null_test_mean"] for row in per_layer_summary])
    ci_lo = np.array([row["test_acc_ci_lower"] for row in per_layer_summary])
    ci_hi = np.array([row["test_acc_ci_upper"] for row in per_layer_summary])

    plt.figure(figsize=(10, 4.5))
    plt.plot(layers_arr, train_arr, marker="o", color="#2b6cb0", label="train")
    plt.plot(layers_arr, test_arr, marker="^", color="#2f855a", label="test")
    plt.fill_between(layers_arr, ci_lo, ci_hi, color="#2f855a", alpha=0.18, label="test 95% CI")
    plt.plot(layers_arr, null_arr, marker="x", color="#a0aec0", label="null mean")
    plt.axvline(best_layer, color="#c53030", linewidth=1.0, linestyle="--", label=f"best layer = {best_layer}")
    plt.axhline(0.5, color="#cbd5e0", linewidth=0.7)
    plt.xlabel("layer"); plt.ylabel("accuracy"); plt.ylim(0.3, 1.05)
    plt.title("Diabetes Type 1 vs Type 2 axis: classification accuracy by layer")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_layer.png", dpi=180)
    plt.close()

    # DLA + logit lens.
    dla_arr = np.array([row["dla"] for row in per_layer_summary])
    lens_arr = np.array([row["logit_lens"] for row in per_layer_summary])
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(layers_arr, dla_arr, marker="o", color="#2b6cb0", label="DLA: axis_unit · (W_U[t1] - W_U[t2])")
    ax1.set_xlabel("layer"); ax1.set_ylabel("DLA", color="#2b6cb0")
    ax1.axhline(0, color="#cbd5e0", linewidth=0.6)
    ax1.axvline(best_layer, color="#c53030", linewidth=1.0, linestyle="--")
    ax2 = ax1.twinx()
    ax2.plot(layers_arr, lens_arr, marker="^", color="#b7791f", label="logit lens (mean_t1 - mean_t2) · (W_U diff)")
    ax2.set_ylabel("logit lens diff", color="#b7791f")
    fig.suptitle("Direct logit attribution and logit-lens difference by layer")
    fig.tight_layout()
    fig.savefig(output_dir / "dla_and_logit_lens.png", dpi=180)
    plt.close(fig)

    # Null distribution plots at the best layer.
    final_metrics = per_layer_metrics(
        np, layer_acts_np[best_layer], folds[-1][0], folds[-1][1], labels,
        null_trials=args.null_trials, null_seed=args.null_seed + best_layer,
    )
    if final_metrics["null_train_scores"]:
        plt.figure(figsize=(7, 4))
        plt.hist(final_metrics["null_train_scores"], bins=30, color="#cbd5e0", edgecolor="#4a5568")
        plt.axvline(final_metrics["train_accuracy"], color="#c53030", linewidth=2, label=f"observed = {final_metrics['train_accuracy']:.3f}")
        plt.xlabel("random-direction accuracy"); plt.ylabel("count")
        plt.title(f"Train null distribution (layer {best_layer})")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / "null_distribution_train.png", dpi=180)
        plt.close()
    if final_metrics["null_test_scores"]:
        plt.figure(figsize=(7, 4))
        plt.hist(final_metrics["null_test_scores"], bins=30, color="#cbd5e0", edgecolor="#4a5568")
        plt.axvline(final_metrics["test_accuracy"], color="#c53030", linewidth=2, label=f"observed = {final_metrics['test_accuracy']:.3f}")
        plt.xlabel("random-direction accuracy"); plt.ylabel("count")
        plt.title(f"Held-out null distribution (layer {best_layer})")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / "null_distribution_test.png", dpi=180)
        plt.close()

    # Summary text.
    header = f"{'layer':>5} {'train':>6} {'test':>6} {'ci95':>13} {'null':>6} {'p_test':>7} {'dla':>8} {'lens':>9} {'|axis|':>8}"
    lines = [
        "Concept Axis Sweep Summary",
        f"model_name: {model_name}",
        f"n_layers: {n_layers}",
        f"prompts: {len(rows)} folds={args.folds}",
        f"type1_token: {args.type1_token!r} (id={type1_id})  type2_token: {args.type2_token!r} (id={type2_id})",
        f"best_layer: {best_layer}",
        f"candidate_layers (for downstream stages): {candidate_layers}",
        "",
        header,
    ]
    for row in per_layer_summary:
        lines.append(
            f"{row['layer']:>5d} "
            f"{row['train_acc_mean']:>6.3f} "
            f"{row['test_acc_mean']:>6.3f} "
            f"[{row['test_acc_ci_lower']:.2f},{row['test_acc_ci_upper']:.2f}] "
            f"{row['null_test_mean']:>6.3f} "
            f"{row['null_p_test']:>7.3f} "
            f"{row['dla']:>+8.3f} "
            f"{row['logit_lens']:>+9.3f} "
            f"{row['axis_norm']:>8.2f}"
        )
    summary_text = "\n".join(lines) + "\n"
    (output_dir / "axis_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
