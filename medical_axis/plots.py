from __future__ import annotations

from pathlib import Path

import numpy as np


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name or exc}. Install requirements.txt first.") from exc
    return plt


def plot_accuracy_by_layer(rows: list[dict[str, object]], output_path: str | Path) -> None:
    plt = require_matplotlib()
    axes = sorted({str(row["axis_id"]) for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for axis_id in axes:
        group = sorted([row for row in rows if row["axis_id"] == axis_id], key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in group]
        test = [float(row["test_accuracy"]) for row in group]
        null = [float(row["random_null_mean"]) for row in group]
        ax.plot(layers, test, marker="o", linewidth=1.4, label=axis_id)
        ax.plot(layers, null, linestyle=":", linewidth=0.9, color="0.65")
    ax.axhline(0.5, color="0.75", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Medical concept-axis separability by layer")
    ax.set_ylim(0.25, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cosine_heatmap(labels: list[str], matrix: np.ndarray, output_path: str | Path) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Similarity between best-layer medical concept axes")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_steering(rows: list[dict[str, object]], output_path: str | Path) -> None:
    plt = require_matplotlib()
    axes = sorted({str(row["axis_id"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for axis_id in axes:
        group = [row for row in rows if row["axis_id"] == axis_id]
        alphas = sorted({float(row["alpha"]) for row in group})
        means = []
        for alpha in alphas:
            vals = [float(row["delta_logprob_diff"]) for row in group if float(row["alpha"]) == alpha]
            means.append(float(np.mean(vals)))
        ax.plot(alphas, means, marker="o", linewidth=1.4, label=axis_id)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("Delta label log-probability difference")
    ax.set_title("Causal steering along medical concept axes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_patching_heatmap(rows: list[dict[str, object]], output_path: str | Path) -> None:
    plt = require_matplotlib()
    layers = sorted({int(row["layer"]) for row in rows})
    positions = sorted({int(row["position"]) for row in rows})
    matrix = np.full((len(layers), len(positions)), np.nan)
    for i, layer in enumerate(layers):
        for j, position in enumerate(positions):
            vals = [
                float(row["normalized_score"])
                for row in rows
                if int(row["layer"]) == layer and int(row["position"]) == position and row["normalized_score"] != ""
            ]
            if vals:
                matrix[i, j] = float(np.nanmean(vals))
    fig, ax = plt.subplots(figsize=(max(6, len(positions) * 1.2), max(4, len(layers) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="mean normalized patching score")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(pos) for pos in positions])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Patched token position")
    ax.set_ylabel("Layer")
    ax.set_title("Activation patching across concept-axis layers")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_sae_features(rows: list[dict[str, object]], output_path: str | Path, *, top_k: int = 20) -> None:
    plt = require_matplotlib()
    ranked = sorted(rows, key=lambda row: float(row["axis_contribution"]), reverse=True)[:top_k]
    ranked = list(reversed(ranked))
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(ranked) * 0.34)))
    labels = [f"{row['axis_id']} L{row['layer']} F{row['feature_id']}" for row in ranked]
    values = [float(row["axis_contribution"]) for row in ranked]
    ax.barh(labels, values, color="0.25")
    ax.set_xlabel("Mean activation difference x decoder-axis projection")
    ax.set_title("Top SAE features aligned with medical concept axes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
