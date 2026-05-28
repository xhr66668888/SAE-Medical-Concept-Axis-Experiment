#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.plots import require_matplotlib
from medical_axis import DEFAULT_AXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw an evidence-rich medical concept-axis schematic.")
    parser.add_argument("--axis-summary", default="outputs/axis/axis_summary.csv")
    parser.add_argument("--sae-features", default="outputs/sae/sae_features.csv")
    parser.add_argument("--output", default="figures/medical_concept_graph.png")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    plt = require_matplotlib()
    axis_rows = read_csv(Path(args.axis_summary))
    sae_rows = read_csv(Path(args.sae_features))
    if axis_rows:
        best_axes = axis_rows[:5]
    else:
        best_axes = [
            {
                "axis_id": axis.axis_id,
                "best_layer": "pending",
                "test_accuracy": "nan",
                "positive": axis.positive.label,
                "negative": axis.negative.label,
            }
            for axis in DEFAULT_AXES
        ]
    top_features = sae_rows[:8]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, *, weight="normal", size=10):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.2, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight)

    ax.text(0.05, 0.92, "PROMPT CONTRASTS", fontsize=11, fontweight="bold")
    ax.text(0.36, 0.92, "RESIDUAL AXES", fontsize=11, fontweight="bold")
    ax.text(0.64, 0.92, "SAE FEATURE EVIDENCE", fontsize=11, fontweight="bold")

    y = 0.78
    for row in best_axes:
        try:
            accuracy = f"{float(row.get('test_accuracy', 0)):.2f}"
        except (TypeError, ValueError):
            accuracy = "pending"
        if row.get("best_layer") == "pending":
            label = f"{row.get('axis_id')}\nresidual fit pending"
        else:
            label = f"{row.get('axis_id')}\nL{row.get('best_layer')} acc={accuracy}"
        prompt_label = row.get("axis_id", "")
        if "positive" in row and "negative" in row:
            prompt_label = f"{row['positive']}\nvs\n{row['negative']}"
        box(0.05, y, 0.22, 0.08, prompt_label, size=8.2)
        box(0.38, y, 0.16, 0.08, label, size=8.5)
        ax.annotate("", xy=(0.38, y + 0.04), xytext=(0.27, y + 0.04), arrowprops=dict(arrowstyle="->", lw=1.2))
        y -= 0.12

    if top_features:
        feature_text = "\n".join(
            f"{row['axis_id']} L{row['layer']} F{row['feature_id']}  c={float(row['axis_contribution']):.3g}"
            for row in top_features
        )
    else:
        feature_text = "Run SAE tracing to populate feature IDs\nand contribution scores."
    box(0.64, 0.30, 0.30, 0.48, feature_text, size=8.2)
    ax.annotate("", xy=(0.64, 0.54), xytext=(0.54, 0.54), arrowprops=dict(arrowstyle="->", lw=1.2))

    box(
        0.28,
        0.10,
        0.42,
        0.10,
        "Evidence chain: held-out template accuracy + null tests + steering + patching + SAE feature tracing",
        size=9,
    )
    ax.text(
        0.05,
        0.04,
        "Figure: medical concept-axis structure. Exact feature IDs and scores are shown when SAE tracing has been run.",
        fontsize=9,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
