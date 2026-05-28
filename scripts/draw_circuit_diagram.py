#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.plots import require_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a mechanistic circuit summary from completed experiment outputs.")
    parser.add_argument("--axis-summary", default="outputs/axis/axis_summary.csv")
    parser.add_argument("--steering-results", default="outputs/steering/steering_results.csv")
    parser.add_argument("--sae-features", default="outputs/sae/sae_features.csv")
    parser.add_argument("--output", default="figures/mechanistic_circuit_diagram.png")
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--max-p", type=float, default=0.01)
    parser.add_argument("--feature-threshold", type=float, default=0.1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def steering_amplitudes(rows: list[dict[str, str]]) -> dict[str, float]:
    amplitudes: dict[str, float] = defaultdict(float)
    for row in rows:
        axis_id = row.get("axis_id", "")
        delta = abs(as_float(row.get("delta_logprob_diff")))
        amplitudes[axis_id] = max(amplitudes[axis_id], delta)
    return dict(amplitudes)


def selected_features(
    rows: list[dict[str, str]],
    axes: set[str],
    *,
    threshold: float,
    per_axis: int = 3,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        axis_id = row.get("axis_id", "")
        if axis_id not in axes:
            continue
        if as_float(row.get("axis_contribution")) < threshold:
            continue
        grouped[axis_id].append(row)
    for axis_id in grouped:
        grouped[axis_id].sort(key=lambda item: as_float(item.get("axis_contribution")), reverse=True)
        grouped[axis_id] = grouped[axis_id][:per_axis]
    return dict(grouped)


def main() -> None:
    args = parse_args()
    plt = require_matplotlib()

    axis_rows = read_csv(Path(args.axis_summary))
    sae_rows = read_csv(Path(args.sae_features))
    steering_rows = read_csv(Path(args.steering_results))
    amplitudes = steering_amplitudes(steering_rows)

    validated = [
        row
        for row in axis_rows
        if as_float(row.get("test_accuracy")) >= args.min_accuracy and as_float(row.get("permutation_p"), 1.0) <= args.max_p
    ]
    weak = [row for row in axis_rows if row not in validated]
    validated.sort(key=lambda row: as_float(row.get("test_accuracy")), reverse=True)
    weak.sort(key=lambda row: as_float(row.get("test_accuracy")), reverse=True)

    feature_map = selected_features(
        sae_rows,
        {row.get("axis_id", "") for row in validated},
        threshold=args.feature_threshold,
    )

    fig, ax = plt.subplots(figsize=(12, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, *, size: float = 8.4, lw: float = 1.1) -> None:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=lw, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.text(0.04, 0.95, "MECHANISTIC CIRCUIT HYPOTHESIS", fontsize=12, fontweight="bold")
    ax.text(0.04, 0.91, "Validated residual axes -> layer-local Gemma Scope 2 features -> concept-label readout", fontsize=9)
    ax.text(0.06, 0.84, "Validated concept axes", fontsize=10, fontweight="bold")
    ax.text(0.41, 0.84, "Residual stream site", fontsize=10, fontweight="bold")
    ax.text(0.67, 0.84, "Candidate SAE features", fontsize=10, fontweight="bold")

    if not validated:
        box(0.30, 0.46, 0.40, 0.12, "No axis passed the validation thresholds.", size=10)
    else:
        y0 = 0.70
        step = min(0.14, 0.54 / max(len(validated), 1))
        for index, row in enumerate(validated):
            y = y0 - index * step
            axis_id = row.get("axis_id", "")
            layer = row.get("best_layer", "?")
            accuracy = as_float(row.get("test_accuracy"))
            p_value = as_float(row.get("permutation_p"), 1.0)
            steer = amplitudes.get(axis_id, 0.0)

            axis_text = f"{axis_id}\nacc={accuracy:.2f}, p={p_value:.3g}\nmax steering delta={steer:.3g}"
            site_text = f"residual stream\nlayer {layer}"
            features = feature_map.get(axis_id, [])
            if features:
                feature_lines = [
                    f"F{feat.get('feature_id')}  c={as_float(feat.get('axis_contribution')):.3g}"
                    for feat in features
                ]
                feature_text = f"{axis_id}\n" + "\n".join(feature_lines)
            else:
                feature_text = f"{axis_id}\nno salient SAE feature\nabove threshold"

            box(0.05, y, 0.24, 0.10, axis_text)
            box(0.40, y + 0.01, 0.18, 0.08, site_text)
            box(0.68, y - 0.005, 0.25, 0.11, feature_text, size=8.0)
            arrow(0.29, y + 0.05, 0.40, y + 0.05)
            arrow(0.58, y + 0.05, 0.68, y + 0.05)

    if weak:
        weak_text = "\n".join(
            f"{row.get('axis_id')}: acc={as_float(row.get('test_accuracy')):.2f}, p={as_float(row.get('permutation_p'), 1.0):.3g}"
            for row in weak
        )
        box(0.05, 0.07, 0.88, 0.10, "Diagnostic axes excluded from main circuit:\n" + weak_text, size=8.2)

    ax.text(
        0.05,
        0.025,
        "Figure: circuit hypothesis derived from held-out axis validation, steering directionality, and SAE decoder alignment.",
        fontsize=8.5,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
