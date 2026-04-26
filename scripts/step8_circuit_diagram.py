#!/usr/bin/env python3
"""Step 8: draw a black-and-white paper schematic.

The figure is a conceptual cartoon, not an attribution graph. Numeric feature
IDs and scores stay in CSV/report tables; the figure uses functional labels.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a black-and-white medical concept schematic.")
    parser.add_argument("--features-csv", default="outputs/circuit_axis/axis_sae_features.csv")
    parser.add_argument("--layer-summary-csv", default="outputs/circuit_axis/axis_sae_layer_summary.csv")
    parser.add_argument("--axis-summary", default="outputs/axis/axis_summary.txt")
    parser.add_argument("--patching-summary", default="outputs/patching/patching_summary.txt")
    parser.add_argument("--output-dir", default="outputs/circuit_diagram")
    parser.add_argument("--top-features", type=int, default=8)
    return parser.parse_args()


def require_plot_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return plt, FancyArrowPatch, FancyBboxPatch


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def best_layer(layer_rows: list[dict[str, str]], feature_rows: list[dict[str, str]]) -> int:
    if layer_rows:
        row = max(layer_rows, key=lambda item: as_float(item, "top_axis_projection_contribution"))
        return int(row["layer"])
    row = max(feature_rows, key=lambda item: as_float(item, "axis_projection_contribution"))
    return int(row["layer"])


def choose_display_features(rows: list[dict[str, str]], top_n: int) -> list[dict[str, str]]:
    robust = [
        row
        for row in rows
        if as_float(row, "axis_projection_ci_low") > 0
        and as_float(row, "sign_consistency") >= 0.70
        and as_float(row, "template_stability") >= 0.75
    ]
    pool = robust or rows
    pool = sorted(pool, key=lambda row: as_float(row, "axis_projection_contribution"), reverse=True)
    return pool[:top_n]


def extract(pattern: str, text: str, fallback: str = "n/a") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else fallback


def draw_box(ax, FancyBboxPatch, x, y, w, h, text, *, lw=1.2, fontsize=10.5, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.006",
        linewidth=lw,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        family="DejaVu Sans",
        linespacing=1.18,
    )


def draw_panel(ax, FancyBboxPatch, x, y, w, h):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        linewidth=0.9,
        edgecolor="black",
        facecolor="white",
        alpha=1.0,
    )
    ax.add_patch(patch)


def draw_arrow(ax, FancyArrowPatch, points, *, lw=1.6, linestyle="-", mutation=13):
    if len(points) > 2:
        xs = [p[0] for p in points[:-1]]
        ys = [p[1] for p in points[:-1]]
        ax.plot(xs, ys, color="black", lw=lw, linestyle=linestyle, solid_capstyle="butt")
    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color="black",
        linestyle=linestyle,
        connectionstyle="arc3,rad=0",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def main() -> None:
    args = parse_args()
    plt, FancyArrowPatch, FancyBboxPatch = require_plot_deps()

    features_path = Path(args.features_csv)
    layer_summary_path = Path(args.layer_summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(features_path)
    layer_rows = read_csv(layer_summary_path) if layer_summary_path.exists() else []
    selected_layer = best_layer(layer_rows, feature_rows)
    layer_features = [row for row in feature_rows if int(row["layer"]) == selected_layer]
    layer_features = choose_display_features(layer_features, args.top_features)

    axis_text = read_text(Path(args.axis_summary))
    patching_text = read_text(Path(args.patching_summary))
    axis_layer = extract(r"best_layer: ([^\n]+)", axis_text, str(selected_layer))
    patch_cell = extract(r"best_cell: ([^\n]+)", patching_text, "patching not run")
    patch_score = extract(r"best_mean_normalized_score: ([^\n]+)", patching_text, "n/a")

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headers.
    ax.text(0.07, 0.91, "PROMPTS", fontsize=11, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(0.37, 0.91, "SAE FEATURES", fontsize=11, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(0.62, 0.91, "CONCEPT AXIS", fontsize=11, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(0.82, 0.91, "DRUG READOUT", fontsize=11, fontweight="bold", family="DejaVu Sans Mono")

    # Two matched prompt lanes.
    draw_box(
        ax,
        FancyBboxPatch,
        0.07,
        0.66,
        0.23,
        0.12,
        'Type 1 prompt\n"... Type 1 diabetes ..."\n"ATC drug is"',
        fontsize=10.2,
    )
    draw_box(
        ax,
        FancyBboxPatch,
        0.07,
        0.43,
        0.23,
        0.12,
        'Type 2 prompt\n"... Type 2 diabetes ..."\n"ATC drug is"',
        fontsize=10.2,
    )

    # Candidate SAE feature clusters. The labels are functional summaries,
    # while exact feature IDs are kept in the CSV/report tables.
    draw_box(
        ax,
        FancyBboxPatch,
        0.37,
        0.665,
        0.20,
        0.11,
        "Type-1 diagnosis\nfeatures",
        fontsize=10.5,
    )
    draw_box(
        ax,
        FancyBboxPatch,
        0.37,
        0.435,
        0.20,
        0.11,
        "Type-2 diagnosis\nfeatures",
        fontsize=10.5,
    )

    # Shared concept axis and readout.
    draw_box(
        ax,
        FancyBboxPatch,
        0.64,
        0.43,
        0.15,
        0.35,
        "Diabetes-type\nconcept axis\n\nresidual stream",
        fontsize=10.8,
    )
    draw_box(ax, FancyBboxPatch, 0.86, 0.675, 0.10, 0.075, "insulin", fontsize=11.5, weight="bold")
    draw_box(ax, FancyBboxPatch, 0.86, 0.455, 0.10, 0.075, "metformin", fontsize=11.5, weight="bold")

    # Main paths: two straight lanes through the same concept-axis box.
    draw_arrow(ax, FancyArrowPatch, [(0.30, 0.72), (0.37, 0.72)], lw=1.8)
    draw_arrow(ax, FancyArrowPatch, [(0.30, 0.49), (0.37, 0.49)], lw=1.8)
    draw_arrow(ax, FancyArrowPatch, [(0.57, 0.72), (0.64, 0.72)], lw=1.8)
    draw_arrow(ax, FancyArrowPatch, [(0.57, 0.49), (0.64, 0.49)], lw=1.8)
    draw_arrow(ax, FancyArrowPatch, [(0.79, 0.72), (0.86, 0.712)], lw=1.8)
    draw_arrow(ax, FancyArrowPatch, [(0.79, 0.49), (0.86, 0.492)], lw=1.8)

    # A small evidence strip keeps the cartoon honest without crowding the graph.
    draw_panel(ax, FancyBboxPatch, 0.23, 0.17, 0.56, 0.10)
    ax.text(
        0.51,
        0.22,
        "Evidence used here: layer sweep + SAE feature tracing + steering + activation patching",
        ha="center",
        va="center",
        fontsize=9.8,
        family="DejaVu Sans",
    )

    ax.text(
        0.07,
        0.08,
        "Figure: simplified candidate circuit for the Type 1 / Type 2 diabetes concept axis.",
        fontsize=9.5,
        family="DejaVu Sans",
    )

    fig.tight_layout()
    png_path = output_dir / "medical_circuit_diagram.png"
    svg_path = output_dir / "medical_circuit_diagram.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    summary = [
        "Medical Concept Circuit Diagram",
        f"features_csv: {features_path}",
        f"selected_layer: {selected_layer}",
        f"display_features: {len(layer_features)}",
        "display_feature_ids: "
        + ", ".join(f"L{row.get('layer')}F{row.get('feature_id')}" for row in layer_features),
        f"axis_best_layer: {axis_layer}",
        f"patching: {patch_cell}",
        f"patching_score: {patch_score}",
        "layout: black-white paper schematic",
        "interpretation: schematic candidate SAE circuit, not complete causal proof",
    ]
    (output_dir / "medical_circuit_diagram_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
