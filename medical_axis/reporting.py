from __future__ import annotations

import csv
from pathlib import Path


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _csv_table(path: Path, columns: list[str], *, max_rows: int = 8) -> str:
    if not path.exists():
        return "_Pending: run the corresponding model stage to populate this table._"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)[:max_rows]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, ""))[:90].replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def _figure(path: str, alt: str) -> str:
    if Path(path).exists():
        return f"![{alt}]({path})"
    return f"_Pending figure: `{path}`._"


def build_markdown_report(output_dir: Path, report_path: Path) -> None:
    axis_summary = output_dir / "axis" / "axis_summary.csv"
    steering_results = output_dir / "steering" / "steering_results.csv"
    patching_results = output_dir / "patching" / "patching_results.csv"
    sae_results = output_dir / "sae" / "sae_features.csv"

    lines = [
        "# Medical Concept Axis Experiment Report",
        "",
        "## Research Question",
        "",
        (
            "This experiment tests whether medical concepts form measurable residual-stream directions "
            "in an instruction-tuned language model, and whether those directions have causal effects on "
            "concept readouts. The design follows the contrast-vector logic of the Assistant Axis paper, "
            "but evaluates medical ontology contrasts rather than persona contrasts."
        ),
        "",
        "## Local Hardware",
        "",
        "- Target runtime: CPU-first local reproduction.",
        "- Main model: `google/gemma-3-1b-it`.",
        "- SAE source: Gemma Scope 2 residual-stream SAEs through SAELens.",
        "",
        "## Data",
        "",
        f"- Prompt rows: {_count_rows(output_dir / 'concept_prompts.csv')}",
        "- Prompt construction uses ICD/CCS diagnosis descriptions and held-out templates.",
        "- Readout uses concept labels, not specific drug names.",
        "",
        "## Axis Sweep",
        "",
        _figure("figures/medical_concept_graph.png", "Medical concept graph"),
        "",
        _figure("figures/accuracy_by_layer.png", "Held-out accuracy by layer"),
        "",
        _figure("figures/axis_cosine_heatmap.png", "Cross-axis cosine similarity"),
        "",
        _csv_table(
            axis_summary,
            ["axis_id", "best_layer", "test_accuracy", "test_ci_low", "test_ci_high", "random_null_mean", "permutation_p"],
        ),
        "",
        "## Mechanistic Circuit Summary",
        "",
        _figure("figures/mechanistic_circuit_diagram.png", "Mechanistic circuit diagram"),
        "",
        (
            "The circuit figure reports the axes that pass held-out and permutation-null checks, "
            "then links them to candidate Gemma Scope 2 residual-stream features. Axes that fail "
            "the validation criteria should be treated as diagnostics rather than primary evidence."
        ),
        "",
        "## Causal Steering",
        "",
        _figure("figures/steering_curves.png", "Steering curves"),
        "",
        _csv_table(steering_results, ["axis_id", "layer", "alpha", "mean_logprob_diff", "delta_logprob_diff"], max_rows=10),
        "",
        "## Activation Patching",
        "",
        _figure("figures/patching_heatmap.png", "Patching heatmap"),
        "",
        _csv_table(patching_results, ["axis_id", "layer", "position", "normalized_score"], max_rows=10),
        "",
        "## SAE Feature Tracing",
        "",
        _figure("figures/sae_top_features.png", "SAE feature contributions"),
        "",
        _csv_table(
            sae_results,
            ["axis_id", "layer", "feature_id", "axis_contribution", "activation_diff", "decoder_axis_dot"],
            max_rows=10,
        ),
        "",
        "## Interpretation",
        "",
        (
            "A concept axis should be treated as evidence for a structured representation only when the layer sweep, "
            "held-out label scoring, steering, patching, and SAE feature tracing agree. SAE features are candidate "
            "mechanistic units; they are not sufficient by themselves to claim a complete circuit."
        ),
        "",
        "## Limitations",
        "",
        "- The primary local run is constrained to small open Gemma models by CPU-only hardware.",
        "- Synthetic prompts test controlled concept representations, not clinical decision quality.",
        "- Linear axes are a useful probe, but medical concepts may also be represented nonlinearly or distributed across many layers.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
