from __future__ import annotations

import csv
import os
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
        return "_Table unavailable: stage output was not generated._"
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


def _figure(path: Path, alt: str, *, report_path: Path) -> str:
    if path.exists():
        relative = os.path.relpath(path, start=report_path.parent)
        return f"![{alt}]({relative})"
    return f"_Figure unavailable: `{path}` was not generated._"


def build_markdown_report(output_dir: Path, report_path: Path, figure_dir: Path = Path("figures")) -> None:
    axis_summary = output_dir / "axis" / "axis_summary.csv"
    steering_results = output_dir / "steering" / "steering_results.csv"
    steering_dose = output_dir / "steering" / "steering_dose_response.csv"
    patching_results = output_dir / "patching" / "patching_results.csv"
    patching_summary = output_dir / "patching" / "patching_summary.csv"
    sae_results = output_dir / "sae" / "sae_features.csv"
    lexical_baseline = output_dir / "lexical_baseline.csv"

    lines = [
        "# Medical Concept Axis Experiment Report",
        "",
        "## Research Question",
        "",
        (
            "This experiment tests whether ICD-9-CM CCS diagnosis concepts form measurable residual-stream "
            "directions in an instruction-tuned language model, and whether those directions have causal "
            "effects on concept readouts. The design follows the contrast-vector logic of the Assistant Axis "
            "paper, but evaluates medical ontology contrasts rather than persona contrasts."
        ),
        "",
        "## Model and Data",
        "",
        "- Model: `google/gemma-3-4b-it`.",
        "- SAE source: Gemma Scope 2 residual-stream SAEs through SAELens.",
        f"- Prompt rows: {_count_rows(output_dir / 'concept_prompts.csv')}",
        "- Primary data source: AHRQ 2015 single-level CCS for ICD-9-CM diagnoses, joined to ICD-9 descriptions.",
        "- Prompt construction uses held-out diagnosis pairs and held-out templates.",
        "- Readout uses concept labels, not specific drug names.",
        "- Lexical baselines are reported separately because several ontology contrasts are keyword-visible.",
        "",
        "## Lexical Baseline",
        "",
        _csv_table(
            lexical_baseline,
            ["axis_id", "split", "rows", "answered", "coverage", "accuracy", "accuracy_with_abstain_wrong"],
            max_rows=8,
        ),
        "",
        "## Axis Sweep",
        "",
        _figure(figure_dir / "medical_concept_graph.png", "Medical concept graph", report_path=report_path),
        "",
        _figure(figure_dir / "accuracy_by_layer.png", "Held-out accuracy by layer", report_path=report_path),
        "",
        _figure(figure_dir / "axis_cosine_heatmap.png", "Cross-axis cosine similarity", report_path=report_path),
        "",
        _csv_table(
            axis_summary,
            [
                "axis_id",
                "best_layer",
                "test_accuracy",
                "test_ci_low",
                "test_ci_high",
                "random_null_mean",
                "permutation_p",
                "permutation_q_bh",
            ],
        ),
        "",
        "## Mechanistic Circuit Summary",
        "",
        _figure(figure_dir / "mechanistic_circuit_diagram.png", "Mechanistic circuit diagram", report_path=report_path),
        "",
        (
            "The circuit figure reports axes that pass held-out and multiple-comparison-adjusted permutation-null checks, "
            "then links them to candidate Gemma Scope 2 residual-stream features. Non-primary or failed axes are diagnostics."
        ),
        "",
        "## Causal Steering",
        "",
        _figure(figure_dir / "steering_curves.png", "Steering curves", report_path=report_path),
        "",
        _csv_table(
            steering_results,
            ["axis_id", "layer", "alpha", "mean_logprob_diff", "delta_logprob_diff", "delta_ci_low", "delta_ci_high"],
            max_rows=10,
        ),
        "",
        _csv_table(
            steering_dose,
            ["axis_id", "layer", "mean_prompt_slope", "slope_ci_low", "slope_ci_high", "positive_slope_fraction", "prompts"],
            max_rows=8,
        ),
        "",
        "## Activation Patching",
        "",
        _figure(figure_dir / "patching_heatmap.png", "Patching heatmap", report_path=report_path),
        "",
        _csv_table(
            patching_summary,
            ["axis_id", "layer", "position", "count", "mean_normalized_score", "ci_low", "ci_high"],
            max_rows=12,
        ),
        "",
        "## SAE Feature Tracing",
        "",
        _figure(figure_dir / "sae_top_features.png", "SAE feature contributions", report_path=report_path),
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
            "A concept axis should be treated as evidence for a structured representation only when it is a primary "
            "ICD-9 CCS contrast and the layer sweep, held-out label scoring, steering, patching, and SAE feature tracing "
            "agree. SAE features are candidate mechanistic units; they are not sufficient by themselves to claim a complete circuit."
        ),
        "",
        "## Reproducibility Checklist",
        "",
        "- Fixed model: `google/gemma-3-4b-it`.",
        "- Primary code system: ICD-9-CM CCS Appendix A; ICD-10 rows are excluded from primary claims.",
        "- Held-out diagnosis pairs and held-out templates are both used for primary evaluation.",
        "- Bootstrap intervals, random-direction nulls, and label-permutation nulls are reported for the axis sweep.",
        "- Steering and patching are evaluated on held-out prompts.",
        "- SAE feature tracing is used as candidate mechanistic decomposition, not as standalone evidence.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
