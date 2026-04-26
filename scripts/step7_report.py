#!/usr/bin/env python3
"""Step 7: aggregate the experiment artifacts into Markdown and HTML reports."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aggregate SAE experiment report.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--report-html", default=None)
    parser.add_argument("--title", default="SAE Diabetes Concept-Axis Experiment Report")
    parser.add_argument("--max-text-chars", type=int, default=12000)
    return parser.parse_args()


def read_text(path: Path, max_chars: int) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]\n"
    return text


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def csv_preview_markdown(
    path: Path,
    columns: list[str] | None = None,
    max_rows: int = 8,
    *,
    nonzero_column: str | None = None,
    sort_numeric_column: str | None = None,
    reverse: bool = False,
) -> str:
    if not path.exists():
        return "_Missing._"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        chosen = columns or fieldnames[:8]
        chosen = [col for col in chosen if col in fieldnames]
        rows = list(reader)
    if nonzero_column:
        rows = [row for row in rows if abs(_float_or_none(row.get(nonzero_column)) or 0.0) > 1e-9]
    if sort_numeric_column:
        rows.sort(key=lambda row: _float_or_none(row.get(sort_numeric_column)) or 0.0, reverse=reverse)
    rows = rows[:max_rows]
    if not chosen:
        return "_No columns found._"
    lines = [
        "| " + " | ".join(chosen) + " |",
        "| " + " | ".join("---" for _ in chosen) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|")[:96] for col in chosen]
        lines.append("| " + " | ".join(values) + " |")
    if not rows:
        lines.append("| " + " | ".join("" for _ in chosen) + " |")
    return "\n".join(lines)


def csv_preview_html(
    path: Path,
    columns: list[str] | None = None,
    max_rows: int = 8,
    *,
    nonzero_column: str | None = None,
    sort_numeric_column: str | None = None,
    reverse: bool = False,
) -> str:
    if not path.exists():
        return "<p><em>Missing.</em></p>"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        chosen = columns or fieldnames[:8]
        chosen = [col for col in chosen if col in fieldnames]
        rows = list(reader)
    if nonzero_column:
        rows = [row for row in rows if abs(_float_or_none(row.get(nonzero_column)) or 0.0) > 1e-9]
    if sort_numeric_column:
        rows.sort(key=lambda row: _float_or_none(row.get(sort_numeric_column)) or 0.0, reverse=reverse)
    rows = rows[:max_rows]
    if not chosen:
        return "<p><em>No columns found.</em></p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in chosen)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, ''))[:160])}</td>" for col in chosen)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def md_image(path: Path, output_root: Path, alt: str) -> str:
    if not path.exists():
        return f"_Missing image: `{path}`_"
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        rel = path
    return f"![{alt}]({rel.as_posix()})"


def html_image(path: Path, alt: str) -> str:
    if not path.exists():
        return f"<p><em>Missing image: {html.escape(str(path))}</em></p>"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


def pre_md(text: str | None) -> str:
    if text is None:
        return "_Missing._"
    return "```text\n" + text.rstrip() + "\n```"


def pre_html(text: str | None) -> str:
    if text is None:
        return "<p><em>Missing.</em></p>"
    return f"<pre>{html.escape(text.rstrip())}</pre>"


def first_existing_text(paths: list[Path], max_chars: int) -> str | None:
    chunks: list[str] = []
    for path in paths:
        text = read_text(path, max_chars)
        if text is None:
            continue
        chunks.append(f"{path.name}\n{text.rstrip()}")
    if not chunks:
        return None
    return "\n\n".join(chunks)


def extract_value(text: str | None, pattern: str) -> str | None:
    if text is None:
        return None
    match = re.search(pattern, text)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_md = Path(args.report_md) if args.report_md else output_root / "report.md"
    report_html = Path(args.report_html) if args.report_html else output_root / "report.html"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)

    axis_dir = output_root / "axis"
    circuit_axis_dir = output_root / "circuit_axis"
    circuit_diagram_dir = output_root / "circuit_diagram"
    steering_dir = output_root / "steering"
    patching_dir = output_root / "patching"
    replay_axis_dir = output_root / "270m" / "axis"
    replay_steering_dir = output_root / "270m" / "steering"

    axis_summary = read_text(axis_dir / "axis_summary.txt", args.max_text_chars)
    axis_sae_summary = read_text(circuit_axis_dir / "axis_sae_summary.txt", args.max_text_chars)
    steering_summary = read_text(steering_dir / "steering_summary.txt", args.max_text_chars)
    patching_summary = read_text(patching_dir / "patching_summary.txt", args.max_text_chars)
    replay_axis_summary = read_text(replay_axis_dir / "axis_summary.txt", args.max_text_chars)
    replay_steering_summary = read_text(replay_steering_dir / "steering_summary.txt", args.max_text_chars)
    best_layers = read_json(axis_dir / "best_layers.json")
    circuit_diagram_summary = read_text(circuit_diagram_dir / "medical_circuit_diagram_summary.txt", args.max_text_chars)

    prompt_count = count_csv_rows(Path("data/diabetes_contrastive_prompts.csv"))
    axis_best = best_layers.get("best_layer", "missing")
    candidates = best_layers.get("candidate_layers", "missing")
    steering_direction = extract_value(steering_summary, r"monotonicity: ([^\n]+)")
    patching_peak = extract_value(patching_summary, r"best_mean_normalized_score: ([^\n]+)")
    axis_sae_best = extract_value(axis_sae_summary, r"best_layer_by_top_feature_contribution: ([^\n]+)")

    md: list[str] = [
        f"# {args.title}",
        "",
        "## Setup",
        "",
        f"- Prompt rows: {prompt_count if prompt_count is not None else 'missing'}",
        f"- Best layer: {axis_best}",
        f"- Candidate layers: {candidates}",
        f"- Axis-aligned SAE best layer: {axis_sae_best or 'missing'}",
        f"- Steering monotonicity: {steering_direction or 'missing'}",
        f"- Patching peak normalized score: {patching_peak or 'missing'}",
        "",
        "## Axis Sweep",
        "",
        pre_md(axis_summary),
        "",
        md_image(axis_dir / "accuracy_by_layer.png", output_root, "Accuracy by layer"),
        "",
        md_image(axis_dir / "dla_and_logit_lens.png", output_root, "DLA and logit lens"),
        "",
        md_image(axis_dir / "concept_axis_pca.png", output_root, "Concept axis PCA"),
        "",
        "## Axis-Aligned SAE Feature Candidates",
        "",
        pre_md(axis_sae_summary),
        "",
        md_image(circuit_axis_dir / "axis_sae_contributions_by_layer.png", output_root, "Axis-aligned SAE contribution by layer"),
        "",
        md_image(circuit_axis_dir / "axis_sae_top_features.png", output_root, "Top axis-aligned SAE features"),
        "",
        csv_preview_markdown(
            circuit_axis_dir / "axis_sae_features.csv",
            [
                "layer",
                "rank",
                "feature_id",
                "axis_projection_contribution",
                "axis_projection_ci_low",
                "axis_projection_ci_high",
                "sign_consistency",
                "complication_stability",
                "template_stability",
                "activation_side",
                "decoder_side",
            ],
            sort_numeric_column="axis_projection_contribution",
            reverse=True,
        ),
        "",
        "## Circuit Diagram",
        "",
        pre_md(circuit_diagram_summary),
        "",
        md_image(circuit_diagram_dir / "medical_circuit_diagram.png", output_root, "Medical concept circuit diagram"),
        "",
        "## Steering",
        "",
        pre_md(steering_summary),
        "",
        md_image(steering_dir / "steering_curve.png", output_root, "Steering curve"),
        "",
        "## Activation Patching",
        "",
        pre_md(patching_summary),
        "",
        md_image(patching_dir / "patching_heatmap.png", output_root, "Patching heatmap"),
        "",
        csv_preview_markdown(
            patching_dir / "patching_results.csv",
            [
                "complication",
                "layer",
                "position",
                "clean_logit_diff_type1_minus_type2",
                "corrupt_logit_diff_type1_minus_type2",
                "patched_logit_diff_type1_minus_type2",
                "normalized_score",
            ],
            max_rows=10,
        ),
        "",
        "## 270M Replay",
        "",
        "### Axis",
        "",
        pre_md(replay_axis_summary),
        "",
        md_image(replay_axis_dir / "accuracy_by_layer.png", output_root, "270M accuracy by layer"),
        "",
        "### Steering",
        "",
        pre_md(replay_steering_summary),
        "",
        md_image(replay_steering_dir / "steering_curve.png", output_root, "270M steering curve"),
        "",
        "## Conclusions",
        "",
        "- Treat the result as phenomenon-level only if the axis peak, DLA/logit-lens alignment, steering slope, and patching peak agree.",
        "- SAE feature IDs are candidate mechanisms, not human-readable concepts by themselves.",
        "- The generated prompts and model outputs are interpretability artifacts, not medical guidance.",
        "",
    ]
    report_md.write_text("\n".join(md), encoding="utf-8")

    html_parts: list[str] = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(args.title)}</title>",
        "<style>",
        "body{font-family:Inter,Arial,sans-serif;line-height:1.55;margin:32px auto;max-width:1120px;padding:0 20px;color:#172033}",
        "h1,h2,h3{line-height:1.2} pre{background:#f6f8fa;border:1px solid #d8dee4;border-radius:6px;padding:12px;overflow:auto}",
        "table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #d8dee4;padding:6px 8px;text-align:left;vertical-align:top}",
        "figure{margin:20px 0} img{max-width:100%;height:auto;border:1px solid #d8dee4;border-radius:6px} figcaption{font-size:13px;color:#57606a;margin-top:6px}",
        ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:16px 0}.summary div{background:#f6f8fa;border:1px solid #d8dee4;border-radius:6px;padding:10px}",
        "</style></head><body>",
        f"<h1>{html.escape(args.title)}</h1>",
        "<h2>Setup</h2>",
        "<div class=\"summary\">",
        f"<div><strong>Prompt rows</strong><br>{prompt_count if prompt_count is not None else 'missing'}</div>",
        f"<div><strong>Best layer</strong><br>{html.escape(str(axis_best))}</div>",
        f"<div><strong>Candidate layers</strong><br>{html.escape(str(candidates))}</div>",
        f"<div><strong>Axis SAE best layer</strong><br>{html.escape(axis_sae_best or 'missing')}</div>",
        f"<div><strong>Steering monotonicity</strong><br>{html.escape(steering_direction or 'missing')}</div>",
        f"<div><strong>Patching peak</strong><br>{html.escape(patching_peak or 'missing')}</div>",
        "</div>",
        "<h2>Axis Sweep</h2>",
        pre_html(axis_summary),
        html_image(axis_dir / "accuracy_by_layer.png", "Accuracy by layer"),
        html_image(axis_dir / "dla_and_logit_lens.png", "DLA and logit lens"),
        html_image(axis_dir / "concept_axis_pca.png", "Concept axis PCA"),
        "<h2>Axis-Aligned SAE Feature Candidates</h2>",
        pre_html(axis_sae_summary),
        html_image(circuit_axis_dir / "axis_sae_contributions_by_layer.png", "Axis-aligned SAE contribution by layer"),
        html_image(circuit_axis_dir / "axis_sae_top_features.png", "Top axis-aligned SAE features"),
        csv_preview_html(
            circuit_axis_dir / "axis_sae_features.csv",
            [
                "layer",
                "rank",
                "feature_id",
                "axis_projection_contribution",
                "axis_projection_ci_low",
                "axis_projection_ci_high",
                "sign_consistency",
                "complication_stability",
                "template_stability",
                "activation_side",
                "decoder_side",
            ],
            sort_numeric_column="axis_projection_contribution",
            reverse=True,
        ),
        "<h2>Circuit Diagram</h2>",
        pre_html(circuit_diagram_summary),
        html_image(circuit_diagram_dir / "medical_circuit_diagram.png", "Medical concept circuit diagram"),
        "<h2>Steering</h2>",
        pre_html(steering_summary),
        html_image(steering_dir / "steering_curve.png", "Steering curve"),
        "<h2>Activation Patching</h2>",
        pre_html(patching_summary),
        html_image(patching_dir / "patching_heatmap.png", "Patching heatmap"),
        csv_preview_html(
            patching_dir / "patching_results.csv",
            [
                "complication",
                "layer",
                "position",
                "clean_logit_diff_type1_minus_type2",
                "corrupt_logit_diff_type1_minus_type2",
                "patched_logit_diff_type1_minus_type2",
                "normalized_score",
            ],
            max_rows=10,
        ),
        "<h2>270M Replay</h2><h3>Axis</h3>",
        pre_html(replay_axis_summary),
        html_image(replay_axis_dir / "accuracy_by_layer.png", "270M accuracy by layer"),
        "<h3>Steering</h3>",
        pre_html(replay_steering_summary),
        html_image(replay_steering_dir / "steering_curve.png", "270M steering curve"),
        "<h2>Conclusions</h2>",
        "<ul><li>Treat the result as phenomenon-level only if the axis peak, DLA/logit-lens alignment, steering slope, and patching peak agree.</li>",
        "<li>SAE feature IDs are candidate mechanisms, not human-readable concepts by themselves.</li>",
        "<li>The generated prompts and model outputs are interpretability artifacts, not medical guidance.</li></ul>",
        "</body></html>",
    ]
    report_html.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Wrote {report_md}")
    print(f"Wrote {report_html}")


if __name__ == "__main__":
    main()
