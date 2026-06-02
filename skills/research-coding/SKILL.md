---
name: research-coding
description: Use when editing research experiment Python code to match a mentor-style scientific scripting style: readable stage-based scripts, shape/rationale comments, direct console and figure outputs, and reliability checks without production-framework overengineering.
---

# Research Coding

## Workflow

- Read nearby `sample-code/*.py` or mentor-written scripts before editing.
- Identify the experiment stages before changing abstractions.
- Preserve scientific behavior, output files, split logic, seeds, and validation checks.

## Script Shape

- Start long scripts with a short module docstring: research question, data shape, method, and outputs.
- Organize long files with visible section banners.
- Prefer stage functions: load, normalize, fit/train, evaluate, plot, summarize.
- Keep plot-specific helpers local when they are not reused.

## Naming

- Use domain variables such as `X`, `X_norm`, `Z`, `axis_id`, `layer`, `patient_X`, and `healthy_X`.
- Use uppercase constants for fixed model names, channel slices, method lists, and default paths.
- Avoid generic framework names unless the object is genuinely framework-level.

## Comments

- Comment tensor shapes, code-system assumptions, sign conventions, split/leakage rules, and plot interpretation.
- Do not add comments that merely restate Python operations.
- Keep docstrings concise and shape-oriented.

## Outputs

- Print stage progress and compact scientific summaries.
- Use `Saved: ...` or `Wrote ...` consistently after artifacts are produced.
- Prefer PNG, CSV, JSON, and NPZ artifacts with human-readable names.

## Reliability

- Keep explicit checks for empty data, bad layers, missing dependencies, split leakage, invalid shapes, and unavailable model access.
- Keep tests for deterministic split logic, code-system parsing, and numeric helpers.
- Do not silently swallow errors except for known degenerate plotting or metric cases.

## Abstraction Limits

- Keep reusable IO, runtime hooks, and statistical helpers in modules when tested.
- Do not introduce unused config files, broad compatibility frameworks, or generic report prose.
- Do not remove a reliability guard only to imitate notebook style.

## Anti-Patterns

- Avoid production-style boilerplate in research scripts.
- Avoid unused configuration layers.
- Avoid generic claims unless tied to actual reported quantities.
- Avoid over-typed `dict[str, object]` plumbing in local script internals.
