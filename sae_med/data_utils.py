from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_prompts(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_dicts(path)
    if not rows:
        raise ValueError(f"No prompt rows found in {path}")
    if "prompt" not in rows[0]:
        raise ValueError(f"CSV must contain a 'prompt' column: {path}")
    return rows


def balanced_take(rows: list[dict[str, str]], label_column: str, labels: tuple[str, str], max_per_label: int | None):
    selected: list[dict[str, str]] = []
    for label in labels:
        group = [row for row in rows if row.get(label_column) == label]
        if max_per_label is not None:
            group = group[:max_per_label]
        selected.extend(group)
    return selected
