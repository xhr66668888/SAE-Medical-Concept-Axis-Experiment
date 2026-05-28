from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
