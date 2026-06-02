from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output) as tmp_path:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def csv_row_key(row: dict[str, object], key_fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in key_fields)


def existing_csv_keys(path: str | Path, key_fields: list[str]) -> set[tuple[str, ...]]:
    input_path = Path(path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        return set()
    return {csv_row_key(row, key_fields) for row in read_csv(input_path)}


def _csv_header(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def ensure_csv_fieldnames(path: str | Path, fieldnames: list[str]) -> None:
    output = Path(path)
    header = _csv_header(output)
    if not header or header == fieldnames:
        return
    rows = read_csv(output)
    write_csv(output, rows, fieldnames)


def append_csv_rows(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    ensure_csv_fieldnames(output, fieldnames)
    needs_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output) as tmp_path:
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
