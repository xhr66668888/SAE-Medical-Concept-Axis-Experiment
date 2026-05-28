#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_axis.reporting import build_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Markdown experiment report.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--report", default="report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_markdown_report(Path(args.output_dir), Path(args.report))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
