#!/usr/bin/env python3
"""Print local hardware/software facts and a Gemma Scope 2 run recommendation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return ""
    return (completed.stdout or completed.stderr).strip()


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def meminfo_gib() -> tuple[float, float]:
    raw = read("/proc/meminfo")
    values = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) / 1024 / 1024
    return values.get("MemTotal", 0.0), values.get("MemAvailable", 0.0)


def current_groups() -> set[str]:
    output = run(["id", "-nG"])
    return set(output.split()) if output else set()


def torch_status() -> list[str]:
    try:
        import torch
    except ModuleNotFoundError:
        return ["torch: not installed"]

    lines = [f"torch: {torch.__version__}"]
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        lines.append("xpu available: torch.xpu missing")
    else:
        try:
            available = xpu.is_available()
        except Exception as exc:  # pragma: no cover - diagnostic only
            lines.append(f"xpu available: error: {exc}")
        else:
            lines.append(f"xpu available: {available}")
            if available:
                lines.append(f"xpu device count: {xpu.device_count()}")
                try:
                    lines.append(f"xpu device name: {xpu.get_device_name(0)}")
                except Exception as exc:  # pragma: no cover - diagnostic only
                    lines.append(f"xpu device name: error: {exc}")
    return lines


def main() -> None:
    total_gib, available_gib = meminfo_gib()
    groups = current_groups()
    gpu_lines = run(["lspci", "-nn"]).splitlines()
    gpu_lines = [line for line in gpu_lines if any(term in line.lower() for term in ("vga", "3d", "display", "graphics"))]
    dri = sorted(path.name for path in Path("/dev/dri").glob("*")) if Path("/dev/dri").exists() else []
    commands = ["sycl-ls", "clinfo", "intel_gpu_top", "xpu-smi", "pip3", "conda", "uv"]
    command_status = {command: shutil.which(command) or "missing" for command in commands}

    print("=== System ===")
    print(f"hostname: {platform.node()}")
    print(f"kernel: {platform.release()}")
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print()

    print("=== CPU / Memory ===")
    print(first_line(run(["lscpu"]) or "lscpu unavailable"))
    model_name = ""
    for line in run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            model_name = line.split(":", 1)[1].strip()
            break
    if model_name:
        print(f"cpu: {model_name}")
    print(f"memory total: {total_gib:.1f} GiB")
    print(f"memory available: {available_gib:.1f} GiB")
    print()

    print("=== Intel GPU Access ===")
    for line in gpu_lines:
        print(line)
    print(f"/dev/dri: {', '.join(dri) if dri else 'not found'}")
    print(f"user groups: {' '.join(sorted(groups))}")
    if "render" not in groups or "video" not in groups:
        print("warning: current user is not in render/video; Intel GPU access may fail.")
        print("fix: sudo usermod -aG render,video $USER && reboot")
    print()

    print("=== Tools ===")
    for command, status in command_status.items():
        print(f"{command}: {status}")
    print()

    print("=== PyTorch ===")
    for line in torch_status():
        print(line)
    print()

    print("=== Recommendation ===")
    print("best default: --preset gemma3-1b-it-res --device xpu --dtype bfloat16")
    print("max local attempt: --preset gemma3-4b-it-res with 1 prompt/group and 1-3 layers")
    print("avoid locally: Gemma 3 12B/27B with TransformerLens + SAEs on 15 GiB memory")
    if "render" not in groups or "video" not in groups:
        print("do first: add this user to render/video groups before trying --device xpu")
    if command_status["pip3"] == "missing" and command_status["conda"] == "missing" and command_status["uv"] == "missing":
        print("do first: install/use a Python 3.11 env with pip, e.g. conda/micromamba")
    print()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


if __name__ == "__main__":
    main()
