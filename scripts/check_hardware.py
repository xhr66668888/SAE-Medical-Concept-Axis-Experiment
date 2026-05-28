#!/usr/bin/env python3
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


def meminfo_gib() -> tuple[float, float]:
    raw = Path("/proc/meminfo").read_text(encoding="utf-8")
    values = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) / 1024 / 1024
    return values.get("MemTotal", 0.0), values.get("MemAvailable", 0.0)


def torch_report() -> list[str]:
    try:
        import torch
    except ModuleNotFoundError:
        return ["torch: not installed"]
    lines = [f"torch: {torch.__version__}"]
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda device: {torch.cuda.get_device_name(0)}")
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        lines.append("xpu available: torch.xpu missing")
    else:
        try:
            available = xpu.is_available()
            lines.append(f"xpu available: {available}")
            if available:
                lines.append(f"xpu device: {xpu.get_device_name(0)}")
        except Exception as exc:
            lines.append(f"xpu available: error: {exc}")
    return lines


def main() -> None:
    total, available = meminfo_gib()
    print("=== System ===")
    print(f"host: {platform.node()}")
    print(f"os: {platform.platform()}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print()

    print("=== CPU / Memory ===")
    for line in run(["lscpu"]).splitlines():
        if line.startswith(("Model name:", "CPU(s):", "Thread(s) per core:", "Core(s) per socket:")):
            print(line)
    print(f"memory total: {total:.1f} GiB")
    print(f"memory available: {available:.1f} GiB")
    print()

    print("=== Accelerators ===")
    gpu_lines = [
        line
        for line in run(["lspci", "-nn"]).splitlines()
        if any(term in line.lower() for term in ("vga", "3d", "display", "graphics", "nvidia", "amd"))
    ]
    print("\n".join(gpu_lines) if gpu_lines else "no PCI GPU lines found")
    print(f"/dev/dri: {', '.join(path.name for path in Path('/dev/dri').glob('*')) if Path('/dev/dri').exists() else 'missing'}")
    print(f"user groups: {run(['id', '-nG'])}")
    print()

    print("=== Python ML Stack ===")
    for line in torch_report():
        print(line)
    for package in ("transformers", "sae_lens", "sklearn", "numpy", "matplotlib"):
        try:
            module = __import__(package)
            print(f"{package}: {getattr(module, '__version__', 'installed')}")
        except ModuleNotFoundError:
            print(f"{package}: not installed")
    print()

    print("=== Recommendation ===")
    print("Use the CPU profile on this machine unless CUDA/XPU becomes available to PyTorch.")
    print("Main local run: google/gemma-3-1b-it, dtype=float32, device=cpu, threads=16.")
    if shutil.which("python3.11"):
        print("Suggested environment: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


if __name__ == "__main__":
    main()
