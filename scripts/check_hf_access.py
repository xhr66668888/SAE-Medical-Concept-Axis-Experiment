#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hugging Face access before model-heavy experiment stages.")
    parser.add_argument("--model-name", default="google/gemma-3-1b-it")
    parser.add_argument("--filename", default="config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Add it to .env or export it before running the experiment."
        )

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit("huggingface_hub is not installed. Install requirements first.") from exc

    try:
        user = HfApi(token=token).whoami()
        name = user.get("name") or user.get("fullname") or "authenticated"
        print(f"HF account: {name}")
    except Exception as exc:
        raise SystemExit(f"HF_TOKEN is not valid: {exc}") from exc

    try:
        path = hf_hub_download(args.model_name, args.filename, token=token)
    except Exception as exc:
        message = str(exc)
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            message = f"{message}\n{cause}"
        if "public gated repositories" in message:
            raise SystemExit(
                "HF token is authenticated, but it cannot download this gated model.\n"
                "Fix the token on Hugging Face: token settings -> enable access to public gated repositories.\n"
                f"Model checked: {args.model_name}"
            ) from exc
        if "gated repo" in message.lower() or "403" in message or "401" in message:
            raise SystemExit(
                "HF token cannot access the gated model files. Confirm that the account accepted the Gemma license "
                "and that the token has read access to this repository.\n"
                f"Model checked: {args.model_name}"
            ) from exc
        raise SystemExit(f"Could not download {args.model_name}/{args.filename}: {exc}") from exc

    print(f"HF gated-model access OK: {args.model_name}/{args.filename}")
    print(f"Cached file: {path}")


if __name__ == "__main__":
    main()
