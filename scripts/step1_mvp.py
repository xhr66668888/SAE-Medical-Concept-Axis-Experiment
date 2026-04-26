#!/usr/bin/env python3
"""Step 1 MVP: run two medical prompts through Gemma + one residual-stream SAE.

This is intentionally small: load one open model, capture one residual stream
activation with TransformerLens, encode it with one SAELens SAE, and print the
top active feature IDs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable


DEFAULT_PROMPTS = [
    "A patient diagnosed with Type 1 diabetes (ICD-10: E10) presents with kidney complications. The recommended ATC drug is",
    "A patient diagnosed with Type 2 diabetes (ICD-10: E11) presents with kidney complications. The recommended ATC drug is",
]


@dataclass(frozen=True)
class Preset:
    model_name: str
    sae_release: str
    sae_id: str
    hook_name: str | None = None


PRESETS = {
    "gemma3-270m-it-res9": Preset(
        model_name="google/gemma-3-270m-it",
        sae_release="gemma-scope-2-270m-it-res-all",
        sae_id="layer_9_width_16k_l0_small",
        hook_name="blocks.9.hook_resid_post",
    ),
    "gemma3-1b-it-res13": Preset(
        model_name="google/gemma-3-1b-it",
        sae_release="gemma-scope-2-1b-it-res-all",
        sae_id="layer_13_width_16k_l0_small",
        hook_name="blocks.13.hook_resid_post",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Gemma + one residual SAE, then print top active features for two prompts."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="gemma3-1b-it-res13",
        help="Known model/SAE pairing to use.",
    )
    parser.add_argument("--model-name", help="Override TransformerLens model name.")
    parser.add_argument("--sae-release", help="Override SAELens release ID.")
    parser.add_argument("--sae-id", help="Override SAELens SAE ID.")
    parser.add_argument(
        "--hook-name",
        help="Override hook name. If omitted, the script uses the SAE config hook_name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, xpu, cpu, or mps. Defaults to cuda, then xpu, then mps/cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Model load dtype. Defaults to bfloat16 on CUDA/XPU when supported, else float32.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of active SAE features to print per prompt.",
    )
    parser.add_argument(
        "--position",
        choices=["last"],
        default="last",
        help="Token position to inspect. Step 1 only uses the final token.",
    )
    parser.add_argument(
        "--no-prepend-bos",
        action="store_true",
        help="Disable TransformerLens BOS token prepending.",
    )
    return parser.parse_args()


def choose_device(torch_module, requested: str | None) -> str:
    if requested:
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module, "xpu", None) is not None and torch_module.xpu.is_available():
        return "xpu"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch_module, device: str, requested: str | None):
    if requested:
        return getattr(torch_module, requested)
    if device == "cuda" and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    if device == "xpu":
        return torch_module.bfloat16
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32


def first_parameter(module):
    return next(module.parameters())


def encode_with_sae(sae, activation):
    if not hasattr(sae, "encode"):
        raise AttributeError("The loaded SAE does not expose an encode(...) method.")
    return sae.encode(activation)


def load_sae_with_metadata(SAE, release: str, sae_id: str, device: str):
    loaded = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    if isinstance(loaded, tuple):
        return loaded

    cfg = getattr(loaded, "cfg", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    elif cfg is not None and hasattr(cfg, "__dict__"):
        cfg_dict = dict(cfg.__dict__)
    else:
        cfg_dict = {}
    return loaded, cfg_dict, getattr(loaded, "sparsity", None)


def resolve_config(args: argparse.Namespace) -> Preset:
    preset = PRESETS[args.preset]
    return Preset(
        model_name=args.model_name or preset.model_name,
        sae_release=args.sae_release or preset.sae_release,
        sae_id=args.sae_id or preset.sae_id,
        hook_name=args.hook_name or preset.hook_name,
    )


def print_header(config: Preset, device: str, dtype) -> None:
    print("=== Step 1 MVP configuration ===")
    print(f"model_name  : {config.model_name}")
    print(f"sae_release : {config.sae_release}")
    print(f"sae_id      : {config.sae_id}")
    print(f"device      : {device}")
    print(f"dtype       : {dtype}")
    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN    : not set (Gemma checkpoints may require an accepted HF license + token)")
    print()


def iter_prompts() -> Iterable[tuple[str, str]]:
    yield "Type 1 + kidney", DEFAULT_PROMPTS[0]
    yield "Type 2 + kidney", DEFAULT_PROMPTS[1]


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    try:
        import torch
        from sae_lens import SAE
        from transformer_lens import HookedTransformer
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt` in a Python 3.10-3.12 environment."
        ) from exc

    torch.set_grad_enabled(False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, (os.cpu_count() or 1) // 2)))
        except RuntimeError:
            pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_header(config, device, dtype)

    print("Loading HookedTransformer model...")
    loader = HookedTransformer.from_pretrained
    if str(dtype) != "torch.float32" and hasattr(HookedTransformer, "from_pretrained_no_processing"):
        loader = HookedTransformer.from_pretrained_no_processing
    model = loader(
        config.model_name,
        device=device,
        dtype=dtype,
        default_prepend_bos=not args.no_prepend_bos,
    )
    model.eval()

    print("Loading SAE...")
    sae, cfg_dict, sparsity = load_sae_with_metadata(SAE, config.sae_release, config.sae_id, device)
    sae.eval()

    hook_name = config.hook_name or cfg_dict.get("hook_name") or getattr(sae.cfg, "hook_name", None)
    if hook_name is None:
        raise RuntimeError("Could not infer hook_name from args or SAE config.")
    print(f"hook_name   : {hook_name}")
    if sparsity is not None:
        print("sparsity    : loaded")
    print()

    sae_param = first_parameter(sae)
    prepend_bos = not args.no_prepend_bos

    for label, prompt in iter_prompts():
        print(f"=== Prompt: {label} ===")
        print(prompt)

        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
        str_tokens = model.to_str_tokens(tokens[0])
        print(f"tokens shape: {tuple(tokens.shape)}")
        print(f"target token: {str_tokens[-1]!r} at position -1")

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == hook_name,
            return_type=None,
        )

        if hook_name not in cache:
            available = ", ".join(cache.keys())
            raise RuntimeError(f"Hook {hook_name!r} not found. Cached hooks: {available}")

        resid = cache[hook_name][:, -1, :]
        print(f"resid shape : {tuple(resid.shape)}")
        resid = resid.to(device=sae_param.device, dtype=sae_param.dtype)

        feature_acts = encode_with_sae(sae, resid)
        print(f"SAE acts    : {tuple(feature_acts.shape)}")

        top_values, top_indices = feature_acts.squeeze(0).topk(args.top_k)
        for rank, (feature_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
            print(f"top {rank}: feature_id={feature_id:<8} activation={value:.6f}")
        print()


if __name__ == "__main__":
    main()
