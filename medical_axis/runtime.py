from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def require_torch_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}. Install the experiment environment with "
            "`python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def configure_runtime(torch: Any, *, threads: int | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    chosen_threads = threads or min(16, os.cpu_count() or 1)
    torch.set_grad_enabled(False)
    torch.set_num_threads(chosen_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, chosen_threads // 2)))
        except RuntimeError:
            pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def choose_device(torch: Any, requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch: Any, device: str, requested: str | None = None):
    if requested:
        return getattr(torch, requested)
    if device in {"cuda", "xpu"}:
        return torch.bfloat16
    return torch.float32


def load_causal_lm(
    model_name: str,
    *,
    device: str,
    dtype: Any,
    cache_dir: str | Path | None = None,
):
    torch, AutoModelForCausalLM, AutoTokenizer = require_torch_transformers()
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        token=token,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def locate_decoder_layers(model: Any):
    candidates = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        obj = model
        ok = True
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate decoder layers for forward hooks.")


def tokenized_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def label_logprob(model: Any, tokenizer: Any, prompt: str, label: str, *, device: str) -> float:
    torch, _, _ = require_torch_transformers()
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    label_ids = tokenizer(" " + label.strip(), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([prompt_ids, label_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    start = int(prompt_ids.numel()) - 1
    total = 0.0
    for offset, token_id in enumerate(label_ids.tolist()):
        log_probs = torch.log_softmax(logits[start + offset].float(), dim=-1)
        total += float(log_probs[token_id].item())
    return total


def label_logprob_diff(model: Any, tokenizer: Any, prompt: str, positive_label: str, negative_label: str, *, device: str) -> float:
    return label_logprob(model, tokenizer, prompt, positive_label, device=device) - label_logprob(
        model,
        tokenizer,
        prompt,
        negative_label,
        device=device,
    )


def capture_hidden_vector(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer: int,
    device: str,
    position: int = -1,
):
    torch, _, _ = require_torch_transformers()
    tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    hidden_states = output.hidden_states
    if layer < 0 or layer + 1 >= len(hidden_states):
        raise ValueError(f"Layer {layer} is outside available hidden states 0..{len(hidden_states) - 2}.")
    vector = hidden_states[layer + 1][0, position, :].detach().float().cpu()
    return vector


def capture_layer_matrix(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layers: list[int],
    device: str,
    position: int = -1,
) -> dict[int, Any]:
    torch, _, _ = require_torch_transformers()
    tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    hidden_states = output.hidden_states
    result = {}
    for layer in layers:
        if layer < 0 or layer + 1 >= len(hidden_states):
            raise ValueError(f"Layer {layer} is outside available hidden states 0..{len(hidden_states) - 2}.")
        result[layer] = hidden_states[layer + 1][0, position, :].detach().float().cpu()
    return result


class ResidualSteeringHook:
    def __init__(self, direction, alpha: float, *, positions: str = "all"):
        self.direction = direction
        self.alpha = float(alpha)
        self.positions = positions

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = self.alpha * self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if self.positions == "all":
            patched_hidden = hidden + delta
        elif self.positions == "last":
            patched_hidden = hidden.clone()
            patched_hidden[:, -1, :] = patched_hidden[:, -1, :] + delta
        else:
            raise ValueError(f"Unsupported steering position mode: {self.positions}")
        if isinstance(output, tuple):
            return (patched_hidden, *output[1:])
        return patched_hidden


class ResidualPatchHook:
    def __init__(self, replacement, *, position: int = -1):
        self.replacement = replacement
        self.position = position

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched_hidden = hidden.clone()
        replacement = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        patched_hidden[:, self.position, :] = replacement
        if isinstance(output, tuple):
            return (patched_hidden, *output[1:])
        return patched_hidden
