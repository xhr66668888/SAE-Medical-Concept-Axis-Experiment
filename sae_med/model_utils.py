from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPreset:
    model_name: str
    sae_release: str
    sae_id_format: str
    hook_name_format: str
    default_layers: tuple[int, ...]

    def sae_id(self, layer: int) -> str:
        return self.sae_id_format.format(layer=layer)

    def hook_name(self, layer: int) -> str:
        return self.hook_name_format.format(layer=layer)


PRESETS: dict[str, ModelPreset] = {
    # Smallest Gemma Scope 2 route. Useful for CPU smoke tests or very tight
    # memory environments.
    "gemma3-270m-it-res": ModelPreset(
        model_name="google/gemma-3-270m-it",
        sae_release="gemma-scope-2-270m-it-res-all",
        sae_id_format="layer_{layer}_width_16k_l0_small",
        hook_name_format="blocks.{layer}.hook_resid_post",
        default_layers=(4, 9, 15),
    ),
    # Recommended default for this repo on Intel Lunar Lake / 16 GB unified
    # memory: real Gemma Scope 2, but small enough to run without CUDA.
    "gemma3-1b-it-res": ModelPreset(
        model_name="google/gemma-3-1b-it",
        sae_release="gemma-scope-2-1b-it-res-all",
        sae_id_format="layer_{layer}_width_16k_l0_small",
        hook_name_format="blocks.{layer}.hook_resid_post",
        default_layers=(7, 13, 17),
    ),
    # Maximum practical local target for this machine. Run only after the 1B
    # smoke test passes, and keep layers/prompts small.
    "gemma3-4b-it-res": ModelPreset(
        model_name="google/gemma-3-4b-it",
        sae_release="gemma-scope-2-4b-it-res-all",
        sae_id_format="layer_{layer}_width_16k_l0_small",
        hook_name_format="blocks.{layer}.hook_resid_post",
        default_layers=(9, 17, 29),
    ),
    # Gemma Scope for Gemma 2 2B. Kept as a fallback for the original prompt's
    # 2B note, but it is not Gemma Scope 2.
    "gemma2-2b-res": ModelPreset(
        model_name="gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id_format="layer_{layer}/width_16k/canonical",
        hook_name_format="blocks.{layer}.hook_resid_post",
        default_layers=(5, 12, 20),
    ),
}


def parse_layers(raw: str | None, default_layers: tuple[int, ...]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(default_layers)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def choose_device(torch_module: Any, requested: str | None) -> str:
    if requested:
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module, "xpu", None) is not None and torch_module.xpu.is_available():
        return "xpu"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch_module: Any, device: str, requested: str | None):
    if requested:
        return getattr(torch_module, requested)
    if device == "cuda" and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    if device == "xpu":
        return torch_module.bfloat16
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32


def empty_device_cache(torch_module: Any, device: str) -> None:
    if device == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    if device == "xpu" and getattr(torch_module, "xpu", None) is not None and torch_module.xpu.is_available():
        torch_module.xpu.empty_cache()


def require_runtime_deps():
    try:
        import torch
        from sae_lens import SAE
        from transformer_lens import HookedTransformer
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt` "
            "in a Python 3.10-3.12 environment."
        ) from exc
    return torch, SAE, HookedTransformer


def configure_torch(torch_module: Any) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch_module.set_grad_enabled(False)
    cpu_threads = min(8, os.cpu_count() or 1)
    torch_module.set_num_threads(cpu_threads)
    if hasattr(torch_module, "set_num_interop_threads"):
        try:
            torch_module.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
        except RuntimeError:
            pass
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")


def load_model(HookedTransformer: Any, model_name: str, device: str, dtype: Any, prepend_bos: bool):
    print("Loading HookedTransformer model...")
    loader = HookedTransformer.from_pretrained
    if str(dtype) != "torch.float32" and hasattr(HookedTransformer, "from_pretrained_no_processing"):
        loader = HookedTransformer.from_pretrained_no_processing
    model = loader(
        model_name,
        device=device,
        dtype=dtype,
        default_prepend_bos=prepend_bos,
    )
    model.eval()
    return model


def load_sae_with_metadata(SAE: Any, release: str, sae_id: str, device: str):
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


def infer_hook_name(sae: Any | None, cfg_dict: dict[str, Any] | None, fallback: str | None) -> str:
    hook_name = None
    if cfg_dict:
        hook_name = cfg_dict.get("hook_name")
    if hook_name is None and sae is not None and hasattr(sae, "cfg"):
        hook_name = getattr(sae.cfg, "hook_name", None)
    if hook_name is None:
        hook_name = fallback
    if hook_name is None:
        raise RuntimeError("Could not infer hook_name from SAE config or fallback.")
    return hook_name


def print_run_header(
    *,
    model_name: str,
    sae_release: str | None,
    device: str,
    dtype: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    print("=== Run configuration ===")
    print(f"model_name  : {model_name}")
    if sae_release is not None:
        print(f"sae_release : {sae_release}")
    print(f"device      : {device}")
    print(f"dtype       : {dtype}")
    if extra:
        for key, value in extra.items():
            print(f"{key:<12}: {value}")
    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN    : not set (Gemma checkpoints may require an accepted HF license + token)")
    print()


def first_parameter(module: Any):
    return next(module.parameters())


def token_position(model: Any, tokens: Any, position: str, keyword: str | None = None) -> tuple[int, str]:
    str_tokens = model.to_str_tokens(tokens[0])
    if position == "last":
        return len(str_tokens) - 1, str_tokens[-1]
    if position != "keyword":
        raise ValueError(f"Unsupported position: {position}")
    if keyword is None or keyword.strip() == "":
        raise ValueError("--keyword is required when --position keyword is used.")

    needle = keyword.lower()
    matches = [idx for idx, token in enumerate(str_tokens) if needle in token.lower()]
    if not matches:
        return len(str_tokens) - 1, str_tokens[-1]
    idx = matches[-1]
    return idx, str_tokens[idx]


def capture_resid_activation(
    *,
    model: Any,
    prompt: str,
    hook_name: str,
    device: str,
    prepend_bos: bool,
    position: str,
    keyword: str | None,
    debug: bool = False,
):
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    pos_idx, pos_token = token_position(model, tokens, position, keyword)
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name == hook_name,
        return_type=None,
    )
    if hook_name not in cache:
        available = ", ".join(cache.keys())
        raise RuntimeError(f"Hook {hook_name!r} not found. Cached hooks: {available}")
    resid = cache[hook_name][:, pos_idx, :].detach()
    if debug:
        print(f"prompt tokens shape: {tuple(tokens.shape)}")
        print(f"target token       : {pos_token!r} at index {pos_idx}")
        print(f"resid shape        : {tuple(resid.shape)}")
    return resid, pos_token, pos_idx


def cache_all_layer_residuals(
    *,
    model: Any,
    prompt: str,
    device: str,
    prepend_bos: bool,
    position: str,
    keyword: str | None,
    layers: list[int] | None = None,
    keep_full_sequence: bool = False,
    return_logits: bool = False,
):
    """Capture residual-stream activations at every (or a chosen subset of)
    `blocks.{layer}.hook_resid_post` hooks in a single forward pass.

    Returns a dict ``{layer: tensor}``. By default each tensor is the residual
    at the chosen target token position (shape ``[d_model]``), already moved to
    CPU and cast to float for safe downstream stacking. If ``keep_full_sequence``
    is True the tensors are full ``[seq_len, d_model]`` slices on CPU.

    Always returns the chosen ``(token, token_idx)`` and the full token list to
    let callers re-derive other positions cheaply. If ``return_logits`` is
    True, appends the forward logits as a fifth return value.
    """
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    pos_idx, pos_token = token_position(model, tokens, position, keyword)
    str_tokens = model.to_str_tokens(tokens[0])
    layer_set = set(layers) if layers is not None else None

    def filter_fn(name: str) -> bool:
        if not name.startswith("blocks.") or not name.endswith(".hook_resid_post"):
            return False
        if layer_set is None:
            return True
        try:
            layer_idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            return False
        return layer_idx in layer_set

    logits, cache = model.run_with_cache(
        tokens,
        names_filter=filter_fn,
        return_type="logits" if return_logits else None,
    )
    layer_to_tensor: dict[int, Any] = {}
    for hook_name, activation in cache.items():
        try:
            layer_idx = int(hook_name.split(".")[1])
        except (IndexError, ValueError):
            continue
        if keep_full_sequence:
            layer_to_tensor[layer_idx] = activation[0].detach().float().cpu()
        else:
            layer_to_tensor[layer_idx] = activation[0, pos_idx, :].detach().float().cpu()
    if layer_set is not None:
        missing_layers = sorted(layer_set - set(layer_to_tensor))
        if missing_layers:
            available = sorted(layer_to_tensor)
            raise RuntimeError(
                f"Did not capture requested residual layers {missing_layers}. "
                f"Captured layers: {available}"
            )
    if return_logits:
        return layer_to_tensor, pos_token, pos_idx, str_tokens, logits.detach().float().cpu()
    return layer_to_tensor, pos_token, pos_idx, str_tokens


def resolve_token_id(tokenizer: Any, candidate: str) -> int:
    """Return a single-token id for ``candidate``. Try the exact string first,
    then a stripped fallback (Gemma-style tokenizers usually need the leading
    space form to match a mid-sentence token)."""
    candidates = [candidate]
    stripped = candidate.lstrip()
    if stripped != candidate:
        candidates.append(stripped)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise SystemExit(
        f"Could not resolve a single-token id for {candidate!r}. "
        "Try a different surface form (with or without a leading space)."
    )


def direct_logit_attribution(model: Any, axis_unit: Any, token_a_id: int, token_b_id: int) -> float:
    """Project a unit-norm direction onto W_U[:, a] - W_U[:, b].

    Positive value = adding the axis increases the logit of token a relative to
    token b at the model's output head. This is the cheapest answer to "does
    the candidate axis directly write to the answer token?".
    """
    import torch  # local import keeps module importable without torch

    w_u = model.W_U.detach().float().cpu()  # [d_model, d_vocab]
    diff = w_u[:, token_a_id] - w_u[:, token_b_id]
    axis_cpu = axis_unit.detach().float().cpu()
    return float(torch.dot(axis_cpu, diff).item())


def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Returns ``(mean, lower, upper)`` where ``[lower, upper]`` is the
    ``(1 - alpha)`` percentile interval. Pure NumPy — no XPU cost."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if n <= 0:
        mean = float(arr.mean())
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lower, upper
