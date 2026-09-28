"""Unified hook-site runtime for Gemma-3 residual-stream work.

CANONICAL SITE (see docs/STAGE1_SPEC.md sec.7):
    resid_post(L) := output hidden state of decoder layer L,
                     i.e. what `model.model.language_model.layers[L]` returns, BEFORE final norm.

Capture, probing, SAE encode/decode, steering and patching ALL attach here, through the same
`_layer_hook` implementation. This removes the capture/intervention mismatch found in the audit
(`hidden_states[layer+1]` vs. decoder-layer forward hook, which disagree at the last layer because
Gemma3's last hidden state has the final RMSNorm applied).
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch


def _load_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for p in (".hf_env", os.path.expanduser("~/.hf_env")):
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return None


@dataclass
class Runtime:
    model_name: str
    model: object
    tokenizer: object
    device: str
    dtype: torch.dtype

    @property
    def layers(self):
        """Decoder layer list. Gemma3 nests it under a language_model; most other families do not."""
        m = self.model.model
        return m.language_model.layers if hasattr(m, "language_model") else m.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        cfg = self.model.config
        return getattr(cfg, "text_config", cfg).hidden_size


def load_runtime(model_name: str, device: str = "cuda:0", dtype=torch.bfloat16) -> Runtime:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _load_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map={"": device}, token=token
    )
    model.eval()
    model.config.use_cache = False
    return Runtime(model_name=model_name, model=model, tokenizer=tokenizer, device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
# hook plumbing
# --------------------------------------------------------------------------------------

def _split(output):
    """Decoder layers may return a tensor or a tuple; normalise."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rejoin(hidden, rest):
    return hidden if rest is None else (hidden, *rest)


@contextlib.contextmanager
def layer_hooks(rt: Runtime, fns: dict[int, callable]):
    """fns: {layer_index: fn(hidden)->hidden|None}. Attached at the canonical site."""
    handles = []

    def make(idx):
        def hook(module, inputs, output):
            hidden, rest = _split(output)
            new = fns[idx](hidden)
            if new is None:
                return output
            return _rejoin(new, rest)

        return hook

    try:
        for idx in fns:
            handles.append(rt.layers[idx].register_forward_hook(make(idx)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def forward_capture(
    rt: Runtime,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    capture_layers: list[int],
    positions: torch.Tensor | None = None,
    *,
    want_logits: bool = False,
    intervene: dict[int, callable] | None = None,
):
    """Single forward pass. Returns (captured, logits).

    captured: {layer: tensor}. If `positions` (LongTensor[batch]) is given, the captured tensor is
    [batch, d_model] gathered at those positions; otherwise [batch, seq, d_model].
    `intervene` maps layer -> fn(hidden)->hidden, applied at the SAME site, in the SAME pass.
    """
    store: dict[int, torch.Tensor] = {}
    fns: dict[int, callable] = {}

    def make_capture(idx, inner):
        def fn(hidden):
            out = inner(hidden) if inner is not None else None
            eff = hidden if out is None else out
            if positions is None:
                store[idx] = eff.detach().float().cpu()
            else:
                pos = positions.to(eff.device)
                store[idx] = eff[torch.arange(eff.shape[0], device=eff.device), pos].detach().float().cpu()
            return out
        return fn

    inter = intervene or {}
    for idx in set(capture_layers) | set(inter):
        inner = inter.get(idx)
        if idx in capture_layers:
            fns[idx] = make_capture(idx, inner)
        else:
            fns[idx] = inner

    with layer_hooks(rt, fns):
        out = rt.model(
            input_ids=input_ids.to(rt.device),
            attention_mask=attention_mask.to(rt.device),
            use_cache=False,
        )
    logits = out.logits.detach().float().cpu() if want_logits else None
    return store, logits


# --------------------------------------------------------------------------------------
# intervention builders (all operate at the canonical site)
# --------------------------------------------------------------------------------------

def _pos_ok(hidden, positions) -> bool:
    """During cached generation the decode passes have seq_len==1, so the prompt-relative read
    position no longer exists. Interventions are defined on the PROMPT pass only; skip otherwise."""
    return hidden.shape[1] > int(positions.max())

def add_direction(direction: torch.Tensor, delta: float, positions: torch.Tensor):
    """h[b, positions[b], :] += delta * direction   (direction must be unit-norm)."""
    def fn(hidden):
        if not _pos_ok(hidden, positions):
            return None
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        pos = positions.to(hidden.device)
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        new = hidden.clone()
        new[idx, pos] = new[idx, pos] + delta * d
        return new
    return fn


def add_direction_span(direction: torch.Tensor, delta: float, spans: torch.Tensor):
    """spans: BoolTensor[batch, seq]; add delta*direction at every True position."""
    def fn(hidden):
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        m = spans.to(hidden.device).unsqueeze(-1)
        return hidden + m * (delta * d)
    return fn


def replace_component(direction: torch.Tensor, new_proj: torch.Tensor, positions: torch.Tensor):
    """Replace only the component along `direction` at the given positions.
    new_proj: FloatTensor[batch] target value of h.d ."""
    def fn(hidden):
        if not _pos_ok(hidden, positions):
            return None
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        pos = positions.to(hidden.device)
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        cur = hidden[idx, pos]
        proj = (cur.float() @ d.float()).to(cur.dtype)
        tgt = new_proj.to(device=hidden.device, dtype=cur.dtype)
        new = hidden.clone()
        new[idx, pos] = cur + (tgt - proj).unsqueeze(-1) * d
        return new
    return fn


def patch_activation(donor: torch.Tensor, positions: torch.Tensor):
    """Full-residual patch: overwrite h[b, positions[b], :] with donor[b]."""
    def fn(hidden):
        if not _pos_ok(hidden, positions):
            return None
        pos = positions.to(hidden.device)
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        new = hidden.clone()
        new[idx, pos] = donor.to(device=hidden.device, dtype=hidden.dtype)
        return new
    return fn


def sae_edit(sae, edit_fn, positions: torch.Tensor):
    """Error-preserving SAE feature edit:  h' = h + W_dec(z' - z)  at the given positions.
    `edit_fn(z) -> z'` operates on the [batch, n_features] code."""
    def fn(hidden):
        if not _pos_ok(hidden, positions):
            return None
        pos = positions.to(hidden.device)
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        h = hidden[idx, pos].float()
        z = sae.encode(h)
        z2 = edit_fn(z)
        delta = sae.decode(z2) - sae.decode(z)
        new = hidden.clone()
        new[idx, pos] = (h + delta).to(hidden.dtype)
        return new
    return fn


def sae_reconstruct(sae, positions: torch.Tensor):
    """Replace h with the SAE reconstruction (the 'reconstruction-damage' control path)."""
    def fn(hidden):
        if not _pos_ok(hidden, positions):
            return None
        pos = positions.to(hidden.device)
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        h = hidden[idx, pos].float()
        new = hidden.clone()
        new[idx, pos] = sae.decode(sae.encode(h)).to(hidden.dtype)
        return new
    return fn


# --------------------------------------------------------------------------------------
# batching helper (LEFT padding, so right-relative positions are stable)
# --------------------------------------------------------------------------------------

def encode_batch(rt: Runtime, texts: list[str], rel_positions: list[int] | None = None):
    """Tokenize `texts` with left padding.

    `rel_positions[i]` is an index counted from the LEFT of the *unpadded* sequence i
    (negative values count from the right). Returns (input_ids, attention_mask, abs_positions).
    """
    tok = rt.tokenizer
    tok.padding_side = "left"
    enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    max_len = input_ids.shape[1]
    lens = attn.sum(-1).tolist()
    if rel_positions is None:
        abs_pos = torch.tensor([max_len - 1] * len(texts), dtype=torch.long)
    else:
        abs_pos = []
        for n, rp in zip(lens, rel_positions):
            idx = rp if rp >= 0 else n + rp
            if not (0 <= idx < n):
                raise IndexError(f"relative position {rp} outside unpadded length {n}")
            abs_pos.append(max_len - n + idx)
        abs_pos = torch.tensor(abs_pos, dtype=torch.long)
    return input_ids, attn, abs_pos


# --------------------------------------------------------------------------------------
# span helpers: an intervention confined to one token can be causally inert at late layers
# --------------------------------------------------------------------------------------

def span_mask(positions: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """BoolTensor[batch, seq] selecting where an intervention applies.

    'read_pos'    just the canonical read position
    'read_to_end' the read position through the end of the prompt (the state as it is carried
                  forward). Needed because at late layers a single non-final token has almost no
                  remaining causal influence on the final logits.
    'prompt_all'  every real (non-padding) token
    'last'        the final token only
    """
    b, s = attention_mask.shape
    ar = torch.arange(s, device=attention_mask.device).unsqueeze(0).expand(b, s)
    pos = positions.to(attention_mask.device).unsqueeze(1)
    real = attention_mask.bool()
    if mode == "read_pos":
        m = ar == pos
    elif mode == "read_to_end":
        m = ar >= pos
    elif mode == "prompt_all":
        m = torch.ones_like(real)
    elif mode == "last":
        m = ar == (s - 1)
    else:
        raise ValueError(f"unknown span mode {mode!r}")
    return m & real


def add_direction_mask(direction: torch.Tensor, delta: float, mask: torch.Tensor):
    def fn(hidden):
        if hidden.shape[1] != mask.shape[1]:
            return None  # decode pass during cached generation
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        m = mask.to(hidden.device).unsqueeze(-1)
        return hidden + m * (delta * d)
    return fn


def replace_component_mask(direction: torch.Tensor, target_proj: torch.Tensor, mask: torch.Tensor):
    """Set the component along `direction` to target_proj[b] at every masked position."""
    def fn(hidden):
        if hidden.shape[1] != mask.shape[1]:
            return None
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        m = mask.to(hidden.device)
        proj = hidden.float() @ d.float()                       # [b, s]
        tgt = target_proj.to(device=hidden.device).unsqueeze(1)  # [b, 1]
        delta = torch.where(m, (tgt - proj).to(hidden.dtype), torch.zeros_like(proj, dtype=hidden.dtype))
        return hidden + delta.unsqueeze(-1) * d
    return fn


def sae_edit_mask(sae, edit_fn, mask: torch.Tensor):
    """Error-preserving SAE feature edit at every masked position.

    `edit_fn(z, rows) -> z'` where `z` is [n_masked, n_features] and `rows` is the batch index of
    each masked position, so a per-item donor code can be broadcast across that item's span.
    """
    def fn(hidden):
        if hidden.shape[1] != mask.shape[1]:
            return None
        m = mask.to(hidden.device)
        idx = m.nonzero(as_tuple=True)
        if idx[0].numel() == 0:
            return None
        h = hidden[idx].float()
        z = sae.encode(h)
        delta = sae.decode(edit_fn(z, idx[0])) - sae.decode(z)
        new = hidden.clone()
        new[idx] = (h + delta).to(hidden.dtype)
        return new
    return fn


def sae_reconstruct_mask(sae, mask: torch.Tensor):
    def fn(hidden):
        if hidden.shape[1] != mask.shape[1]:
            return None
        m = mask.to(hidden.device)
        idx = m.nonzero(as_tuple=True)
        if idx[0].numel() == 0:
            return None
        h = hidden[idx].float()
        new = hidden.clone()
        new[idx] = sae.decode(sae.encode(h)).to(hidden.dtype)
        return new
    return fn
