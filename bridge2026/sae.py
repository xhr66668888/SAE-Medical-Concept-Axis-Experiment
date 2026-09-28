"""Gemma Scope 2 JumpReLU SAE loader.

The released configs declare `hf_hook_point_in = "model.layers.<L>.output"`, i.e. the decoder-layer
output. That is exactly the canonical site in bridge2026/runtime.py, so SAE codes, steering and
patching all live in the same activation space. `affine_connection: false` for the resid_post
releases used here, so no input rescaling is needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = {
    "google/gemma-3-4b-it": "google/gemma-scope-2-4b-it",
    "google/gemma-3-12b-it": "google/gemma-scope-2-12b-it",
}


@dataclass
class JumpReluSAE:
    w_enc: torch.Tensor      # [d_model, n_features]
    b_enc: torch.Tensor      # [n_features]
    w_dec: torch.Tensor      # [n_features, d_model]
    b_dec: torch.Tensor      # [d_model]
    threshold: torch.Tensor  # [n_features]
    cfg: dict
    sae_id: str

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[1]

    @property
    def d_model(self) -> int:
        return self.w_enc.shape[0]

    def to(self, device, dtype=torch.float32):
        for name in ("w_enc", "b_enc", "w_dec", "b_dec", "threshold"):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        return self

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = h @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def encode_pre(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.w_enc + self.b_enc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w_dec + self.b_dec

    def decoder_dirs(self) -> torch.Tensor:
        """Unit-norm decoder directions, [n_features, d_model]."""
        return self.w_dec / self.w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def load_sae(model_name: str, layer: int, width: str = "16k", l0: str = "medium",
             site: str = "resid_post", device: str = "cpu") -> JumpReluSAE:
    repo = REPO[model_name]
    sae_id = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
    cfg = json.load(open(hf_hub_download(repo, f"{sae_id}/config.json")))
    params = load_file(hf_hub_download(repo, f"{sae_id}/params.safetensors"))
    sae = JumpReluSAE(
        w_enc=params["w_enc"], b_enc=params["b_enc"], w_dec=params["w_dec"],
        b_dec=params["b_dec"], threshold=params["threshold"], cfg=cfg, sae_id=f"{repo}/{sae_id}",
    )
    assert cfg.get("affine_connection") is False, f"unexpected affine_connection in {sae_id}"
    assert cfg["architecture"] == "jump_relu", f"unexpected architecture in {sae_id}"
    return sae.to(device)


def reconstruction_stats(sae: JumpReluSAE, acts: torch.Tensor) -> dict:
    """acts: [n, d_model] float32 on the SAE's device."""
    z = sae.encode(acts)
    rec = sae.decode(z)
    resid = acts - rec
    var = (acts - acts.mean(0, keepdim=True)).pow(2).sum()
    fvu = (resid.pow(2).sum() / var.clamp_min(1e-9)).item()
    return {
        "fvu": fvu,
        "l0_mean": (z > 0).float().sum(-1).mean().item(),
        "cos_mean": torch.nn.functional.cosine_similarity(acts, rec, dim=-1).mean().item(),
        "rel_err": (resid.norm(dim=-1) / acts.norm(dim=-1).clamp_min(1e-9)).mean().item(),
        "n": int(acts.shape[0]),
    }
