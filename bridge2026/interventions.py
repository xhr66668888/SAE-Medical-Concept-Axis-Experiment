"""Intervention arms for E3 (causal selectivity).

Every arm acts at the canonical hook site, at the canonical read position, inside the READOUT
forward pass. Dose is calibrated in TRAIN projection standard deviations:
        h' = h + alpha * sd_train(h.d) * d
so alphas are comparable across layers (the old code's raw alpha=6 was not).

Controls are deliberate and matched, because the plan's claim is SELECTIVITY, not just "an effect":
  random_dir        - equal perturbation norm, random direction
  repetition_axis   - equal perturbation norm, the competing surface-repetition direction
  sae_random        - same feature count, matched activation frequency and magnitude
  sae_recon         - SAE reconstruction with no feature edit (isolates reconstruction damage)
  full_patch        - whole-residual donor patch (coarse; reported as a control, not as evidence)
"""
from __future__ import annotations

import numpy as np
import torch

from bridge2026 import runtime as R


class Intervention:
    """Callable(slice, abs_positions, attention_mask) -> {layer: hook_fn}, plus a record.

    `span` selects which token positions the intervention touches. E0.11 showed that the
    single-token 'read_pos' site is causally inert for this readout at every candidate layer
    (|delta (A-B) margin| <= 0.006 nats), so the default span is 'read_to_end'.
    """

    def __init__(self, name, layer, fn_factory, meta=None, span="read_to_end"):
        self.name = name
        self.layer = layer
        self._f = fn_factory
        self.span = span
        self.meta = dict(meta or {})
        self.meta.setdefault("span", span)

    def mask(self, pos, attn):
        return R.span_mask(pos, attn, self.span)

    def __call__(self, sl, pos, attn):
        f = self._f(sl, pos, attn, self)
        return {self.layer: f} if f is not None else None


def clean(layer=0):
    return Intervention("clean", layer, lambda sl, pos, attn, iv: None)


def dense_axis(layer, direction, alpha, sd_train, span="read_to_end"):
    d = torch.as_tensor(direction, dtype=torch.float32)
    delta = float(alpha) * float(sd_train)
    return Intervention(
        "dense_axis", layer,
        lambda sl, pos, attn, iv: R.add_direction_mask(d, delta, iv.mask(pos, attn)),
        {"alpha": alpha, "sd_train": sd_train, "delta_norm": abs(delta)}, span=span,
    )


def random_dir(layer, d_model, alpha, sd_train, seed, span="read_to_end"):
    g = np.random.default_rng(seed)
    v = g.standard_normal(d_model)
    v = v / np.linalg.norm(v)
    d = torch.as_tensor(v, dtype=torch.float32)
    delta = float(alpha) * float(sd_train)
    return Intervention(
        "random_dir", layer,
        lambda sl, pos, attn, iv: R.add_direction_mask(d, delta, iv.mask(pos, attn)),
        {"alpha": alpha, "sd_train": sd_train, "delta_norm": abs(delta), "seed": seed}, span=span,
    )


def other_axis(name, layer, direction, alpha, sd_match, span="read_to_end"):
    """Steer along a competing direction with the SAME perturbation norm as the target arm."""
    d = torch.as_tensor(direction, dtype=torch.float32)
    delta = float(alpha) * float(sd_match)
    return Intervention(
        name, layer,
        lambda sl, pos, attn, iv: R.add_direction_mask(d, delta, iv.mask(pos, attn)),
        {"alpha": alpha, "delta_norm": abs(delta)}, span=span,
    )


def component_swap(layer, direction, donor_proj, sign=+1, span="read_to_end"):
    """Replace ONLY the component along `direction` with a matched donor's value.

    This is the main causal probe: the rest of the residual is untouched, so an effect cannot be
    explained by having perturbed the representation in general.
    `donor_proj`: FloatTensor[n_rows] aligned with the row order handed to score_rows.
    """
    d = torch.as_tensor(direction, dtype=torch.float32)
    dp = torch.as_tensor(donor_proj, dtype=torch.float32)
    return Intervention(
        "component_swap", layer,
        lambda sl, pos, attn, iv: R.replace_component_mask(d, dp[sl], iv.mask(pos, attn)),
        {"direction": "unresolved_axis", "sign": sign}, span=span,
    )


def full_patch(layer, donor_acts):
    """Whole-residual patch. Only defined at the single read position, so it is reported as a
    coarse control and its causal reach is limited (see E0.11)."""
    da = torch.as_tensor(donor_acts, dtype=torch.float32)
    return Intervention("full_patch", layer,
                        lambda sl, pos, attn, iv: R.patch_activation(da[sl], pos),
                        span="read_pos")


def sae_feature_edit(name, layer, sae, features, mode, donor_codes=None, scale=0.0,
                     span="read_to_end"):
    """mode:
       'ablate'  z[f] = 0
       'donor'   z[f] = donor_codes[row, f]   (matched opposite-label donor)
       'amplify' z[f] = z[f] + scale * (per-feature train sd, supplied via `scale` already scaled)
    Uses the error-preserving path h' = h + W_dec(z'-z), so the SAE's own reconstruction error is
    carried through unchanged and cannot masquerade as a feature effect.
    """
    idx = torch.as_tensor(np.asarray(features), dtype=torch.long)
    dc = None if donor_codes is None else torch.as_tensor(donor_codes, dtype=torch.float32)

    def factory(sl, pos, attn, iv):
        cols = idx.to(sae.w_enc.device)

        donor = None if dc is None else dc[sl]

        def edit(z, rows=None):
            z2 = z.clone()
            if mode == "ablate":
                z2[:, cols] = 0.0
            elif mode == "donor":
                d = donor.to(z.device)
                # one donor code per ITEM, broadcast across that item's masked span
                z2[:, cols] = d[rows] if rows is not None else d[:, :len(cols)]
            elif mode == "amplify":
                z2[:, cols] = z2[:, cols] + float(scale)
            else:
                raise ValueError(mode)
            return z2

        return R.sae_edit_mask(sae, edit, iv.mask(pos, attn))

    return Intervention(name, layer, factory,
                        {"mode": mode, "n_features": len(features),
                         "features": [int(f) for f in features], "sae_id": sae.sae_id}, span=span)


def sae_reconstruct(layer, sae, span="read_to_end"):
    return Intervention("sae_recon", layer,
                        lambda sl, pos, attn, iv: R.sae_reconstruct_mask(sae, iv.mask(pos, attn)),
                        {"sae_id": sae.sae_id}, span=span)


def match_random_features(sae, codes, target_features, seed, tol=0.25):
    """Pick the same number of features, matched on activation frequency and mean magnitude.
    `codes`: [n, n_features] train SAE codes."""
    g = np.random.default_rng(seed)
    rate = (codes > 0).mean(0)
    mag = np.where(rate > 0, codes.sum(0) / np.maximum((codes > 0).sum(0), 1), 0.0)
    chosen, used = [], set(int(f) for f in target_features)
    for f in target_features:
        f = int(f)
        ok = np.flatnonzero(
            (np.abs(rate - rate[f]) <= tol * max(rate[f], 1e-6))
            & (np.abs(mag - mag[f]) <= tol * max(abs(mag[f]), 1e-6))
        )
        ok = [int(c) for c in ok if int(c) not in used]
        if not ok:  # fall back to nearest activation rate
            cand = np.argsort(np.abs(rate - rate[f]))
            ok = [int(c) for c in cand if int(c) not in used][:20]
        pick = int(g.choice(ok))
        chosen.append(pick)
        used.add(pick)
    return chosen
