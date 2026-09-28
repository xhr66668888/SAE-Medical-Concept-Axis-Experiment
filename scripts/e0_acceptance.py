#!/usr/bin/env python3
"""E0 implementation-acceptance checks (spec sec.7). Writes a machine-readable record."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import runtime as R
from bridge2026 import sae as S
from bridge2026.prompts import render_pure, render_readout, read_position
from bridge2026.schema import load_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sae-layers", default="9,17,22,29")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--families-dir", default="data/bridge2026/families")
    ap.add_argument("--out", default="runs/bridge2026/stage2/e0_acceptance.json")
    args = ap.parse_args()

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    n_layers = rt.n_layers
    rec = {"model": args.model, "dtype": args.dtype, "n_layers": n_layers,
           "d_model": rt.d_model, "checks": {}}
    print(f"model={args.model} n_layers={n_layers} d_model={rt.d_model}")

    fams = [load_family(p) for p in sorted(Path(args.families_dir).glob("*.json"))]
    items = [f.conditions[c] for f in fams
             for c in ("rep_unres", "rep_res", "norep_unres", "norep_res") if c in f.conditions]
    rec["n_items"] = len(items)
    print(f"acceptance set: {len(fams)} families / {len(items)} items")
    pure_texts = [render_pure(rt.tokenizer, it.turns) for it in items]
    read_texts = [render_readout(rt.tokenizer, it.turns, "W1", 0).text for it in items]

    # ---------------- E0.1 / E0.2  hook vs output_hidden_states ----------------
    pure_pos = [read_position(rt.tokenizer, it.turns) for it in items]
    ids, attn, pos = R.encode_batch(rt, pure_texts, pure_pos)
    probe_layers = sorted({0, 1, n_layers // 2, n_layers - 2, n_layers - 1})
    nb = min(8, ids.shape[0])
    store, _ = R.forward_capture(rt, ids[:nb], attn[:nb], probe_layers, pos[:nb])
    with torch.inference_mode():
        out = rt.model(input_ids=ids[:nb].to(rt.device), attention_mask=attn[:nb].to(rt.device),
                       output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    e01, e02 = {}, {}
    ar = torch.arange(nb, device=rt.device)
    for L in probe_layers:
        ref = hs[L + 1][ar, pos[:nb].to(rt.device)].detach().float().cpu()
        diff = (store[L] - ref).abs().max().item()
        entry = {"max_abs_diff": diff, "hook_norm": store[L].norm(dim=-1).mean().item(),
                 "hs_norm": ref.norm(dim=-1).mean().item()}
        (e02 if L == n_layers - 1 else e01)[str(L)] = entry
    rec["checks"]["E0.1_hook_equals_hidden_states"] = {
        "per_layer": e01,
        "pass": all(v["max_abs_diff"] == 0.0 for v in e01.values()),
    }
    last = e02[str(n_layers - 1)]
    rec["checks"]["E0.2_final_layer_diverges"] = {
        **last,
        "norm_ratio_hs_over_hook": last["hs_norm"] / max(last["hook_norm"], 1e-9),
        "pass": last["max_abs_diff"] > 0.0,
        "note": "hidden_states[n_layers] has the final RMSNorm applied; the canonical site does not.",
    }
    print("E0.1", rec["checks"]["E0.1_hook_equals_hidden_states"]["pass"],
          "E0.2", rec["checks"]["E0.2_final_layer_diverges"]["pass"],
          f"(norm ratio {rec['checks']['E0.2_final_layer_diverges']['norm_ratio_hs_over_hook']:.4f})")

    # ---------------- readout-side setup: steer at the PURE end position ----------------
    rids, rattn, rpos = R.encode_batch(rt, read_texts, pure_pos)  # same left-counted index
    L = min(22, n_layers - 2)

    def logits_of(intervene=None, bs=16):
        outs = []
        for i in range(0, rids.shape[0], bs):
            sl = slice(i, i + bs)
            iv = None
            if intervene is not None:
                iv = {k: v(sl) for k, v in intervene.items()}
            _, lg = R.forward_capture(rt, rids[sl], rattn[sl], [], None, want_logits=True, intervene=iv)
            outs.append(lg[:, -1, :])
        return torch.cat(outs, 0)

    clean = logits_of()

    # ---------------- E0.7 determinism ----------------
    again = logits_of()
    d = (clean - again).abs().max().item()
    rec["checks"]["E0.7_determinism"] = {"max_abs_diff": d, "pass": d == 0.0}

    # ---------------- E0.3 zero intervention ----------------
    direction = torch.randn(rt.d_model)
    direction = direction / direction.norm()
    z0 = logits_of({L: lambda sl: R.add_direction(direction, 0.0, rpos[sl])})
    d = (clean - z0).abs().max().item()
    rec["checks"]["E0.3_zero_dose_noop"] = {"layer": L, "max_abs_diff": d, "pass": d == 0.0}

    # ---------------- E0.4 self-patch ----------------
    parts = []
    for i in range(0, rids.shape[0], 16):
        sl = slice(i, i + 16)
        c, _ = R.forward_capture(rt, rids[sl], rattn[sl], [L], rpos[sl])
        parts.append(c[L])
    donor = torch.cat(parts, 0)
    sp = logits_of({L: lambda sl: R.patch_activation(donor[sl], rpos[sl])})
    d = (clean - sp).abs().max().item()
    rec["checks"]["E0.4_self_patch_noop"] = {"layer": L, "max_abs_diff": d, "pass": d == 0.0}

    # ---------------- E0.8 dose linearity at the hook ----------------
    alpha = 1.0
    sd = float(donor.float().matmul(direction).std().item()) or 1.0
    delta = alpha * sd
    parts = []
    for i in range(0, rids.shape[0], 16):
        sl = slice(i, i + 16)
        c, _ = R.forward_capture(rt, rids[sl], rattn[sl], [L], rpos[sl],
                                 intervene={L: R.add_direction(direction, delta, rpos[sl])})
        parts.append(c[L])
    got = (torch.cat(parts, 0) - donor).norm(dim=-1)
    want = torch.full_like(got, delta)
    err = (got - want).abs().max().item()
    rec["checks"]["E0.8_dose_linearity"] = {
        "layer": L, "alpha": alpha, "sd_proj": sd, "delta_norm_want": delta,
        "delta_norm_got_mean": got.mean().item(), "max_abs_err": err,
        "pert_over_resid": (got / donor.norm(dim=-1)).mean().item(),
        "rel_err": err / max(delta, 1e-9),
        "tol_rel": 0.05 if args.dtype == "bfloat16" else 1e-3,
        "pass": err / max(delta, 1e-9) < (0.05 if args.dtype == "bfloat16" else 1e-3),
        "note": ("achieved perturbation norm vs requested; the tolerance tracks the model dtype "
                 "because the addition happens in the model's own precision"),
    }

    # ---------------- E0.5 / E0.6 SAE ----------------
    sae_rows, e06 = {}, {}
    for lyr in [int(x) for x in args.sae_layers.split(",") if x]:
        if lyr >= n_layers:
            continue
        sae = S.load_sae(args.model, lyr, device=rt.device)
        acts_parts = []
        for i in range(0, rids.shape[0], 16):
            sl = slice(i, i + 16)
            cap_s, _ = R.forward_capture(rt, rids[sl], rattn[sl], [lyr], rpos[sl])
            acts_parts.append(cap_s[lyr])
        acts = torch.cat(acts_parts, 0).to(rt.device).float()
        st = S.reconstruction_stats(sae, acts)
        st["l0_cfg"] = sae.cfg["l0"]
        st["l0_ratio"] = st["l0_mean"] / max(sae.cfg["l0"], 1)
        st["sae_id"] = sae.sae_id
        st["fvu_valid"] = st["n"] >= 64  # FVU needs a real sample to estimate the mean
        sae_rows[str(lyr)] = st
        ident = logits_of({lyr: lambda sl: R.sae_edit(sae, lambda z: z, rpos[sl])})
        dd = (clean - ident).abs().max().item()
        e06[str(lyr)] = {"max_abs_diff": dd, "pass": dd == 0.0}
        del sae
        torch.cuda.empty_cache()
    rec["checks"]["E0.5_sae_reconstruction"] = {
        "per_layer": sae_rows,
        "pass": all(0.8 <= v["l0_ratio"] <= 1.25 and v["cos_mean"] > 0.95 for v in sae_rows.values()),
    }
    rec["checks"]["E0.6_sae_identity_path_noop"] = {
        "per_layer": e06, "pass": all(v["pass"] for v in e06.values())
    }

    # ---------------- E0.9 padding invariance / E0.10 PURE==READOUT at the read position ----------
    nb = min(16, len(items))
    solo = {L2: [] for L2 in probe_layers}
    for t, pp in list(zip(pure_texts, pure_pos))[:nb]:
        i2, a2, p2 = R.encode_batch(rt, [t], [pp])
        st2, _ = R.forward_capture(rt, i2, a2, probe_layers, p2)
        for L2 in probe_layers:
            solo[L2].append(st2[L2][0])
    pid, pat, pap = R.encode_batch(rt, pure_texts[:nb], pure_pos[:nb])
    batched, _ = R.forward_capture(rt, pid, pat, probe_layers, pap)
    e09 = {}
    for L2 in probe_layers:
        ss = torch.stack(solo[L2]); bb = batched[L2]
        e09[str(L2)] = {"max_rel_norm_diff": float(((bb - ss).norm(dim=-1) / ss.norm(dim=-1)).max())}
    rec["checks"]["E0.9_left_padding_invariance"] = {
        "per_layer": e09, "tol_rel": 1e-4,
        "pass": all(v["max_rel_norm_diff"] < 1e-4 for v in e09.values()),
        "note": "padded-batch vs unpadded-single capture; guards RoPE/attention-mask handling",
    }
    rd, _ = R.forward_capture(rt, rids[:nb], rattn[:nb], probe_layers, rpos[:nb])
    e10 = {}
    for L2 in probe_layers:
        aa, bb2 = batched[L2], rd[L2]
        e10[str(L2)] = {"max_rel_norm_diff": float(((aa - bb2).norm(dim=-1) / aa.norm(dim=-1)).max()),
                        "resid_norm": float(aa.norm(dim=-1).mean())}
    rec["checks"]["E0.10_pure_equals_readout_at_read_pos"] = {
        "per_layer": e10, "tol_rel": 1e-4,
        "pass": all(v["max_rel_norm_diff"] < 1e-4 for v in e10.values()),
        "note": ("causal attention + shared token prefix means a direction fitted on the PURE "
                 "rendering applies unchanged inside the READOUT rendering"),
    }

    rec["all_pass"] = all(v.get("pass") for v in rec["checks"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    for k, v in rec["checks"].items():
        print(f"  {'PASS' if v.get('pass') else 'FAIL'}  {k}")
    print(f"ALL_PASS={rec['all_pass']}  ->  {args.out}")
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
