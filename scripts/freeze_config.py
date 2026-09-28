#!/usr/bin/env python3
"""Freeze the intervention configuration from DEV results.

After this runs, layer / features / dose / position are fixed. Stage 3 and Stage 4 read this file
and may not deviate from it. The file records a hash of the dev evidence it was derived from, so a
later reader can check that nothing was re-tuned after the test was read.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/bridge2026/stage2/e2_layer_sweep_dev.csv")
    ap.add_argument("--sae-features", default="runs/bridge2026/stage2/e2_sae_features_dev.csv")
    ap.add_argument("--acts", default="runs/bridge2026/cache/acts_4b.npz")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--selection-metric", default="paired_acc", choices=["paired_acc", "auroc"])
    ap.add_argument("--candidate-layers", default="", help="restrict to layers with a released SAE")
    ap.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--control-alphas", default="-1,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sweep)))
    feats = list(csv.DictReader(open(args.sae_features))) if Path(args.sae_features).exists() else []

    cand = None
    if args.candidate_layers:
        cand = {int(x) for x in args.candidate_layers.split(",") if x}

    # 1) choose the SAE k on dev, among the sae_topk rows
    sae_rows = [r for r in rows if r["method"].startswith("sae_topk")]
    if cand:
        sae_rows = [r for r in sae_rows if int(r["layer"]) in cand]
    # Deterministic, documented tie-break. SAE paired accuracy first, then SAE AUROC, then the
    # DENSE readout at the same layer. Ties on the headline metric were common (L22 and L29 both
    # reached 0.680), and `max` would otherwise have resolved them by file order.
    dense_by_layer = {int(r["layer"]): r for r in rows if r["method"] == "mean_diff"}

    def sae_key(r):
        L = int(r["layer"])
        d = dense_by_layer.get(L, {})
        return (round(float(r[args.selection_metric] or 0), 6),
                round(float(r.get("auroc") or 0), 6),
                round(float(d.get("paired_acc") or 0), 6),
                round(float(d.get("auroc") or 0), 6))

    best_sae = max(sae_rows, key=sae_key) if sae_rows else None

    # 2) choose the intervention layer: the SAE-bearing layer with the best dev readout
    dense_rows = [r for r in rows if r["method"] == "mean_diff"]
    if cand:
        dense_rows = [r for r in dense_rows if int(r["layer"]) in cand]
    best_dense = max(dense_rows, key=lambda r: float(r[args.selection_metric] or 0))
    layer = int(best_sae["layer"]) if best_sae else int(best_dense["layer"])

    sd_train = float(next(r for r in dense_rows if int(r["layer"]) == layer)["sd_train_proj"])
    k = int(best_sae["method"].replace("sae_topk", "")) if best_sae else 0
    sae_feats = json.loads(best_sae["features"]) if best_sae else []

    layer_feats = [f for f in feats if int(f["layer"]) == layer][:k]
    cfg = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "layer": layer,
        "hook_site": "resid_post(L) = output of model.model.language_model.layers[L], pre final-norm",
        "read_position": "last token of the final user turn's own text (shared prefix of PURE/GEN/READOUT)",
        "sd_train": sd_train,
        "alphas": [float(x) for x in args.alphas.split(",")],
        "control_alphas": [float(x) for x in args.control_alphas.split(",")],
        "component_swap": True,
        "full_patch": True,
        "selection_metric": args.selection_metric,
        "selected_on": "dev",
        "dense_dev_at_selected_layer": {m: float(dense_by_layer[layer][m])
                                        for m in ("auroc", "paired_acc", "perm_p", "cos_unres_rep")
                                        if dense_by_layer.get(layer, {}).get(m)},
        "dense_best_layer_overall": {"layer": int(best_dense["layer"]),
                                     **{m: float(best_dense[m]) for m in ("auroc", "paired_acc")
                                        if best_dense.get(m)}},
        "tie_break": "sae paired_acc -> sae auroc -> dense paired_acc -> dense auroc",
        "sae": ({"width": "16k", "l0": "medium", "k": k, "features": sae_feats,
                 "dev": {m: float(best_sae[m]) for m in ("auroc", "paired_acc") if best_sae.get(m)},
                 "feature_notes": [{k2: f[k2] for k2 in ("feature", "train_smd", "train_auroc", "train_act_rate")}
                                   for f in layer_feats],
                 "freeform_amplify": 0.0} if best_sae else None),
        "freeform_alpha": max(float(x) for x in args.alphas.split(",")),
        "evidence_hash": hashlib.sha256(
            Path(args.sweep).read_bytes()
            + (Path(args.sae_features).read_bytes() if Path(args.sae_features).exists() else b"")
        ).hexdigest(),
    }

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already frozen. Use --force only with a changelog entry explaining why.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in cfg.items() if k != "sae"}, indent=2))
    if cfg["sae"]:
        print("sae:", json.dumps({k: v for k, v in cfg["sae"].items() if k != "feature_notes"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
