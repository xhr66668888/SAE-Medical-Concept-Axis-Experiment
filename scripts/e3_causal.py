#!/usr/bin/env python3
"""E3 - causal selectivity.

Runs a fixed set of intervention arms through the same A/B readout and reports, for every arm:
  * the effect on conditions that SHOULD move (unresolved), and
  * the collateral effect on conditions that should NOT move (resolved) -- the false-trigger side.

The headline is the pair, never the average: an intervention that raises 'unresolved' everywhere is
a global response-style shift, not evidence that the model uses the representation for this
judgement.

Layer, feature set, dose and position must already be locked (frozen config) before this touches the
test split; `--split test` additionally requires --i-am-running-the-final-test.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import interventions as IV
from bridge2026 import readout as RO
from bridge2026 import runtime as R
from bridge2026 import sae as S
from bridge2026 import stats as ST
from bridge2026.prompts import read_position, render_pure
from bridge2026.schema import Turn


def load_items(path, split, sets=("main",)):
    return [r for r in (json.loads(l) for l in open(path))
            if r["set"] in sets and r["split"] == split and r["label"] != "indeterminate"]


def partner_map(items):
    """Opposite-label partner within the same family AND the same repetition level."""
    by = {}
    for it in items:
        by.setdefault((it["family_id"], it["repetition"]), {})[it["label"]] = it
    out = {}
    for (_f, _r), d in by.items():
        if "unresolved" in d and "resolved" in d:
            out[d["unresolved"]["item_id"]] = d["resolved"]
            out[d["resolved"]["item_id"]] = d["unresolved"]
    return out


def capture_at_read(rt, items, layer):
    texts = [render_pure(rt.tokenizer, [Turn(t["speaker"], t["text"]) for t in it["turns"]]) for it in items]
    pos = [read_position(rt.tokenizer, [Turn(t["speaker"], t["text"]) for t in it["turns"]]) for it in items]
    out = []
    for i in range(0, len(texts), 8):
        sl = slice(i, i + 8)
        ids, attn, ap = R.encode_batch(rt, texts[sl], pos[sl])
        st, _ = R.forward_capture(rt, ids, attn, [layer], ap)
        out.append(st[layer])
    return torch.cat(out, 0)


def summarize(arm, rows, scores, clean_by_item, boot, seed):
    agg = RO.aggregate(rows, scores)
    for a in agg:
        a["delta"] = a["score"] - clean_by_item.get(a["item_id"], np.nan)
    res = {"arm": arm, "n_items": len(agg)}
    for lab, tag in (("unresolved", "target"), ("resolved", "nontarget")):
        sub = [a for a in agg if a["label"] == lab]
        if not sub:
            continue
        fams = np.array([a["family_id"] for a in sub])
        dl = np.array([a["delta"] for a in sub])
        b = ST.clustered_bootstrap(lambda v: float(np.mean(v)), fams, dl, n=boot, seed=seed)
        res[f"delta_{tag}"] = b["point"]
        res[f"delta_{tag}_lo"], res[f"delta_{tag}_hi"] = b["lo"], b["hi"]
        res[f"pred_unres_rate_{tag}"] = float(np.mean([a["pred"] for a in sub]))
    pairs = ST.make_pairs(agg)
    pb = ST.paired_bootstrap(pairs, ST.paired_accuracy_from, n=boot, seed=seed)
    res["paired_acc"], res["paired_acc_lo"], res["paired_acc_hi"] = pb["point"], pb["lo"], pb["hi"]
    mb = ST.paired_bootstrap(pairs, ST.paired_margin_from, n=boot, seed=seed)
    res["paired_margin"], res["paired_margin_lo"], res["paired_margin_hi"] = mb["point"], mb["lo"], mb["hi"]
    y = np.array([a["y"] for a in agg]); sc = np.array([a["score"] for a in agg])
    fa = np.array([a["family_id"] for a in agg]); pr = np.array([a["pred"] for a in agg])
    ab = ST.clustered_bootstrap(ST.auroc, fa, y, sc, n=boot, seed=seed)
    res["auroc"], res["auroc_lo"], res["auroc_hi"] = ab["point"], ab["lo"], ab["hi"]
    res["hit_rate"] = ST.hit_rate(y, pr)
    res["false_alarm"] = ST.false_alarm_rate(y, pr)
    # selectivity: how much of the movement lands on the target side
    dt, dn = res.get("delta_target", np.nan), res.get("delta_nontarget", np.nan)
    res["selectivity_gap"] = dt - dn
    return res, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--config", required=True, help="frozen intervention config JSON")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--directions", default="runs/bridge2026/stage2/e2_directions.npz")
    ap.add_argument("--acts", default="", help="train acts npz, for donor codes / sd_train")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-am-running-the-final-test", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.i_am_running_the_final_test:
        raise SystemExit("test split is one-shot: pass --i-am-running-the-final-test deliberately")

    cfg = json.loads(Path(args.config).read_text())
    layer = int(cfg["layer"])
    items = load_items(args.items, args.split)
    print(f"split={args.split} items={len(items)} families={len({i['family_id'] for i in items})} "
          f"layer={layer} span={cfg.get('position_mode','read_to_end')}")

    rt = R.load_runtime(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    dirs = np.load(args.directions)
    d_un = dirs[f"L{layer}_unresolved"]
    d_rep = dirs[f"L{layer}_repetition"]
    sd_train = float(cfg["sd_train"])

    rows, texts, pos = RO.build_rows(rt.tokenizer, items, context="full")
    order = {r["item_id"]: k for k, r in enumerate(rows)}

    # donor material, aligned to ROW order
    pmap = partner_map(items)
    acts = capture_at_read(rt, items, layer)                     # [n_items, d]
    idx_of = {it["item_id"]: k for k, it in enumerate(items)}
    donor_rows = np.zeros((len(rows), acts.shape[1]), dtype=np.float32)
    donor_proj = np.zeros(len(rows), dtype=np.float32)
    has_donor = np.zeros(len(rows), dtype=bool)
    for k, r in enumerate(rows):
        p = pmap.get(r["item_id"])
        if p is None:
            donor_rows[k] = acts[idx_of[r["item_id"]]].numpy()
            donor_proj[k] = float(acts[idx_of[r["item_id"]]].numpy() @ d_un)
            continue
        a = acts[idx_of[p["item_id"]]].numpy()
        donor_rows[k] = a
        donor_proj[k] = float(a @ d_un)
        has_donor[k] = True
    print(f"donor coverage: {has_donor.mean():.0%} of renderings")

    # clean pass
    sc_clean, _ = RO.score_rows(rt, rows, texts, pos, batch_size=args.batch_size)
    agg_clean = RO.aggregate(rows, sc_clean)
    clean_by_item = {a["item_id"]: a["score"] for a in agg_clean}
    results, per_item = [], []
    r0, a0 = summarize("clean", rows, sc_clean, clean_by_item, args.boot, args.seed)
    results.append(r0); per_item += [{**a, "arm": "clean"} for a in a0]
    print(f"  clean: paired_acc={r0['paired_acc']:.3f} AUROC={r0['auroc']:.3f} FA={r0['false_alarm']:.3f}")

    span = cfg.get("position_mode", "read_to_end")
    arms: list[IV.Intervention] = []
    for alpha in cfg.get("alphas", []):
        a = IV.dense_axis(layer, d_un, alpha, sd_train, span=span)
        a.name = f"dense_axis(a={alpha:+g})"; arms.append(a)
    # the inert single-token site, kept as a reported arm so the E0.11 finding is visible in-table
    if cfg.get("report_read_pos_arm", True):
        a = IV.dense_axis(layer, d_un, max(cfg["alphas"]), sd_train, span="read_pos")
        a.name = f"dense_axis(a={max(cfg['alphas']):+g}, read_pos only)"; arms.append(a)
    for alpha in cfg.get("control_alphas", []):
        a1 = IV.random_dir(layer, rt.d_model, alpha, sd_train, args.seed, span=span)
        a1.name = f"random_dir(a={alpha:+g})"; arms.append(a1)
        a2 = IV.other_axis("x", layer, d_rep, alpha, sd_train, span=span)
        a2.name = f"repetition_axis(a={alpha:+g})"; arms.append(a2)
        cl = Path(args.directions).with_name("competing_directions.npz")
        if cl.exists():
            dc = np.load(cl)[f"L{layer}_clarification"]
            a3 = IV.other_axis("x", layer, dc, alpha, sd_train, span=span)
            a3.name = f"clarification_axis(a={alpha:+g})"; arms.append(a3)
    if cfg.get("component_swap", True):
        a = IV.component_swap(layer, d_un, donor_proj, span=span)
        a.name = "component_swap(donor)"; arms.append(a)
    if cfg.get("full_patch", True):
        a = IV.full_patch(layer, donor_rows); a.name = "full_patch(donor)"; arms.append(a)

    sae_cfg = cfg.get("sae")
    sae = None
    if sae_cfg:
        sae = S.load_sae(args.model, layer, width=sae_cfg.get("width", "16k"),
                         l0=sae_cfg.get("l0", "medium"), device=rt.device)
        feats = [int(f) for f in sae_cfg["features"]]
        codes_all = sae.encode(acts.to(rt.device).float()).cpu().numpy()
        donor_codes = np.zeros((len(rows), len(feats)), dtype=np.float32)
        for k, r in enumerate(rows):
            p = pmap.get(r["item_id"])
            src = idx_of[p["item_id"]] if p else idx_of[r["item_id"]]
            donor_codes[k] = codes_all[src][feats]
        arms.append(IV.sae_reconstruct(layer, sae, span=span))
        arms.append(IV.sae_feature_edit("sae_ablate", layer, sae, feats, "ablate", span=span))
        arms.append(IV.sae_feature_edit("sae_donor", layer, sae, feats, "donor",
                                        donor_codes=donor_codes, span=span))
        rnd = IV.match_random_features(sae, codes_all, feats, args.seed)
        arms.append(IV.sae_feature_edit("sae_random_ablate", layer, sae, rnd, "ablate", span=span))
        drc = np.zeros((len(rows), len(rnd)), dtype=np.float32)
        for k, r in enumerate(rows):
            p = pmap.get(r["item_id"])
            src = idx_of[p["item_id"]] if p else idx_of[r["item_id"]]
            drc[k] = codes_all[src][rnd]
        arms.append(IV.sae_feature_edit("sae_random_donor", layer, sae, rnd, "donor",
                                        donor_codes=drc, span=span))
        print(f"  SAE arms on features {feats} (random-matched: {rnd})")

    for arm in arms:
        sc, _ = RO.score_rows(rt, rows, texts, pos, intervene=arm, batch_size=args.batch_size)
        r, a = summarize(arm.name, rows, sc, clean_by_item, args.boot, args.seed)
        r["layer"] = layer
        r["span"] = arm.span
        r.update({f"meta_{k}": (json.dumps(v) if isinstance(v, (list, dict)) else v)
                  for k, v in arm.meta.items() if k != "features"})
        results.append(r)
        per_item += [{**x, "arm": arm.name} for x in a]
        print(f"  {arm.name:28s} d_target={r.get('delta_target', float('nan')):+.3f} "
              f"d_nontarget={r.get('delta_nontarget', float('nan')):+.3f} "
              f"gap={r['selectivity_gap']:+.3f} paired={r['paired_acc']:.3f} FA={r['false_alarm']:.3f}",
              flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(results)
    pi = out.with_name(out.stem + "_per_item.csv")
    pk = sorted({k for r in per_item for k in r})
    with open(pi, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pk, extrasaction="ignore"); w.writeheader(); w.writerows(per_item)
    print(f"\n-> {out}\n-> {pi}")


if __name__ == "__main__":
    main()
