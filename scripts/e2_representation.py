#!/usr/bin/env python3
"""E2 - where (if anywhere) is the interaction state readable from the residual stream?

Everything is fitted on TRAIN and scored on DEV. Test is never touched here.
Three representation families, as specified in the plan:
  dense mean-difference direction   d = normalize(mean(h_unresolved) - mean(h_resolved))
  L2 logistic probe
  SAE sparse probe (top-k features ranked on TRAIN only)

Plus the competing explanations and nulls that make a positive result interpretable:
  - a surface REPETITION direction fitted the same way (rep vs norep, ignoring repair state)
  - random-direction null, per layer
  - within-family label-permutation null
  - the within-family paired contrast, which is immune to any constant offset
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import stats as ST

from sklearn.linear_model import LogisticRegression


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def paired_rows(ids, fams, conds, labels, scores):
    return [{"item_id": i, "family_id": f, "repetition": "rep" if c.startswith("rep") else "norep",
             "label": l, "score": float(s)}
            for i, f, c, l, s in zip(ids, fams, conds, labels, scores)]


def eval_scores(y, s, fams, rows, boot, seed):
    out = {}
    b = ST.clustered_bootstrap(ST.auroc, fams, y, s, n=boot, seed=seed)
    out["auroc"], out["auroc_lo"], out["auroc_hi"] = b["point"], b["lo"], b["hi"]
    pairs = ST.make_pairs(rows)
    pb = ST.paired_bootstrap(pairs, ST.paired_accuracy_from, n=boot, seed=seed)
    out["paired_acc"], out["paired_acc_lo"], out["paired_acc_hi"] = pb["point"], pb["lo"], pb["hi"]
    out["n_pairs"] = pb["n_pairs"]
    for lvl in ("rep", "norep"):
        sub = [p for p in pairs if p["level"] == lvl]
        out[f"paired_acc_{lvl}"] = ST.paired_accuracy_from(sub)
    xp = ST.cross_pairs(rows)
    out["within_family_auroc"] = ST.paired_accuracy_from(xp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--sae-layers", default="9,17,22,29")
    ap.add_argument("--sae-topk", default="1,4,16")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--null-dirs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--outdir", default="runs/bridge2026/stage2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    A = z["acts"]                       # [n_layers, n_items, d]
    split, label = z["split"], z["label"]
    fam, cond, ids, rep = z["family_id"], z["condition"], z["item_id"], z["repetition"]
    n_layers, n_items, d = A.shape
    tr = np.flatnonzero((split == args.fit_split) & (label != "indeterminate"))
    ev = np.flatnonzero((split == args.eval_split) & (label != "indeterminate"))
    y = (label == "unresolved").astype(int)
    print(f"layers={n_layers} d={d} | fit n={len(tr)} ({len(set(fam[tr]))} fam) | "
          f"eval n={len(ev)} ({len(set(fam[ev]))} fam)")

    rng = np.random.default_rng(args.seed)
    rows, dirs = [], {}
    sae_layers = [int(x) for x in args.sae_layers.split(",") if x]
    topks = [int(x) for x in args.sae_topk.split(",") if x]

    for L in range(n_layers):
        H = A[L]
        # ---- dense mean-difference direction (TRAIN only) ----
        d_un = unit(H[tr][y[tr] == 1].mean(0) - H[tr][y[tr] == 0].mean(0))
        # ---- competing surface-repetition direction (TRAIN only) ----
        is_rep = (rep == "rep")
        d_rep = unit(H[tr][is_rep[tr]].mean(0) - H[tr][~is_rep[tr]].mean(0))
        dirs[f"L{L}_unresolved"] = d_un
        dirs[f"L{L}_repetition"] = d_rep

        s_ev = H[ev] @ d_un
        r = {"layer": L, "method": "mean_diff", "n_eval": len(ev),
             "cos_unres_rep": float(d_un @ d_rep),
             "sd_train_proj": float((H[tr] @ d_un).std()),
             "resid_norm_train": float(np.linalg.norm(H[tr], axis=1).mean())}
        r.update(eval_scores(y[ev], s_ev, fam[ev],
                             paired_rows(ids[ev], fam[ev], cond[ev], label[ev], s_ev),
                             args.boot, args.seed))
        # random-direction null at this layer
        null = []
        for _ in range(args.null_dirs):
            rd = unit(rng.standard_normal(d))
            null.append(ST.auroc(y[ev], H[ev] @ rd))
        null = np.array(null)
        r["null_auroc_mean"] = float(np.abs(null - .5).mean() + .5)
        r["null_auroc_p95"] = float(np.percentile(np.abs(null - .5), 95) + .5)
        r["beats_random_null"] = bool(abs(r["auroc"] - .5) > np.percentile(np.abs(null - .5), 95))
        # within-family label-permutation null
        pm = ST.permutation_within_family(lambda yy, ss: abs(ST.auroc(yy, ss) - .5),
                                          fam[ev], y[ev], s_ev, n=2000, seed=args.seed)
        r["perm_p"] = pm["p_value"]
        rows.append(r)

        # ---- repetition-direction readout of the REPETITION factor (sanity: is rep readable?) ----
        s_rep = H[ev] @ d_rep
        rows.append({"layer": L, "method": "mean_diff_repetition_axis", "n_eval": len(ev),
                     "auroc": ST.auroc(is_rep[ev].astype(int), s_rep),
                     "cos_unres_rep": float(d_un @ d_rep)})

        # ---- L2 logistic probe ----
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-8
        Xtr, Xev = (H[tr] - mu) / sd, (H[ev] - mu) / sd
        clf = LogisticRegression(max_iter=3000, C=0.05)
        clf.fit(Xtr, y[tr])
        s_lr = clf.decision_function(Xev)
        r2 = {"layer": L, "method": "logistic_l2", "n_eval": len(ev)}
        r2.update(eval_scores(y[ev], s_lr, fam[ev],
                              paired_rows(ids[ev], fam[ev], cond[ev], label[ev], s_lr),
                              args.boot, args.seed))
        r2["train_auroc"] = ST.auroc(y[tr], clf.decision_function(Xtr))
        rows.append(r2)
        print(f"  L{L:2d} meandiff AUROC={r['auroc']:.3f} paired={r['paired_acc']:.3f} "
              f"| logreg AUROC={r2['auroc']:.3f} | cos(unres,rep)={r['cos_unres_rep']:+.3f}", flush=True)

    # ---------------- SAE sparse probes ----------------
    import torch
    from bridge2026 import sae as S
    sae_rows, feat_rows = [], []
    for L in sae_layers:
        if L >= n_layers:
            continue
        sae = S.load_sae(args.model, L, device=args.device)
        H = torch.from_numpy(A[L]).to(args.device).float()
        Z = sae.encode(H).cpu().numpy()
        Ztr, Zev = Z[tr], Z[ev]
        act_rate = (Z > 0).mean(0)
        # rank on TRAIN only: standardised mean difference, restricted to features that actually fire
        m1, m0 = Ztr[y[tr] == 1].mean(0), Ztr[y[tr] == 0].mean(0)
        pooled = Ztr.std(0) + 1e-6
        score = np.abs(m1 - m0) / pooled
        alive = (Ztr > 0).mean(0) >= 0.05
        score = np.where(alive, score, -np.inf)
        order = np.argsort(-score)
        for rank, fi in enumerate(order[:30]):
            feat_rows.append({
                "layer": L, "rank": rank, "feature": int(fi),
                "train_smd": float(score[fi]), "train_mean_unres": float(m1[fi]),
                "train_mean_res": float(m0[fi]), "train_act_rate": float((Ztr[:, fi] > 0).mean()),
                "all_act_rate": float(act_rate[fi]),
                "train_auroc": float(ST.auroc(y[tr], Ztr[:, fi])),
                "sae_id": sae.sae_id,
            })
        for k in topks:
            cols = order[:k]
            clf = LogisticRegression(max_iter=3000, C=1.0)
            sc = Ztr[:, cols].std(0) + 1e-8
            clf.fit(Ztr[:, cols] / sc, y[tr])
            s_ev = clf.decision_function(Zev[:, cols] / sc)
            rr = {"layer": L, "method": f"sae_topk{k}", "n_eval": len(ev),
                  "features": json.dumps([int(c) for c in cols]),
                  "train_auroc": ST.auroc(y[tr], clf.decision_function(Ztr[:, cols] / sc))}
            rr.update(eval_scores(y[ev], s_ev, fam[ev],
                                  paired_rows(ids[ev], fam[ev], cond[ev], label[ev], s_ev),
                                  args.boot, args.seed))
            sae_rows.append(rr)
            print(f"  SAE L{L} k={k:2d} dev AUROC={rr['auroc']:.3f} paired={rr['paired_acc']:.3f}")
        del sae, H
        torch.cuda.empty_cache()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    allrows = rows + sae_rows
    keys = sorted({k for r in allrows for k in r})
    with open(out / f"e2_layer_sweep_{args.eval_split}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    with open(out / f"e2_sae_features_{args.eval_split}.csv", "w", newline="") as fh:
        if feat_rows:
            w = csv.DictWriter(fh, fieldnames=list(feat_rows[0].keys())); w.writeheader(); w.writerows(feat_rows)
    np.savez_compressed(out / "e2_directions.npz", **dirs)

    best = max((r for r in rows if r["method"] == "mean_diff"), key=lambda r: r["auroc"])
    bestlr = max((r for r in rows if r["method"] == "logistic_l2"), key=lambda r: r["auroc"])
    print(f"\nbest mean_diff  : L{best['layer']} AUROC={best['auroc']:.3f}"
          f"[{best['auroc_lo']:.3f},{best['auroc_hi']:.3f}] paired={best['paired_acc']:.3f} "
          f"perm_p={best['perm_p']:.4f}")
    print(f"best logistic_l2: L{bestlr['layer']} AUROC={bestlr['auroc']:.3f}"
          f"[{bestlr['auroc_lo']:.3f},{bestlr['auroc_hi']:.3f}] paired={bestlr['paired_acc']:.3f}")
    print(f"-> {out}/e2_layer_sweep_{args.eval_split}.csv")


if __name__ == "__main__":
    main()
