#!/usr/bin/env python3
"""Does the residual stream carry information about the interaction state BEYOND surface statistics?

The paired-level shortcut audit showed that 9 of 19 hand-built surface features separate the labels
within families (worst: type-token ratio at 0.706 paired). Mean-matching the corpus hid this,
because the leak is a consistent WITHIN-PAIR direction rather than a difference of means. Any claim
that "the interaction state is linearly readable from the residual stream" is therefore only
meaningful as an INCREMENTAL claim over a surface model.

Three models, all fitted on TRAIN, all scored on DEV, all compared with the same family-clustered
paired bootstrap:
    surface   19 hand-built surface features
    probe     standardised residual activations at layer L
    joint     both, concatenated

Primary quantity: paired_acc(joint) - paired_acc(surface), with a paired CI. Plus a
surface-matched subset: the third of pairs the surface model separates least, where any remaining
probe advantage cannot be a surface effect.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026 import stats as ST
from bridge2026.features import FEATURE_NAMES, feature_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def prows(ids, fams, conds, labels, scores):
    return [{"item_id": i, "family_id": f, "repetition": "rep" if c.startswith("rep") else "norep",
             "label": l, "score": float(s)}
            for i, f, c, l, s in zip(ids, fams, conds, labels, scores)]


def fit_score(Xtr, ytr, Xev, C=0.05):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xev)), clf.decision_function(sc.transform(Xtr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = [r for r in (json.loads(l) for l in open(args.items)) if r["set"] == "main"]
    by_id = {r["item_id"]: r for r in items}
    z = np.load(args.acts, allow_pickle=True)
    A, ids = z["acts"], z["item_id"]
    split, label, fam, cond = z["split"], z["label"], z["family_id"], z["condition"]
    y = (label == "unresolved").astype(int)
    Xs = feature_matrix([by_id[i] for i in ids])
    tr = np.flatnonzero((split == "train") & (label != "indeterminate"))
    ev = np.flatnonzero((split == args.eval_split) & (label != "indeterminate"))
    print(f"train n={len(tr)} ({len(set(fam[tr]))} fam) | {args.eval_split} n={len(ev)} ({len(set(fam[ev]))} fam)")

    # ---- surface-only reference ----
    s_surf, s_surf_tr = fit_score(Xs[tr], y[tr], Xs[ev], C=1.0)
    rs = prows(ids[ev], fam[ev], cond[ev], label[ev], s_surf)
    pairs_surf = ST.make_pairs(rs)
    base = ST.paired_bootstrap(pairs_surf, ST.paired_accuracy_from, n=args.boot, seed=args.seed)
    print(f"surface-only: paired={base['point']:.3f} [{base['lo']:.3f},{base['hi']:.3f}]  "
          f"AUROC={ST.auroc(y[ev], s_surf):.3f}")

    # surface-matched subset: the third of pairs the surface model separates least
    margins = np.array([p["pos"]["score"] - p["neg"]["score"] for p in pairs_surf])
    thr = np.percentile(np.abs(margins), 33.3)
    keep = {(p["family_id"], p["level"]) for p, m in zip(pairs_surf, margins) if abs(m) <= thr}
    print(f"surface-matched subset: {len(keep)}/{len(pairs_surf)} pairs with |surface margin| <= {thr:.3f}")

    rows = []
    for L in range(A.shape[0]):
        H = A[L]
        s_probe, _ = fit_score(H[tr], y[tr], H[ev], C=0.05)
        Xj_tr = np.hstack([Xs[tr], H[tr]])
        Xj_ev = np.hstack([Xs[ev], H[ev]])
        s_joint, _ = fit_score(Xj_tr, y[tr], Xj_ev, C=0.05)

        r = {"layer": L}
        for nm, s in (("surface", s_surf), ("probe", s_probe), ("joint", s_joint)):
            pr = prows(ids[ev], fam[ev], cond[ev], label[ev], s)
            pp = ST.make_pairs(pr)
            b = ST.paired_bootstrap(pp, ST.paired_accuracy_from, n=args.boot, seed=args.seed)
            r[f"{nm}_paired"] = b["point"]; r[f"{nm}_lo"] = b["lo"]; r[f"{nm}_hi"] = b["hi"]
            r[f"{nm}_auroc"] = ST.auroc(y[ev], s)
            sub = [p for p in pp if (p["family_id"], p["level"]) in keep]
            r[f"{nm}_paired_matched"] = ST.paired_accuracy_from(sub)

        # paired CI on the INCREMENT, resampling the same families for both models
        pp_j = ST.make_pairs(prows(ids[ev], fam[ev], cond[ev], label[ev], s_joint))
        pp_s = pairs_surf
        fams_u = sorted({p["family_id"] for p in pp_s})
        byf_j = {f: [p for p in pp_j if p["family_id"] == f] for f in fams_u}
        byf_s = {f: [p for p in pp_s if p["family_id"] == f] for f in fams_u}
        rng = np.random.default_rng(args.seed)
        draws = np.empty(args.boot)
        for b_ in range(args.boot):
            pick = rng.integers(0, len(fams_u), size=len(fams_u))
            sj = [p for i in pick for p in byf_j[fams_u[i]]]
            ss = [p for i in pick for p in byf_s[fams_u[i]]]
            draws[b_] = ST.paired_accuracy_from(sj) - ST.paired_accuracy_from(ss)
        r["increment"] = r["joint_paired"] - r["surface_paired"]
        r["increment_lo"], r["increment_hi"] = np.percentile(draws, [2.5, 97.5])
        r["increment_p"] = float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))
        rows.append(r)
        print(f"  L{L:2d} surface={r['surface_paired']:.3f} probe={r['probe_paired']:.3f} "
              f"joint={r['joint_paired']:.3f} | increment={r['increment']:+.3f} "
              f"[{r['increment_lo']:+.3f},{r['increment_hi']:+.3f}] p={r['increment_p']:.3f} "
              f"| matched: surf={r['surface_paired_matched']:.3f} probe={r['probe_paired_matched']:.3f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["increment"])
    bp = max(rows, key=lambda r: r["probe_paired"])
    print(f"\nbest increment over surface: L{best['layer']} {best['increment']:+.3f} "
          f"[{best['increment_lo']:+.3f},{best['increment_hi']:+.3f}] p={best['increment_p']:.4f}")
    print(f"best probe-alone: L{bp['layer']} paired={bp['probe_paired']:.3f} "
          f"[{bp['probe_lo']:.3f},{bp['probe_hi']:.3f}]  (surface reference {base['point']:.3f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
