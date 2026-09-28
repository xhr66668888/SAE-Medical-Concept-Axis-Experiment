#!/usr/bin/env python3
"""E1 non-model baselines + the shortcut audit that gates Stage 2.

Baselines (all read ONLY the dialogue text; none may see condition ids, subtypes or metadata):
  majority            - predict the majority train label
  rep_rule            - predict 'unresolved' iff a >=5-token echo is present  (the surface heuristic
                        the whole design is built to defeat; must sit at chance)
  length_rule         - predict 'unresolved' iff length above the train median
  surface_lr          - logistic regression on the 19 hand-built surface features
  tfidf_word          - word 1-2gram TF-IDF + L2 logistic regression
  tfidf_char          - char_wb 3-5gram TF-IDF + L2 logistic regression
  tfidf_word_char     - union of the two

This replaces the old `run_lexical_baseline.py`, which read the CCS code and re-applied the
label-construction rule -- an ontology oracle, not a text baseline.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge2026.features import FEATURE_NAMES, dialogue_text, feature_matrix
from bridge2026 import stats as ST

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler


def load_items(path, splits=("train", "dev", "test"), main_only=True):
    out = []
    for line in open(path):
        r = json.loads(line)
        if main_only and r["set"] != "main":
            continue
        if r["split"] in splits:
            out.append(r)
    return out


def y_of(items):
    return np.array([1 if it["label"] == "unresolved" else 0 for it in items])


def fam_of(items):
    return np.array([it["family_id"] for it in items])


def evaluate(name, y, pred, score, fams, n_boot, seed, extra=None, items=None):
    row = {"method": name, "n": int(len(y))}
    if items is not None and score is not None:
        rows_p = [{"family_id": it["family_id"], "repetition": it["repetition"],
                   "label": it["label"], "score": float(s)} for it, s in zip(items, score)]
        pairs = ST.make_pairs(rows_p)
        pb = ST.paired_bootstrap(pairs, ST.paired_accuracy_from, n=n_boot, seed=seed)
        sp = ST.paired_sign_permutation(pairs, ST.paired_accuracy_from, n=2000, seed=seed)
        row["paired_acc"] = pb["point"]; row["paired_acc_lo"] = pb["lo"]; row["paired_acc_hi"] = pb["hi"]
        row["paired_perm_p"] = sp["p_two_sided"]
        row["within_family_auroc"] = ST.paired_accuracy_from(ST.cross_pairs(rows_p))
    for stat, fn in (("acc", ST.accuracy), ("macro_f1", ST.macro_f1)):
        b = ST.clustered_bootstrap(fn, fams, y, pred, n=n_boot, seed=seed)
        row[stat] = b["point"]; row[f"{stat}_lo"] = b["lo"]; row[f"{stat}_hi"] = b["hi"]
    if score is not None and len(np.unique(y)) > 1:
        b = ST.clustered_bootstrap(ST.auroc, fams, y, score, n=n_boot, seed=seed)
        row["auroc"] = b["point"]; row["auroc_lo"] = b["lo"]; row["auroc_hi"] = b["hi"]
    row["hit_rate_unresolved"] = ST.hit_rate(y, pred)
    row["false_alarm_on_resolved"] = ST.false_alarm_rate(y, pred)
    if extra:
        row.update(extra)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bridge2026/items.jsonl")
    ap.add_argument("--eval-split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="runs/bridge2026/stage2/e1_baselines_dev.csv")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    tr = load_items(args.items, ("train",))
    ev = load_items(args.items, (args.eval_split,))
    if not tr or not ev:
        raise SystemExit("empty train or eval split")
    ytr, yev = y_of(tr), y_of(ev)
    ftr, fev = fam_of(tr), fam_of(ev)
    Ttr = [dialogue_text(i) for i in tr]
    Tev = [dialogue_text(i) for i in ev]
    rows = []

    # majority
    maj = int(np.bincount(ytr).argmax())
    rows.append(evaluate("majority", yev, np.full(len(yev), maj), None, fev, args.boot, args.seed, items=ev))

    # repetition rule: 'unresolved' iff a >=5-token echo exists
    Xtr, Xev = feature_matrix(tr), feature_matrix(ev)
    ri = FEATURE_NAMES.index("rep_any")
    pred = (Xev[:, ri] >= 5).astype(int)
    rows.append(evaluate("rep_rule(>=5-token echo)", yev, pred, Xev[:, ri], fev, args.boot, args.seed, items=ev))

    # length rule
    li = FEATURE_NAMES.index("n_words")
    thr = float(np.median(Xtr[:, li]))
    pred = (Xev[:, li] > thr).astype(int)
    rows.append(evaluate(f"length_rule(>{thr:.0f} words)", yev, pred, Xev[:, li], fev, args.boot, args.seed, items=ev))

    # surface-feature logistic regression  <-- the shortcut detector
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000, C=1.0))])
    pipe.fit(Xtr, ytr)
    sc = pipe.predict_proba(Xev)[:, 1]
    coefs = dict(sorted(zip(FEATURE_NAMES, pipe.named_steps["lr"].coef_[0].tolist()),
                        key=lambda kv: -abs(kv[1])))
    rows.append(evaluate("surface_lr(19 feats)", yev, (sc > .5).astype(int), sc, fev, args.boot, args.seed,
                         extra={"top_coefs": json.dumps({k: round(v, 3) for k, v in list(coefs.items())[:6]})}, items=ev))

    # TF-IDF variants
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    for nm, vec in (("tfidf_word(1-2)", word),
                    ("tfidf_char_wb(3-5)", char),
                    ("tfidf_word+char", FeatureUnion([
                        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))]))):
        p = Pipeline([("v", vec), ("lr", LogisticRegression(max_iter=5000, C=4.0))])
        p.fit(Ttr, ytr)
        s = p.predict_proba(Tev)[:, 1]
        rows.append(evaluate(nm, yev, (s > .5).astype(int), s, fev, args.boot, args.seed, items=ev))

    # --- shortcut audit: per-feature separation, within-family ---
    audit = []
    for j, nm in enumerate(FEATURE_NAMES):
        col = np.concatenate([Xtr[:, j], Xev[:, j]])
        yy = np.concatenate([ytr, yev])
        ff = np.concatenate([ftr, fev])
        if len(np.unique(col)) < 2:
            audit.append({"feature": nm, "auroc": 0.5, "p_perm": 1.0, "note": "constant"})
            continue
        a = ST.auroc(yy, col)
        pm = ST.permutation_within_family(lambda y_, c_: abs(ST.auroc(y_, c_) - .5), ff, yy, col,
                                          n=2000, seed=args.seed)
        audit.append({"feature": nm, "auroc": a, "abs_dev": abs(a - .5), "p_perm": pm["p_value"]})
    audit.sort(key=lambda r: -r.get("abs_dev", 0))
    flags = ST.benjamini_hochberg(np.array([r["p_perm"] for r in audit]), q=0.05)
    for r, f in zip(audit, flags):
        r["bh_flagged"] = bool(f)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        cols = ["method", "n", "paired_acc", "paired_acc_lo", "paired_acc_hi",
                "paired_perm_p", "within_family_auroc", "acc", "acc_lo", "acc_hi",
                "macro_f1", "macro_f1_lo", "macro_f1_hi", "auroc", "auroc_lo", "auroc_hi",
                "hit_rate_unresolved", "false_alarm_on_resolved", "top_coefs"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ap_out = Path(args.out).with_name(Path(args.out).stem + "_shortcut_audit.csv")
    with open(ap_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "auroc", "abs_dev", "p_perm", "bh_flagged", "note"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(audit)

    print(f"=== E1 text/surface baselines on {args.eval_split} (n={len(ev)} items, "
          f"{len(set(fev))} families) ===")
    for r in rows:
        au = f"{r['auroc']:.3f}[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]" if "auroc" in r else "   -   "
        pa = (f"{r['paired_acc']:.3f}[{r['paired_acc_lo']:.3f},{r['paired_acc_hi']:.3f}] p={r['paired_perm_p']:.3f}"
              if "paired_acc" in r else "        -        ")
        print(f"  {r['method']:28s} paired={pa} AUROC={au} acc={r['acc']:.3f} FA={r['false_alarm_on_resolved']:.3f}")
    print(f"\n=== shortcut audit (single-feature AUROC, within-family permutation) ===")
    for r in audit[:8]:
        print(f"  {r['feature']:22s} AUROC={r['auroc']:.3f}  p={r['p_perm']:.4f}  "
              f"{'<-- FLAGGED' if r['bh_flagged'] else ''}")
    print(f"\nflagged features: {sum(r['bh_flagged'] for r in audit)}/{len(audit)}  -> {ap_out}")


if __name__ == "__main__":
    main()
