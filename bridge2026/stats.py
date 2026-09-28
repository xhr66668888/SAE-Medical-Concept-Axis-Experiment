"""Family-clustered inference.

Every dialogue in a family shares entities, task goal and tail anchor, so items are NOT independent.
All intervals resample FAMILIES, and all permutation tests permute within / across families as the
cluster unit (plan sec.6).
"""
from __future__ import annotations

import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def group_index(families: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq = np.unique(families)
    idx = [np.flatnonzero(families == f) for f in uniq]
    return uniq, idx


def clustered_bootstrap(stat_fn, families: np.ndarray, *arrays, n: int = 5000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Resample families with replacement; recompute `stat_fn(*arrays_resampled)`."""
    uniq, idx = group_index(np.asarray(families))
    rng = _rng(seed)
    point = float(stat_fn(*arrays))
    draws = np.empty(n, dtype=float)
    k = len(uniq)
    for b in range(n):
        pick = rng.integers(0, k, size=k)
        sel = np.concatenate([idx[p] for p in pick])
        try:
            draws[b] = stat_fn(*[np.asarray(a)[sel] for a in arrays])
        except Exception:  # degenerate resample (e.g. one class only)
            draws[b] = np.nan
    ok = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.size else (np.nan, np.nan))
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": int(ok.size),
            "se": float(ok.std(ddof=1)) if ok.size > 1 else np.nan, "n_families": int(k)}


def permutation_within_family(stat_fn, families: np.ndarray, labels: np.ndarray, *arrays,
                              n: int = 5000, seed: int = 0) -> dict:
    """Null: labels are exchangeable WITHIN a family. This is the right null for the 2x2 design,
    because it holds entities, task and tail anchor fixed and only scrambles the repair state."""
    fams = np.asarray(families)
    y = np.asarray(labels)
    uniq, idx = group_index(fams)
    rng = _rng(seed)
    point = float(stat_fn(y, *arrays))
    draws = np.empty(n, dtype=float)
    for b in range(n):
        yp = y.copy()
        for ix in idx:
            yp[ix] = rng.permutation(y[ix])
        try:
            draws[b] = stat_fn(yp, *arrays)
        except Exception:
            draws[b] = np.nan
    ok = draws[~np.isnan(draws)]
    p = float((np.sum(ok >= point) + 1) / (ok.size + 1)) if ok.size else np.nan
    return {"point": point, "null_mean": float(ok.mean()) if ok.size else np.nan,
            "null_sd": float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
            "p_value": p, "n_perm": int(ok.size)}


def paired_bootstrap_diff(stat_fn, families: np.ndarray, a_arrays: tuple, b_arrays: tuple,
                          n: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for stat(A) - stat(B) on the SAME families (paired: the same resample is used for both)."""
    uniq, idx = group_index(np.asarray(families))
    rng = _rng(seed)
    point = float(stat_fn(*a_arrays) - stat_fn(*b_arrays))
    draws = np.empty(n, dtype=float)
    k = len(uniq)
    for b in range(n):
        pick = rng.integers(0, k, size=k)
        sel = np.concatenate([idx[p] for p in pick])
        try:
            draws[b] = (stat_fn(*[np.asarray(x)[sel] for x in a_arrays])
                        - stat_fn(*[np.asarray(x)[sel] for x in b_arrays]))
        except Exception:
            draws[b] = np.nan
    ok = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.size else (np.nan, np.nan))
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))) if ok.size else np.nan,
            "n_boot": int(ok.size), "n_families": int(k)}


# ---------------- statistics ----------------

def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    fs = []
    for c in np.unique(np.concatenate([y, pred])):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. y in {0,1}."""
    y, s = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pos, neg = y == 1, y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def false_alarm_rate(y: np.ndarray, pred: np.ndarray, positive=1) -> float:
    """Rate of predicting 'unresolved' on genuinely resolved items (the normal-repetition
    false-trigger rate that the plan makes a headline endpoint)."""
    y, pred = np.asarray(y), np.asarray(pred)
    neg = y != positive
    return float(np.mean(pred[neg] == positive)) if neg.any() else np.nan


def hit_rate(y: np.ndarray, pred: np.ndarray, positive=1) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    pos = y == positive
    return float(np.mean(pred[pos] == positive)) if pos.any() else np.nan


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


# ---------------- paired statistics for the 2x2 family design ----------------

def make_pairs(rows, level_key="repetition"):
    """Within-family (unresolved, resolved) pairs at matched repetition level.

    These pairs share family, entities, task goal, tail anchor AND repetition level, so any constant
    readout bias (option position, label prior, wording prior) cancels exactly in the difference.
    """
    by = {}
    for r in rows:
        by.setdefault((r["family_id"], r.get(level_key, "na")), {})[r["label"]] = r
    pairs = []
    for (fam, lvl), d in sorted(by.items()):
        if "unresolved" in d and "resolved" in d:
            pairs.append({"family_id": fam, "level": lvl, "pos": d["unresolved"], "neg": d["resolved"]})
    return pairs


def paired_accuracy_from(pairs, key="score") -> float:
    """Ties count as 0.5, matching AUROC's tie convention. Without this a rule-based baseline that
    assigns both members of a pair the same score would score 0 rather than chance."""
    if not pairs:
        return float("nan")
    vals = []
    for p in pairs:
        a, b = p["pos"][key], p["neg"][key]
        vals.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return float(np.mean(vals))


def paired_margin_from(pairs, key="score") -> float:
    if not pairs:
        return float("nan")
    return float(np.mean([p["pos"][key] - p["neg"][key] for p in pairs]))


def cross_pairs(rows):
    """All within-family unresolved x resolved pairs (ignoring repetition level).
    The mean of [pos>neg] over these is the within-family AUROC."""
    by = {}
    for r in rows:
        by.setdefault(r["family_id"], {"unresolved": [], "resolved": []}).setdefault(r["label"], []).append(r)
    out = []
    for fam, d in sorted(by.items()):
        for p in d.get("unresolved", []):
            for n in d.get("resolved", []):
                out.append({"family_id": fam, "level": "cross", "pos": p, "neg": n})
    return out


def paired_bootstrap(pairs, stat_fn, n: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample FAMILIES; all pairs of a drawn family travel together."""
    fams = sorted({p["family_id"] for p in pairs})
    by_fam = {f: [p for p in pairs if p["family_id"] == f] for f in fams}
    rng = _rng(seed)
    point = float(stat_fn(pairs))
    draws = np.empty(n, dtype=float)
    k = len(fams)
    for b in range(n):
        pick = rng.integers(0, k, size=k)
        sel = [p for i in pick for p in by_fam[fams[i]]]
        draws[b] = stat_fn(sel)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_pairs": len(pairs),
            "n_families": k, "se": float(draws.std(ddof=1))}


def paired_sign_permutation(pairs, stat_fn, n: int = 5000, seed: int = 0, key="score") -> dict:
    """Null: within a pair the unresolved/resolved assignment is exchangeable. Flip signs per pair."""
    rng = _rng(seed)
    point = float(stat_fn(pairs))
    draws = np.empty(n, dtype=float)
    for b in range(n):
        flipped = []
        for p in pairs:
            if rng.random() < 0.5:
                flipped.append({**p, "pos": p["neg"], "neg": p["pos"]})
            else:
                flipped.append(p)
        draws[b] = stat_fn(flipped)
    p_two = float(2 * min((draws <= point).mean(), (draws >= point).mean()))
    return {"point": point, "null_mean": float(draws.mean()), "null_sd": float(draws.std(ddof=1)),
            "p_two_sided": min(1.0, p_two), "n_perm": n}
