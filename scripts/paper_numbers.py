#!/usr/bin/env python3
"""Emit every number the paper cites, straight from the result files.

The paper text must never contain a number that is not produced here, so a later re-run
immediately shows any drift.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

def rd(p):
    p = Path(p); return list(csv.DictReader(open(p))) if p.exists() else []
def jd(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def pick(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()): return r
    return None
def f3(r, k):
    try: return round(float(r[k]), 3)
    except Exception: return None
def ci(r, k):
    return [f3(r, k), f3(r, f"{k}_lo"), f3(r, f"{k}_hi")]

S2, S3, S4, S5 = ("runs/bridge2026/stage2", "runs/bridge2026/stage3",
                  "runs/bridge2026/stage4", "runs/bridge2026/stage5_generality")
out = {}

sp = jd("data/bridge2026/splits.json")
out["dataset"] = {"families": sp["n_families"], "main_items": sp["n_items_main"],
                  "ambiguity_items": sp["n_items_ambiguity"], "splits": sp["item_counts"],
                  "content_hash": sp["content_hash"][:16], "per_domain": sp["per_domain"]}

rows = rd(f"{S4}/item_rows.csv") or rd(f"{S2}/item_rows.csv")
if rows:
    import statistics as st
    bal = {}
    for lab in ("unresolved", "resolved"):
        s = [r for r in rows if r["label"] == lab]
        bal[lab] = {"n": len(s),
                    "chars": round(st.mean(int(r["n_chars"]) for r in s), 1),
                    "words": round(st.mean(int(r["n_words"]) for r in s), 1),
                    "qmarks": round(st.mean(int(r["n_qmarks"]) for r in s), 2),
                    "turns": round(st.mean(int(r["n_turns"]) for r in s), 2)}
    out["balance"] = bal

ag = jd(f"{S2.replace('stage2','stage1')}/blind_agreement.json") or jd("runs/bridge2026/stage1/blind_agreement.json")
if ag:
    out["blind_annotation"] = {"n": ag["n_scored"], "kappa": round(ag["cohen_kappa_all"], 3),
                               "kappa_decidable": round(ag["cohen_kappa_decidable"], 3),
                               "raw_agreement": round(ag["raw_agreement_all"], 3),
                               "n_disputes": len(ag["disputes"])}

e0 = jd("runs/bridge2026/stage1/e0_smoke.json")
if e0:
    out["e0_4b"] = {"all_pass": e0["all_pass"],
                    "final_layer_norm_ratio": round(e0["checks"]["E0.2_final_layer_diverges"]["norm_ratio_hs_over_hook"], 4)}
out["e0_12b_norm_ratio"] = round(jd(f"{S3}/e0_acceptance_12b.json")["checks"]["E0.2_final_layer_diverges"]["norm_ratio_hs_over_hook"], 5)

reach = rd(f"{S2}/e0_reachability.csv")
out["reachability_4b"] = {f"L{r['layer']}_{r['mode']}": round(float(r["mean_abs_d_AB_margin"]), 4) for r in reach}

base = rd(f"{S4}/e1_baselines_test.csv")
out["baselines_test"] = {r["method"]: {"paired": ci(r, "paired_acc"), "p": f3(r, "paired_perm_p"),
                                       "auroc": ci(r, "auroc"), "fa": f3(r, "false_alarm_on_resolved")}
                         for r in base}

for tag, p in (("4b", f"{S4}/e1_model_readout_test.csv"), ("12b", f"{S4}/e1_model_readout_test_12b.csv"),
               ("qwen", f"{S5}/e1_model_readout_test_qwen.csv")):
    mr = rd(p)
    if mr:
        out[f"readout_test_{tag}"] = {r["arm"]: {"paired": ci(r, "paired_acc"), "p": f3(r, "paired_perm_p"),
                                                 "auroc": ci(r, "auroc"), "fa": f3(r, "false_alarm")} for r in mr}

for tag, p, cfgp in (("4b", f"{S4}/e2_incremental_test.csv", "runs/bridge2026/frozen_config_4b.json"),
                     ("12b", f"{S4}/e2_incremental_test_12b.csv", "runs/bridge2026/frozen_config_12b.json")):
    inc, cfg = rd(p), jd(cfgp)
    if inc and cfg:
        fr = pick(inc, layer=str(cfg["layer"]))
        best = max(inc, key=lambda r: float(r["increment"]))
        out[f"incremental_test_{tag}"] = {
            "frozen_layer": cfg["layer"],
            "frozen": {"probe": f3(fr, "probe_paired"), "surface": f3(fr, "surface_paired"),
                       "increment": ci(fr, "increment"), "p": f3(fr, "increment_p")},
            "best_on_test": {"layer": int(best["layer"]), "probe": f3(best, "probe_paired"),
                             "increment": ci(best, "increment"), "p": f3(best, "increment_p")}}

for tag, p in (("4b", f"{S4}/e3_causal_test.csv"), ("12b", f"{S4}/e3_causal_test_12b.csv")):
    ca = rd(p)
    if ca:
        gaps = {r["arm"]: f3(r, "selectivity_gap") for r in ca if r.get("selectivity_gap")}
        out[f"causal_test_{tag}"] = {
            "n_arms": len(ca), "max_abs_gap": round(max(abs(v) for v in gaps.values() if v is not None), 3),
            "arms": {r["arm"]: {"d_target": ci(r, "delta_target"), "d_nontarget": f3(r, "delta_nontarget"),
                                "gap": f3(r, "selectivity_gap"), "paired": f3(r, "paired_acc"),
                                "fa": f3(r, "false_alarm")} for r in ca}}

for tag, p in (("4b", f"{S4}/e4_external_4b.csv"), ("12b", f"{S4}/e4_external_12b.csv"),
               ("qwen", f"{S5}/e4_external_qwen.csv")):
    ex = rd(p)
    if ex:
        out[f"external_{tag}"] = {r["arm"]: {"fa": ci(r, "false_alarm"), "fa_rep": f3(r, "false_alarm_rep"),
                                             "fa_norep": f3(r, "false_alarm_norep"),
                                             "n_resolved": r.get("n_resolved")} for r in ex}

for tag, p in (("4b", f"{S4}/context_contrast_test_4b.csv"), ("12b", f"{S4}/context_contrast_test_12b.csv"),
               ("qwen", f"{S5}/context_contrast_test_qwen.csv")):
    cc = rd(p)
    if cc:
        out[f"context_contrast_test_{tag}"] = {r["contrast"]: {"full": f3(r, "paired_a"), "ablated": f3(r, "paired_b"),
                                                               "diff": f3(r, "diff"), "lo": f3(r, "lo"),
                                                               "hi": f3(r, "hi"), "p": f3(r, "p_two_sided")} for r in cc}

rb = rd(f"{S4}/rubric_summary.csv")
if rb:
    out["rubric"] = {r["arm"]: {"addresses_open": f3(r, "addresses_open_issue_on_unresolved"),
                                "false_trigger": ci(r, "false_trigger_on_resolved"),
                                "facts": f3(r, "maintains_task_facts"),
                                "cog_inf": f3(r, "cognitive_inference"),
                                "appropriate": f3(r, "overall_appropriate")} for r in rb}

ff = rd(f"{S4}/e4_freeform_test.csv")
if ff:
    from collections import defaultdict
    by = defaultdict(dict)
    for r in ff: by[r["item_id"]][r["arm"]] = r["response"]
    out["freeform_identity"] = {a: f"{sum(1 for d in by.values() if d.get(a)==d.get('original'))}/{len(by)}"
                                for a in sorted({r['arm'] for r in ff}) if a != "original"}

amb = rd(f"{S4}/ambiguity_4b.csv")
if amb:
    out["ambiguity"] = {r["set"]: {"n": r["n"], "mean_abs": f3(r, "mean_abs_score"),
                                   "near_indiff": f3(r, "frac_abs_score_lt_1")} for r in amb}

ea = Path(f"{S4}/ERROR_ANALYSIS_4b.md")
if ea.exists():
    sub, inblk = {}, False
    for line in ea.read_text().splitlines():
        if line.startswith("## By evidence subtype"): inblk = True; continue
        if inblk and line.startswith("## "): break
        if inblk and line.startswith("| `"):
            c = [x.strip().strip("`") for x in line.strip("|").split("|")]
            sub[c[0]] = {"n": c[1], "acc": c[2], "mean_score": c[5]}
    out["error_by_subtype_4b"] = sub

Path("paper/NUMBERS.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)[:2600])
print(f"\n-> paper/NUMBERS.json  ({len(json.dumps(out))} bytes)")
