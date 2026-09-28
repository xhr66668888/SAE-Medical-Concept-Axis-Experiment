#!/usr/bin/env python3
"""Paper figures. Every value is read from paper/NUMBERS.json, never typed in.

Palette: slots 1-3 of the reference categorical theme (#2a78d6 / #eb6834 / #1baf7a), which that
palette documents as passing the all-pairs CVD and normal-vision floors in both modes. Marker shape
and hatch carry identity as well as colour, so the figures survive greyscale print.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = json.loads(Path("paper/NUMBERS.json").read_text())
C = {"4b": "#2a78d6", "12b": "#eb6834", "qwen": "#1baf7a"}
MK = {"4b": "o", "12b": "s", "qwen": "^"}
HA = {"4b": "", "12b": "///", "qwen": "..."}
LB = {"4b": "Gemma-3-4B", "12b": "Gemma-3-12B", "qwen": "Qwen2.5-7B"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans",
                     "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# ---------------------------------------------------------------- Figure 1: over-attribution
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

ax = axes[0]
models = ["4b", "12b", "qwen"]
vals = [N[f"external_{m}"]["original"]["fa"] for m in models]
pt = [v[0] for v in vals]
err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
x = np.arange(len(models))
for i, m in enumerate(models):
    ax.bar(x[i], pt[i], width=0.62, color=C[m], hatch=HA[m], edgecolor="white", linewidth=1.2, zorder=2)
ax.errorbar(x, pt, yerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)
base = 3 / 100
ax.axhline(base, color=INK, ls=(0, (4, 2)), lw=0.9, zorder=4)
ax.text(-0.42, base + 0.03, "true rate of open problems (3%)", ha="left", va="bottom",
        fontsize=6.4, color=INK)
for i, v in enumerate(pt):
    ax.text(x[i], v + 0.055, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4, color=INK, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([LB[m] for m in models], fontsize=6.8)
ax.set_ylim(0, 1.22); ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_ylabel("fraction judged “unresolved”", fontsize=7.2)
ax.set_title("(a)  97 ordinary human conversations (CCPE-M)", fontsize=7.4, loc="left", pad=6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0); ax.set_axisbelow(True)

ax = axes[1]
w = 0.36
for i, m in enumerate(models):
    r = N[f"external_{m}"]["original"]
    ax.bar(i - w / 2, r["fa_rep"], width=w, color=C[m], hatch=HA[m],
           edgecolor="white", linewidth=1.0, zorder=2)
    ax.bar(i + w / 2, r["fa_norep"], width=w, color=C[m], alpha=0.42,
           edgecolor="white", linewidth=1.0, zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels([LB[m] for m in models], fontsize=6.8)
ax.set_ylim(0, 1.22); ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_ylabel("false-alarm rate", fontsize=7.2)
ax.set_title("(b)  split by whether the segment actually repeats", fontsize=7.4, loc="left", pad=6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#7a7a78", label="contains a ≥5-token echo"),
                   Patch(facecolor="#7a7a78", alpha=0.42, label="no echo")],
          fontsize=6.3, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
          handlelength=1.1, handleheight=0.9, columnspacing=1.2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0); ax.set_axisbelow(True)

fig.tight_layout(pad=0.6)
fig.savefig("paper/fig1_overattribution.pdf", bbox_inches="tight")
fig.savefig("paper/fig1_overattribution.png", dpi=220, bbox_inches="tight")
print("fig1 ->", [f"{LB[m]}: {N[f'external_{m}']['original']['fa'][0]:.3f}" for m in models])

# ---------------------------------------------------------------- Figure 2: no selectivity
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
ax = axes[0]
allv = []
for m, path in (("4b", "runs/bridge2026/stage4/e3_causal_test.csv"),
                ("12b", "runs/bridge2026/stage4/e3_causal_test_12b.csv")):
    rows = [r for r in csv.DictReader(open(path)) if r["arm"] != "clean" and r.get("delta_target")]
    xt = [float(r["delta_target"]) for r in rows]
    yt = [float(r["delta_nontarget"]) for r in rows]
    allv += xt + yt
    ax.scatter(xt, yt, s=26, c=C[m], marker=MK[m], edgecolors="white", linewidths=0.7,
               label=f"{LB[m]} ({len(rows)} arms)", zorder=3, alpha=0.92)
lim = max(abs(min(allv)), abs(max(allv))) * 1.12
ax.plot([-lim, lim], [-lim, lim], color=INK, ls=(0, (4, 2)), lw=0.9, zorder=2)
ax.annotate("y = x   (no selectivity)", xy=(lim * 0.52, lim * 0.52), rotation=45,
            rotation_mode="anchor", ha="center", va="bottom", fontsize=6.2, color=INK)
ax.axhline(0, color="#d6d5d0", lw=0.6, zorder=1); ax.axvline(0, color="#d6d5d0", lw=0.6, zorder=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("Δ on items that SHOULD move (unresolved)", fontsize=7.0)
ax.set_ylabel("Δ on items that should NOT (resolved)", fontsize=7.0)
ax.set_title("(a)  every intervention arm sits on the diagonal", fontsize=7.4, loc="left", pad=6)
ax.legend(fontsize=6.3, frameon=False, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

axi = ax.inset_axes([0.63, 0.07, 0.33, 0.33])
r4 = [r for r in csv.DictReader(open("runs/bridge2026/stage4/e3_causal_test.csv"))
      if r["arm"] != "clean" and r.get("delta_target")]
x4 = [float(r["delta_target"]) for r in r4]; y4 = [float(r["delta_nontarget"]) for r in r4]
l4 = max(max(map(abs, x4)), max(map(abs, y4))) * 1.2
axi.plot([-l4, l4], [-l4, l4], color=INK, ls=(0, (3, 2)), lw=0.7, zorder=2)
axi.scatter(x4, y4, s=11, c=C["4b"], marker="o", edgecolors="white", linewidths=0.5, zorder=3)
axi.set_xlim(-l4, l4); axi.set_ylim(-l4, l4); axi.set_aspect("equal")
axi.set_title(f"Gemma-3-4B, zoomed ±{l4:.2f}", fontsize=5.8, color=MUTED, pad=2)
axi.set_xticks([]); axi.set_yticks([])
for sp in axi.spines.values(): sp.set_color("#c9c8c3"); sp.set_linewidth(0.5)

ax = axes[1]
rb = list(csv.DictReader(open("runs/bridge2026/stage4/rubric_summary.csv")))
order = ["dense_axis", "sae_ablate", "prompt_only"]   # 'original' is the reference: zero by definition
nice = {"dense_axis": "dense axis", "sae_ablate": "SAE ablate", "prompt_only": "prompt only"}
o = {r["arm"]: r for r in rb}
gain = [float(o[a]["addresses_open_issue_on_unresolved"]) - float(o["original"]["addresses_open_issue_on_unresolved"]) for a in order]
cost = [float(o[a]["false_trigger_on_resolved"]) - float(o["original"]["false_trigger_on_resolved"]) for a in order]
y = np.arange(len(order)); h = 0.34
ax.barh(y + h / 2, gain, height=h, color="#1baf7a", edgecolor="white", linewidth=1.0, zorder=2)
ax.barh(y - h / 2, cost, height=h, color="#e34948", edgecolor="white", linewidth=1.0, zorder=2)
for i, (g_, c_) in enumerate(zip(gain, cost)):
    ax.text(max(g_, 0) + 0.006, i + h / 2, f"{g_:+.3f}", va="center", fontsize=6.4,
            color=INK if abs(g_) > 1e-9 else MUTED)
    ax.text(max(c_, 0) + 0.006, i - h / 2, f"{c_:+.3f}", va="center", fontsize=6.4,
            color=INK if abs(c_) > 1e-9 else MUTED)
ax.axvline(0, color=INK, lw=0.8, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([nice[a] for a in order], fontsize=6.8)
ax.set_xlabel("change vs the unmodified model (blind rubric)", fontsize=7.0)
ax.set_title("(b)  what an intervention buys, and what it costs", fontsize=7.4, loc="left", pad=6)
ax.legend(handles=[Patch(facecolor="#1baf7a", label="takes up a real open issue  (want ↑)"),
                   Patch(facecolor="#e34948", label="false trigger when nothing is open  (want ↓)")],
          fontsize=6.3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02),
          handlelength=1.1, handleheight=0.9)
ax.set_xlim(-0.055, 0.245); ax.set_ylim(-0.75, 2.75)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e6e5e0", lw=0.6, zorder=0); ax.set_axisbelow(True)

fig.tight_layout(pad=0.6)
fig.savefig("paper/fig2_selectivity.pdf", bbox_inches="tight")
fig.savefig("paper/fig2_selectivity.png", dpi=220, bbox_inches="tight")
print("fig2 -> gain", [round(g,3) for g in gain], "cost", [round(c,3) for c in cost])
