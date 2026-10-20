"""Figure: settling test (frozen primary + post hoc floor-matched). Claude, 2026-10-20."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
r = json.load(open("score_result.json")); p = json.load(open("posthoc_floor_matched.json"))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
lv = [r["primary"]["lower"]["level_A"], r["primary"]["lower"]["level_B"], r["secondary_day"]["lower"]["level_B"]]
uv = [r["primary"]["upper"]["level_A"], r["primary"]["upper"]["level_B"], r["secondary_day"]["upper"]["level_B"]]
x = np.arange(3); lab = ["cold\n(just mounted,\n03-25)", "warm\n(after play-in,\n03-26)", "A1 main\n(03-27,\n0.10 m/s)"]
ax[0].plot(x, lv, "o-", color="#2563a0", ms=8, label="minimum force (lower)"); ax[0].plot(x, uv, "s--", color="#d38325", ms=7, label="maximum force (upper)")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel("boundary level (ln; lower vs A1+A2 law,\nupper vs strict cZv/β)")
ax[0].set_title("(a) one type-A T1 string over its first days", fontsize=10); ax[0].legend(fontsize=8, frameon=False)
rows = [("Δ lower warm−cold\n(frozen)", r["primary"]["lower"]), ("Δ lower warm−cold\n(floor-matched, post hoc)", p["warm_minus_cold"]["lower"]),
        ("Δ upper warm−cold\n(frozen)", r["primary"]["upper"]), ("Δ lower main−warm\n(frozen secondary)", r["secondary_day"]["lower"])]
for i, (n, d) in enumerate(rows):
    ax[1].errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci"][0]], [d["ci"][1] - d["diff"]]], fmt="o", color="k", capsize=3)
    ax[1].axvspan(-d["threshold"], d["threshold"], ymin=(i + .1) / 4, ymax=(i + .9) / 4, color="0.88")
ax[1].axvline(0, color="k", lw=.7); ax[1].set_yticks(range(4)); ax[1].set_yticklabels([r_[0] for r_ in rows], fontsize=8)
ax[1].set_xlabel("difference (ln), 95 % paired β-column bootstrap"); ax[1].set_title("(b) frozen decision: grey = ±ω/2", fontsize=10)
on = {"cold": 0.480, "warm": 0.170, "A1 main": 0.154}
ax[2].bar(list(on), list(on.values()), color=["#7a9cc6", "#2563a0", "#86589a"])
ax[2].set_ylabel("median onset force of the 20 lowest\nsettings per β column (N)"); ax[2].set_title("(c) same settings, other force", fontsize=10)
for s in ax:
    for k in ("top", "right"): s.spines[k].set_visible(False)
fig.tight_layout(); fig.savefig("fig_settling.png", dpi=170)
