"""Figure for thermal_upper (Claude, 2026-10-19). Reads analysis.json, analysis2.json, FROZEN_v2.json -> fig_thermal_upper.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
A = json.loads((HERE / "analysis.json").read_text())
A2 = json.loads((HERE / "analysis2.json").read_text())
V2 = json.loads((HERE / "FROZEN_v2.json").read_text())
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_PLAIN, C_TH, C_EMP = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

# (a) exponents vs similarity lines
a = ax[0]
ev = np.linspace(0.3, 1.1, 50)
a.plot(ev, -(1 + ev) / 2, color=C_TH, label="thermal, power-controlled friction: e_β = −(1+e_v)/2")
a.plot(ev, -1.25 * (1 + ev) / 2, color=C_TH, ls="--", lw=1.5, label="same with extra β^−0.25 (m = 1.25, post hoc)")
a.plot(ev, -ev, color=INK2, ls=":", lw=1.5, label="slip-velocity friction: e_β = −e_v")
a.plot([1], [-1], marker="s", ms=8, color=C_PLAIN, ls="none", label="plain Schelleng (1, −1)")
for t, d in A["per_diagram"].items():
    a.errorbar(d["e_v"], d["e_beta"], xerr=[[d["e_v"] - d["e_v_ci"][0]], [d["e_v_ci"][1] - d["e_v"]]],
               yerr=[[d["e_beta"] - d["e_beta_ci"][0]], [d["e_beta_ci"][1] - d["e_beta"]]], fmt="o", ms=4, color=INK, ecolor="#b9b8b2", elinewidth=1)
    a.annotate(t, (d["e_v"], d["e_beta"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=INK2)
p = A["pooled_free"]
a.plot(p["e_v"], p["e_beta"], marker="D", ms=8, color=C_EMP, ls="none", label="pooled 11 diagrams (free censored fit)")
a.set_xlabel("speed exponent e_v (upper boundary)"); a.set_ylabel("β exponent e_β")
a.set_title("(a) measured exponents vs friction-law identities", fontsize=9.5, loc="left", color=INK)
a.legend(fontsize=6.3, loc="lower left", frameon=False)
a.set_xlim(0.3, 1.1); a.set_ylim(-1.7, -0.5)

# (b) held-out (LODO) mean NLL difference vs plain Schelleng
b = ax[1]
names = [("T1k_peak", "thermal μs 0.84 + k (1 par)", C_TH), ("T1k_break", "thermal μs 1.5 + k (1 par)", C_TH), ("T1mu", "thermal, μs fitted (1 par)", C_TH),
         ("T2km_break", "thermal + β^−m (2 par, post hoc)", C_TH), ("E3_free_Z", "free exponents + Z (3 par)", C_EMP)]
for i, (m, lab, col) in enumerate(names):
    src = A2["LODO_summary"][m] if m in A2["LODO_summary"] else None
    if m in A["LODO_paired_vs_S1"]:
        x = A["LODO_paired_vs_S1"][m]; mean, ci, w = x["nll_diff_mean"], x["nll_diff_ci"], x["wins_nll"]
    else:
        mean, ci, w = src["dnll_vs_S1"], src["dnll_ci"], src["wins"]
    b.barh(i, mean, height=0.55, color=col, edgecolor=SURF, linewidth=2)
    b.errorbar(mean, i, xerr=[[mean - ci[0]], [ci[1] - mean]], color=INK, capsize=3, lw=1)
    b.text(max(ci[1], 0) + 0.004, i, f"{mean:+.3f}  ({w}/11 better)", va="center", fontsize=7, color=INK)
b.axvline(0, color=INK2, lw=1)
b.set_yticks(range(len(names))); b.set_yticklabels([n[1] for n in names], fontsize=7.5)
b.set_xlabel("held-out NLL per column − plain Schelleng (LODO; < 0 = better)")
b.set_title("(b) leave-one-diagram-out vs plain Schelleng c·Zv/β", fontsize=9.5, loc="left", color=INK)
b.set_xlim(-0.11, 0.12); b.grid(axis="y", visible=False)

# (c) frozen high-force predictions, B T2, v = 0.1 m/s
c = ax[2]
beta = np.array([0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07])
for m, lab, col, ls in (("S1_strict", "plain Schelleng (e_β −1.00)", C_PLAIN, "-"), ("T1k_break", "thermal μs 1.5 (e_β −0.94)", C_TH, "-"),
                        ("E3_free_Z", "free exponents (e_β −1.06)", C_EMP, "--")):
    tab = V2["high_force"]["B T2"][m]["Fmax_N"]["0.10"]
    y = [tab[f"{b_:.3f}"] for b_ in beta]
    c.plot(beta, y, color=col, ls=ls, marker="o", ms=4, label=lab)
c.axhspan(4, 12, color="#efeeea", zorder=0)
c.text(0.021, 4.3, "high-force archives: 4–12 N", fontsize=7, color=INK2, ha="left")
c.set_xscale("log"); c.set_yscale("log")
c.legend(fontsize=7, frameon=False, loc="upper right")
c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); c.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
c.set_xticks([0.02, 0.03, 0.05, 0.07]); c.set_xticklabels(["0.02", "0.03", "0.05", "0.07"])
c.set_yticks([3, 4, 6, 8, 12, 16]); c.set_yticklabels(["3", "4", "6", "8", "12", "16"])
c.set_xlabel("β"); c.set_ylabel("frozen F_max at v = 0.1 m/s (N), B T2 string")
c.set_title("(c) frozen high-force predictions (FROZEN_v2)", fontsize=9.5, loc="left", color=INK)
fig.tight_layout()
fig.savefig(HERE / "fig_thermal_upper.png", dpi=170)
print("wrote fig_thermal_upper.png")
