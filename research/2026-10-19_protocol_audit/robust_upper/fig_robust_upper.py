"""Figure for robust_upper (post hoc). Reads results.json only. Usage: python -B fig_robust_upper.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
V = R["variants"]; O = R["originals"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"original": "#8a8984", "base": "#2a78d6", "U1_no020": "#eb6834", "U2_K3": "#1baf7a"}
LAB = {"original": "original interval (as published)", "base": "base, canonical bootstrap (U3)",
       "U1_no020": "U1: without 0.20 m/s", "U2_K3": "U2: K = 3"}
MK = {"original": "s", "base": "o", "U1_no020": "^", "U2_K3": "D"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def series(ax, x, est, ci, key, off):
    ax.errorbar(x + off, est, yerr=[np.array(est) - np.array([c[0] for c in ci]), np.array([c[1] for c in ci]) - np.array(est)],
                fmt=MK[key], color=COL[key], ms=6, lw=2, capsize=0, mec="white", mew=1, label=LAB[key])


fig, axs = plt.subplots(2, 3, figsize=(13, 7.6))
ORDER = ["original", "base", "U1_no020", "U2_K3"]
offs = dict(zip(ORDER, (-0.24, -0.08, 0.08, 0.24)))

# (a) pair Z exponents
ax = axs[0, 0]
P = ["A", "B", "C", "D"]; x = np.arange(4)
series(ax, x, [O["pairs"][p]["g"] for p in P], [O["pairs"][p]["g_ci95"] for p in P], "original", offs["original"])
for k in ORDER[1:]:
    series(ax, x, [V[k]["pairs"][p]["g"] for p in P], [V[k]["pairs"][p]["g_ci95"] for p in P], k, offs[k])
ax.axhline(1, color=INK2, lw=1, ls="--"); ax.axhline(0, color=GRID, lw=1)
ax.text(3.45, 1.08, "Z¹", color=INK2, ha="right")
ax.set_xticks(x, [f"{p}\n{R['variants']['base']['pairs'][p]['dlnZ']:.3f}" for p in P]); ax.set_xlabel("pair (Δln Z below)")
ax.set_ylabel("Z exponent g = Δlevel / Δln Z"); ax.set_title("(a) T1→T2 pair Z exponents (95 %)", loc="left", color=INK)

# (b) pooled exponents
for j, (key, okey, name, ref) in enumerate((("e_v", "e_v", "speed exponent e_v", None), ("e_beta", "e_beta", "β exponent e_β", -1.0))):
    ax = axs[0, 1 + j]
    ax.errorbar([0], [O["pooled_shape"][okey]], yerr=[[O["pooled_shape"][okey] - O["pooled_shape"][okey + "_ci95"][0]],
                [O["pooled_shape"][okey + "_ci95"][1] - O["pooled_shape"][okey]]], fmt=MK["original"], color=COL["original"], ms=6, lw=2, mec="white")
    xs = [1, 2, 3]
    for xi, k in zip(xs, ORDER[1:]):
        ps = V[k]["pooled_shape"]
        c = ps["diag_boot"][key + "_ci95"]
        ax.errorbar([xi - 0.1], [ps[key]], yerr=[[ps[key] - c[0]], [c[1] - ps[key]]], fmt=MK[k], color=COL[k], ms=6, lw=2, mec="white")
        if "canonical" in ps:
            c2 = ps["canonical"][key + "_ci95"]
            ax.errorbar([xi + 0.1], [ps[key]], yerr=[[ps[key] - c2[0]], [c2[1] - ps[key]]], fmt=MK[k], color=COL[k], mfc="white", ms=6, lw=2)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1, ls="--"); ax.text(3.4, ref + 0.01, "Schelleng −1", color=INK2, ha="right", va="bottom")
    else:
        ax.axhline(1, color=INK2, lw=1, ls="--"); ax.text(3.4, 0.99, "Schelleng +1", color=INK2, ha="right", va="top")
    ax.set_xticks([0, 1, 2, 3], ["original\n(200 β-clust.)", "base", "U1\nno 0.20", "U2\nK = 3"])
    ax.set_title(f"({'bc'[j]}) pooled {name}", loc="left", color=INK)
    ax.text(0.02, 0.02, "filled: diagram bootstrap (B 1000)\nopen: canonical β-cluster (B 2000)", transform=ax.transAxes, color=INK2, fontsize=8)

# (d) D_T2 strict level error and A_T2 shift
ax = axs[1, 0]
items = [("D_T2 strict err", O["D_T2"]["err"], O["D_T2"]["ci95"], "original")]
for k in ORDER[1:]:
    items.append(("D_T2 strict err", V[k]["D_T2_strict"]["err"], V[k]["D_T2_strict"]["ci95_fixed_c"], k))
for i, (_, e, c, k) in enumerate(items):
    ax.errorbar([e], [i], xerr=[[e - c[0]], [c[1] - e]], fmt=MK[k], color=COL[k], ms=6, lw=2, mec="white")
items2 = [("A_T2", O["A_T2"]["shift"], O["A_T2"]["shift_ci95"], "original")] + \
         [("A_T2", V[k]["A_T2"]["shift"], V[k]["A_T2"]["shift_ci95_fixed_law"], k) for k in ORDER[1:]]
for i, (_, e, c, k) in enumerate(items2):
    ax.errorbar([e], [i + 5], xerr=[[e - c[0]], [c[1] - e]], fmt=MK[k], color=COL[k], ms=6, lw=2, mec="white")
ax.axvspan(-0.12, 0.12, ymin=0, ymax=0.45, color=GRID, zorder=0)
ax.axvline(0, color=INK2, lw=1)
ax.axvline(np.log(1.1884 / 1.0592), color=INK2, lw=1, ls="--")
ax.text(np.log(1.1884 / 1.0592) + 0.005, 3.6, "Z¹ shift", color=INK2, fontsize=8)
ax.set_yticks([1.5, 6.5], ["D_T2 strict\nlevel error\n(±0.12 band)", "A_T2 shift vs\nA1+A2 law"])
ax.set_xlabel("log level (natural log)"); ax.set_title("(d) frozen statistics re-bootstrapped (95 %)", loc="left", color=INK)

# (e) torsion bracket
ax = axs[1, 1]
br = R["U4_torsion"]["bracket_ratio_sensitivity"]
ks = list(br)
w = 0.18
ax.bar(x - 1.5 * w, [br[ks[0]][p]["dlnZ0"] for p in P], w, color=COL["original"], label="Δln Z0 (published)")
tc = ["#6250d6", "#e87ba4", "#eda100"]
for j, k in enumerate(ks):
    ax.bar(x + (j - 0.5) * w, [br[k][p]["dlnZtot"] for p in P], w, color=tc[j], label=f"Δln Z_tot, Z_R(T1) = {k.split('=')[1]}·Z0")
ax.axhline(0, color=INK2, lw=1)
ax.axhspan(-0.02, 0.02, color=GRID, zorder=0)
ax.set_xticks(x, P); ax.set_ylabel("T2 − T1 impedance change (ln)")
ax.set_title("(e) U4 torsion bracket (assumption-dependent)", loc="left", color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# (f) random effects
ax = axs[1, 2]
U5 = R["U5_random_effects"]
rows = []
for lab in ("A,B,C", "A,B,C,D"):
    m = U5[lab]
    rows += [(f"{lab}: fixed", m["fixed"]["g"], [m["fixed"]["g"] - 1.96 * m["fixed"]["se"], m["fixed"]["g"] + 1.96 * m["fixed"]["se"]]),
             (f"{lab}: RE (PM) normal", m["PM"]["g"], m["PM"]["ci95_normal"]),
             (f"{lab}: RE (PM) HKSJ", m["PM"]["g"], m["PM"]["ci95_hksj"])]
for i, (lab, e, c) in enumerate(rows[::-1]):
    ax.errorbar([e], [i], xerr=[[e - c[0]], [c[1] - e]], fmt="o", color=COL["base"], ms=6, lw=2, mec="white")
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8)
ax.axvline(1, color=INK2, lw=1, ls="--"); ax.axvline(0, color=GRID, lw=1)
ax.set_xlabel("pooled Z exponent"); ax.set_title("(f) U5 pooled Z exponent (95 %)", loc="left", color=INK)

for a in axs.flat:
    a.grid(axis="y" if a not in (axs[1, 0], axs[1, 2]) else "x", color=GRID, lw=0.8); a.set_axisbelow(True)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Maximum-force robustness on the 11 development diagrams (post hoc; t = 0.2 s; canonical B 2000, seed 20261019)",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(HERE / "fig_robust_upper.png", dpi=160, facecolor="white")
print("wrote fig_robust_upper.png")
