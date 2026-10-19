"""English Figs 1-3 for draft v2 (Claude, 2026-10-19). Re-plots saved results only; no fitting, no confirmation data.
Fig 1: protocol audit (A2 force change vs travel and time; attack vs trial index, A1 reverse scan vs A2).
Fig 2: upper-boundary level vs published ln Z and ln EI (novelty/upper_law/upper_law_t0.2.json).
Fig 3: minimum-force association vs prediction (astra_20261019 protocol_results + nested results).
Originals with Korean labels are untouched."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path.home() / "violin-rne"
OUT = R / "paper"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig1():
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    Z = np.load(R / "claude_extra/blocks/blocks_A2.npz"); D = Z["data"]; fl = list(Z["fields"]); folder = Z["folder"]; trial = Z["trial"]
    rank = (trial - 1) % 50 + 1
    sp, f, pos = D[:, fl.index("speed")], D[:, fl.index("force_dc")], D[:, fl.index("position")]
    vmax = np.nanmax(sp, 1); mov = sp > 0.02 * vmax[:, None]; on = np.argmax(mov, 1)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for k, s in enumerate(sorted(set(folder))):
        m = np.flatnonzero((folder == s) & (rank <= 20) & (on >= 2))
        xs, ts, ys = [], [], []
        for i in m:
            j = np.flatnonzero(mov[i]); f0 = np.nanmedian(f[i, on[i] - 2:on[i]])
            xs.append(pos[i, j] - pos[i, on[i] - 1]); ts.append((j - on[i]) * 0.05); ys.append(f[i, j] - f0)
        x, t, y = map(np.concatenate, (xs, ts, ys))
        for ax, var, bins in ((axes[0], x * 100, np.linspace(0, 16, 17)), (axes[1], t, np.linspace(0, 4, 21))):
            c = 0.5 * (bins[1:] + bins[:-1])
            med = [np.nanmedian(y[(var >= a) & (var < b)]) if np.sum((var >= a) & (var < b)) > 50 else np.nan for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(c, med, "o-", ms=3, lw=2, label=f"{['0.05', '0.10', '0.20'][k]} m/s")
    axes[0].set(xlabel="bow travel (cm)", ylabel="measured force change from onset (N)", title="(a) A2, force ranks 1-20: vs travel")
    axes[1].set(xlabel="time since bow started (s)", title="(b) same strokes: vs elapsed time")
    for ax in axes:
        ax.axhline(0, color="#999", lw=0.8); ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    files = {"A2 (ascending scan)": R / "claude_extra/ramp_meta_A2.csv",
             "A1 (reverse scan)": R / "claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv"}
    for (lab, p), col in zip(files.items(), ("#2563a0", "#d38325")):
        by = defaultdict(list)
        for r in csv.DictReader(open(p)):
            if not r["error"] and abs(float(r["v_max"]) - 0.05) < 1e-3:
                by["x"].append((float(r["beta"]), int(r["trial"]), float(r["force_at_onset"]), float(r["accel"])))
        s = sorted(by["x"])
        cols = [s[i:i + 50] for i in range(0, len(s), 50)]
        c = cols[24]  # one mid-beta column (beta ~0.08)
        c = sorted(c, key=lambda z: z[2])
        ax.semilogy(np.arange(1, 51), [z[3] for z in c], "o-", ms=3, color=col, label=f"{lab}, β={np.median([z[0] for z in c]):.3f}")
    ax.set(xlabel="onset-force rank within the β column (1 = lowest)", ylabel="attack acceleration (m/s²)",
           title="(c) attack follows recording order, 0.05 m/s")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "fig1_protocol_en.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def fig2():
    Rj = json.load(open(R / "novelty/upper_law/upper_law_t0.2.json"))
    PROP = {"A T1": (1.0592, 3.032), "B T1": (1.0998, 4.397), "C T1": (1.0525, 3.142), "D T1": (1.0398, 5.246),
            "A T2": (1.1884, 3.969), "B T2": (1.1899, 7.295), "C T2": (1.1154, 3.969)}
    COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
    dg = Rj["diagrams"]; pts = []
    for tag, lab in (("A1", "A1"), ("A2", "A2"), ("B1", "B1"), ("C1", "C1"), ("D1", "D1"), ("A_T2", "A_T2"), ("B_T2s1", "B_T2 s1"), ("C_T2", "C_T2")):
        pts.append((lab, dg[tag]["group"], dg[tag]["level"], *dg[tag]["level_ci"]))
    p6 = Rj["levels"]["B_T2p6"]
    pts.append(("B_T2 P6", "B T2", p6, np.mean([dg[t]["level_ci"][0] for t in ("B_T2r1", "B_T2r2")]),
                np.mean([dg[t]["level_ci"][1] for t in ("B_T2r1", "B_T2r2")])))
    lnc = Rj["transfer_T1_to_T2"]["Schelleng Z^1 (slope fixed)"]["coef"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, j, xl in ((axes[0], 0, "ln Z (published, kg/s)"), (axes[1], 1, "ln EI (published, 1e-4 N m²)")):
        for lab, g, y, lo, hi in pts:
            c = COL[g[0]]
            jit = ({"A1": -0.003, "A2": 0.004, "C1": -0.004, "B_T2 s1": -0.004, "B_T2 P6": 0.004} if j == 0 else
                   {"A1": -0.012, "A2": 0.012, "A_T2": -0.012, "C_T2": 0.012, "B_T2 s1": -0.012, "B_T2 P6": 0.012}).get(lab, 0.0)
            x = np.log(PROP[g][j]) + jit
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=1.2, alpha=0.8)
            ax.plot(x, y, "o", ms=8, mfc=(c if g.endswith("T2") else "white"), mec=c, mew=2)
            off = {"A1": (-14, 9), "C1": (-22, -14) if j == 0 else (8, 3), "A2": (6, -11) if j == 0 else (-4, -16), "B_T2 P6": (6, -11),
                   "A_T2": (-36, 8) if j == 1 else (6, -14), "C_T2": (6, -12) if j == 1 else (6, 4)}.get(lab, (6, 4))
            ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8)
        for a, b in (("A T1", "A T2"), ("B T1", "B T2"), ("C T1", "C T2")):
            ya = np.mean([p[2] for p in pts if p[1] == a]); yb = np.mean([p[2] for p in pts if p[1] == b])
            ax.annotate("", xy=(np.log(PROP[b][j]), yb), xytext=(np.log(PROP[a][j]), ya),
                        arrowprops=dict(arrowstyle="-|>", color=COL[a[0]], lw=1.2, alpha=0.55, shrinkA=7, shrinkB=7))
        if j == 0:
            xx = np.linspace(0.03, 0.19, 10)
            ax.plot(xx, lnc + xx, color="#8a8984", lw=2, ls="--")
            ax.text(0.12, lnc + 0.12 - 0.04, "slope 1 in ln Z, one constant\nfrom the five T1 diagrams\n(on empirical-shape levels)", fontsize=8, color="#8a8984", va="top")
        ax.set_xlabel(xl); ax.grid(color="#e6e5e0", lw=0.6)
    axes[0].set_ylabel("upper-boundary level vs the A1+A2 law (ln)")
    axes[0].plot([], [], "o", mfc="white", mec="k", mew=2, label="T1"); axes[0].plot([], [], "o", mfc="k", mec="k", label="T2")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Maximum-force levels, t = 0.2 s; bars: column-bootstrap 95 % intervals (post hoc)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig2_upper_en.png", dpi=180); plt.close(fig)


def fig3():
    A = R / "astra_20261019"
    protocol = json.load(open(A / "protocol_results/results.json")); nested = json.load(open(A / "nested_protocol_results.json"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    a = protocol["variants"]["measured_K2_t0.2"]; tags = protocol["tags"]; x = np.array(protocol["descriptor"])
    y = np.array([a["levels"][t] for t in tags])
    colors = {"A": "#2563a0", "B": "#d38325", "C": "#278270", "D": "#86589a"}
    ax = axes[0, 0]
    for typ in colors:
        m = np.array([t[0] == typ for t in tags]); ax.scatter(x[m], y[m], c=colors[typ], s=45, label=typ, zorder=3)
    for tag, xx, yy in zip(tags, x, y):
        ax.annotate(tag + (" (reverse scan)" if tag == "A1" else ""), (xx, yy), xytext=(6, 6), textcoords="offset points", fontsize=8)
    xx = np.linspace(x.min(), x.max(), 80); ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color="#777", ls="--", lw=1)
    ax.set(xlabel="low-force load decrease dec5_pl (N; negative = decrease)", ylabel="lower-boundary level (ln, vs A1+A2 shape)",
           title=f"(a) post hoc association, r = {a['association']['r']:.3f}")
    ax.legend(title="string type", ncol=4, loc="lower right", frameon=False, fontsize=8)
    for k, (key, title) in enumerate([("level_mae", "(b) diagram-level error: descriptor helps"),
                                      ("diagram_mean_column_median_error", "(c) finite-column error: descriptor hurts")]):
        ax = axes[0, 1] if k == 0 else axes[1, 0]
        for shift, model, label, color in [(-.18, "mean", "mean level only", "#a3a9b2"), (.18, "dec5_pl", "load-decrease descriptor", "#2563a0")]:
            vals = [nested["variants"]["measured_" + g]["models"][model][key] for g in ("leave_diagram_out", "leave_type_out")]
            bars = ax.bar(np.arange(2) + shift, vals, width=.34, label=label, color=color); ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set(xticks=np.arange(2), xticklabels=["leave one diagram out", "leave one type out"], ylabel="absolute log error", title=title, ylim=(0, .41))
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax = axes[1, 1]
    for j, (k, label) in enumerate([("measured_K2_t0.2", "measured force axis"), ("nominal_rank_K2_t0.2", "nominal force axis")]):
        v = protocol["variants"][k]["association"]["cv"]
        vals = [v[g]["dec5_pl"]["cv_r2"] for g in ("leave_diagram_out", "leave_type_out")]
        bars = ax.bar(np.arange(2) + (j - .5) * .36, vals, width=.34, label=label, color=["#2563a0", "#d38325"][j]); ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(0, color="#777", lw=.8)
    ax.set(xticks=np.arange(2), xticklabels=["leave one diagram out", "leave one type out"], ylabel="cross-validated R² (regression step)",
           title="(d) weaker on the nominal force axis", ylim=(-.42, .92))
    ax.legend(frameon=False, fontsize=9)
    fig.supxlabel("10 development diagrams, t = 0.2 s, K = 2; (b, c) refit the reference shape in every fold, (a, d) fixed A1+A2 shape; post hoc", fontsize=9)
    fig.savefig(OUT / "fig3_lower_en.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); print("saved fig1_protocol_en.png fig2_upper_en.png fig3_lower_en.png")
