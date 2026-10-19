"""Summary figure for REPORT.md (fig_skeptic_maintenance.png)."""
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load, column_ratios, entry_low, runs_of

BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))

# A: null intervals for the column-median ratio, A2 v1
d = np.load("null_results_A2_f0.npy", allow_pickle=True).item()
obs = d["obs"]["ratio"]
models = [("indep", "independent windows (no persistence)"),
          ("copula_best", "latent-AR copula, persistence matched"),
          ("settling", "settling-only null (b), fixed stroke threshold"),
          ("markov", "Markov (a), memory = previous window"),
          ("hyst1.0", "+ explicit hysteresis delta=1 (positive control)"),
          ("nosettle_None", "no-settling world, no hysteresis"),
          ("nosettle_1.0", "no-settling world + hysteresis delta=1")]
a = ax[0, 0]
for i, (k, lab) in enumerate(models):
    v = np.array([s["ratio"] for s in d["res"][k]], float)
    v = v[np.isfinite(v)]
    lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
    col = ORANGE if k.startswith("hyst") or k == "nosettle_1.0" else (GRAY if k.startswith("nosettle") or k == "indep" else AQUA)
    a.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
    a.plot([md], [i], "o", color=col, ms=7, mec="#fcfcfb", mew=1.5)
a.axvline(obs, color=BLUE, lw=2)
a.text(obs, len(models) - 0.35, f" observed {obs:.2f}", color=INK, va="bottom", ha="left")
a.set_yticks(range(len(models))); a.set_yticklabels([m[1] for m in models])
a.set_ylim(-0.6, len(models) - 0.1); a.invert_yaxis()
a.set_xlabel("median over columns of q10(low) / q10(entry)   (A2, 0.05 m/s, floor strokes)")
a.set_title("A  Null models reproduce 0.65; adding hysteresis does not move it", loc="left", fontsize=10, color=INK)
a.grid(axis="x", color=GRID, lw=0.8); a.set_axisbelow(True)

# B: observed vs force-fall-only prediction per column, all six folders
b = ax[0, 1]
for tag in ("A1", "A2"):
    S, speed = load(tag)
    fl = [s for s in S if s.floor]
    for f in sorted(speed):
        obsd = {c: r for c, r, *_ in column_ratios(fl)[f]}
        cells = defaultdict(list)
        for s in fl:
            if s.folder != f:
                continue
            el = entry_low(s.F, s.H)
            if el is not None:
                cells[s.col].append((el[0], min(s.F[el[2]:].min(), el[0])))
        pred = {c: np.quantile([x[1] for x in g], .1) / np.quantile([x[0] for x in g], .1) for c, g in cells.items() if len(g) >= 5}
        cs = sorted(set(obsd) & set(pred))
        hl = tag == "A2" and f == 0
        b.scatter([pred[c] for c in cs], [obsd[c] for c in cs], s=26 if hl else 16, color=BLUE if hl else GRAY,
                  edgecolor="#fcfcfb", linewidth=0.8, zorder=3 if hl else 2,
                  label="A2 0.05 m/s" if hl else ("other folders (A1 all speeds, A2 0.10/0.20)" if (tag, f) == ("A1", 0) else None))
b.plot([0.15, 1.05], [0.15, 1.05], color=INK2, lw=1, ls="--")
b.set_xlabel("prediction from force trajectory alone: q10(min force after entry) / q10(entry)")
b.set_ylabel("observed q10(low) / q10(entry)")
b.set_title("B  The statistic equals the post-entry force fall (r = 1.00)", loc="left", fontsize=10, color=INK)
b.legend(frameon=False, loc="upper left")
b.grid(color=GRID, lw=0.8); b.set_axisbelow(True)

# C: example column trajectories
c_ax = ax[1, 0]
S, speed = load("A2")
col = 12
g = sorted([s for s in S if s.folder == 0 and s.floor and s.col == col], key=lambda s: s.rank)
ent = [entry_low(s.F, s.H) for s in g]
E = [e[0] for e in ent if e is not None]; L = [e[1] for e in ent if e is not None]
FE, FM = np.quantile(E, .1), np.quantile(L, .1)
for s in g[:30]:
    c_ax.plot(s.t, s.F, color=GRID if not s.H.any() else "#d4d3cf", lw=1, zorder=1)
    h = s.H
    Fm = np.where(h, s.F, np.nan)
    c_ax.plot(s.t, Fm, color=BLUE, lw=1.6, zorder=2)
for s, e in zip(g, ent):
    if e is not None and s.rank <= 30:
        c_ax.plot(s.t[e[2]], e[0], "o", color=ORANGE, ms=4, zorder=3)
c_ax.axhline(FE, color=ORANGE, lw=1, ls="--"); c_ax.axhline(FM, color=BLUE, lw=1, ls="--")
c_ax.text(1.9, FE, " F_E,cur (q10 entries)", va="center", color=INK2, fontsize=8)
c_ax.text(1.9, FM, " F_M (q10 lows)", va="center", color=INK2, fontsize=8)
c_ax.set_yscale("log"); c_ax.set_xlim(0, 2.45)
c_ax.set_xlabel("plateau time t (s)   [floor strokes: time since bow onset = t + 1.92 s]")
c_ax.set_ylabel("current bow force (N)")
c_ax.set_title(f"C  A2 0.05 m/s, column {col} (ranks 1-30): blue = Helmholtz, dots = entries", loc="left", fontsize=10, color=INK)

# D: matched-time comparison (normalised by column F_E,cur), A2 v1
dd = ax[1, 1]
fl = [s for s in S if s.folder == 0 and s.floor]
out = column_ratios(fl)[0]
used = {c for c, *_ in out}
FEc = {c: np.quantile([entry_low(s.F, s.H)[0] for s in fl if s.col == c and entry_low(s.F, s.H) is not None], .1) for c in used}
bins = [(0.0, 0.3), (0.3, 0.8), (0.8, 1.3), (1.3, 2.5)]
for k, (a0, b0) in enumerate(bins):
    ent_v, mnt_v = [], []
    for s in fl:
        if s.col not in used:
            continue
        r = runs_of(s.H, 2); ends = r + 1
        ie = ends[0] if r.size else 10 ** 9
        inb = (s.t >= a0 - 1e-9) & (s.t < b0 - 1e-9)
        Fn = s.F / FEc[s.col]
        if ie < len(s.t) and inb[ie]:
            ent_v.append(Fn[ie])
        m = [j for j in ends if j > ie and inb[j]]
        if m:
            mnt_v.append(Fn[m].min())
    for vals, off, colr, lab in ((ent_v, -0.12, ORANGE, "new entries in this time bin"), (mnt_v, 0.12, BLUE, "maintained Helmholtz (per stroke min)")):
        vals = np.array(vals)
        q10, mn = np.quantile(vals, .1), vals.min()
        dd.plot([k + off, k + off], [mn, q10], color=colr, lw=2, solid_capstyle="round")
        dd.plot([k + off], [q10], "o", color=colr, ms=7, mec="#fcfcfb", mew=1.5, label=lab if k == 0 else None)
        dd.plot([k + off], [mn], "v", color=colr, ms=5)
        dd.text(k + off, mn * 0.82, f"n={vals.size}", ha="center", va="top", fontsize=7, color=INK2)
dd.set_yscale("log")
dd.axhline(1, color=INK2, lw=0.8, ls=":")
dd.set_xticks(range(len(bins))); dd.set_xticklabels([f"{a0:.1f}-{min(b0, 1.9):.1f} s" for a0, b0 in bins])
dd.set_ylabel("force / column F_E,cur   (circle = q10, triangle = min)")
dd.set_title("D  Matched time: late entries reach the maintained forces", loc="left", fontsize=10, color=INK)
dd.legend(frameon=False, loc="upper right")
dd.grid(axis="y", color=GRID, lw=0.8); dd.set_axisbelow(True)

fig.tight_layout()
fig.savefig("fig_skeptic_maintenance.png", dpi=150)
print("saved")
