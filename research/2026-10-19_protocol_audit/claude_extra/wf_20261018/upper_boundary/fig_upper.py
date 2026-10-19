"""Figures for the upper-boundary track.
fig_schelleng_t02.png: Schelleng diagrams (A1/A2 x 3 speeds, t = 0.2 s, current window force) with labels,
  per-column upper/lower boundary intervals and the joint interval-censored fits.
fig_exponents.png: speed and beta exponents of the upper vs lower boundary (A1, A2, 4 plateau times, column-bootstrap
  95 % intervals from results_main.json) with Schelleng and Schoonderwaldt 2008 (same estimator) references.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ub

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def schelleng():
    T = 0.2
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    for r, tag in enumerate(("A1", "A2")):
        sp = ub.Spec(tag)
        groups = ub.column_groups(sp, T)
        Du = ub.as_array(ub.table(sp, T, "upper", groups=groups))
        Dl = ub.as_array(ub.table(sp, T, "lower", groups=groups), drop_none=False)
        cu, _, _ = ub.fit(Du); cl, _, _ = ub.fit(Dl)
        for f in range(3):
            ax = axes[r, f]
            pts = {"H": [], "M": [], "O": []}
            for (ff, c), g in groups.items():
                if ff != f:
                    continue
                for x in g:
                    key = "H" if x[1] else ("M" if x[2] == 1 else "O")
                    pts[key].append((x[3], x[0]))
            for key, col, mk, lab in (("O", AQUA, "x", "aperiodic / ambiguous / no osc. / subharmonic"),
                                      ("M", ORANGE, "+", "multiple slip"), ("H", BLUE, "s", "Helmholtz")):
                p = np.array(pts[key])
                if len(p):
                    ax.scatter(p[:, 0], p[:, 1], s=7 if key == "H" else 9, marker=mk, color=col, linewidths=0.7,
                               facecolors="none" if key == "H" else None, label=lab, alpha=0.8)
            for D, colr in ((Du, INK), (Dl, MUTED)):
                for row in D[D[:, 0] == f]:
                    b = None
                    lo, hi = row[4], row[5]
                    beta = np.exp(row[3])
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.plot([beta, beta], [np.exp(lo), np.exp(hi)], color=colr, lw=2.2, solid_capstyle="butt")
                    elif np.isfinite(lo):
                        ax.annotate("", xy=(beta, np.exp(lo) * 1.35), xytext=(beta, np.exp(lo)),
                                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.0, mutation_scale=6))
            bb = np.geomspace(0.019, 0.21, 50)
            lv = np.log(sp.speed[f])
            ax.plot(bb, np.exp(cu[0] + cu[1] * lv + cu[2] * np.log(bb)), color=INK, lw=1.2, ls="--")
            ax.plot(bb, np.exp(cl[0] + cl[1] * lv + cl[2] * np.log(bb)), color=MUTED, lw=1.2, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylim(0.01, 6)
            ax.set_title(f"{tag}, v = {sp.speed[f]:.2f} m/s", color=INK, fontsize=10)
            if r == 1:
                ax.set_xlabel("relative bow position β")
            if f == 0:
                ax.set_ylabel("current window bow force (N)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK, lw=2.2), plt.Line2D([], [], color=INK, lw=1.2, ls="--"),
          plt.Line2D([], [], color=MUTED, lw=2.2), plt.Line2D([], [], color=MUTED, lw=1.2, ls=":")]
    l += ["upper boundary interval (arrow: right-censored)", "upper fit (joint, interval-censored)",
          "lower boundary interval", "lower fit"]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Schelleng diagrams at plateau time t = 0.2 s (0.2 s windows, K = 2 runs)", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(ub.HERE / "fig_schelleng_t02.png", dpi=150)
    plt.close(fig)


def exponents():
    R = json.load(open(ub.HERE / "results_main.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (coef, ref, name) in enumerate((("a", 1.0, "speed exponent a"), ("b", None, "β exponent b"))):
        ax = axes[j]
        y = 0
        ticks, labels = [], []
        for kind, ref_k in (("upper", 1.0 if coef == "a" else -1.0), ("lower", 1.0 if coef == "a" else -2.0)):
            for tag, col in (("A1", ORANGE), ("A2", BLUE)):
                for t in ub.TIMES:
                    r = R[kind][f"{tag}_t{t}"]
                    lo, hi = r[f"{coef}_ci"]
                    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
                    ax.plot([r[coef]], [y], "o", color=col, ms=5, mec="white", mew=1)
                    ticks.append(y); labels.append(f"{kind} {tag} t={t:.1f}")
                    y += 1
                y += 0.5
            ax.axvline(ref_k, color=GRAY, lw=0.8, ls="--")
            y += 1
        # Schoonderwaldt 2008 D4 (same estimator; schoonderwaldt_upper.log)
        s_up = {"a": (0.619, 0.53, 0.71), "b": (-0.649, -0.73, -0.57)}[coef]
        s_lo = {"a": (-0.085, -0.24, 0.09), "b": (-2.179, -2.52, -1.94)}[coef]
        for val, lab in ((s_up, "Schoonderwaldt 2008 D4 upper"), (s_lo, "Schoonderwaldt 2008 D4 lower")):
            ax.plot([val[1], val[2]], [y, y], color=AQUA, lw=2)
            ax.plot([val[0]], [y], "D", color=AQUA, ms=5, mec="white", mew=1)
            ticks.append(y); labels.append(lab); y += 1
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(name + ("  (dashed: Schelleng, upper +1 / lower +1)" if coef == "a" else "  (dashed: Schelleng, upper −1 / lower −2)"))
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    fig.suptitle("Upper vs lower boundary exponents, column-bootstrap 95 % intervals (orange A1, blue A2)", color=INK)
    fig.tight_layout()
    fig.savefig(ub.HERE / "fig_exponents.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    schelleng()
    exponents()
