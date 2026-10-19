"""Figure: upper-boundary level of every local diagram vs published log Z and log EI (reads upper_law_t0.2.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"; plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).parent
R = json.load(open(HERE / "upper_law_t0.2.json"))
PROP = {"A T1": (1.0592, 3.032), "B T1": (1.0998, 4.397), "C T1": (1.0525, 3.142), "D T1": (1.0398, 5.246),
        "A T2": (1.1884, 3.969), "B T2": (1.1899, 7.295), "C T2": (1.1154, 3.969)}
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
INK, MUTED = "#2b2b2a", "#8a8984"
dg = R["diagrams"]
pts = []  # (label, group, level, lo, hi)
for tag, lab in (("A1", "A1"), ("A2", "A2"), ("B1", "B1"), ("C1", "C1"), ("D1", "D1"), ("A_T2", "A_T2"), ("B_T2s1", "B_T2 s1"), ("C_T2", "C_T2")):
    pts.append((lab, dg[tag]["group"], dg[tag]["level"], *dg[tag]["level_ci"]))
p6 = R["levels"]["B_T2p6"]
p6lo = np.mean([dg[t]["level_ci"][0] for t in ("B_T2r1", "B_T2r2")]); p6hi = np.mean([dg[t]["level_ci"][1] for t in ("B_T2r1", "B_T2r2")])
pts.append(("B_T2 P6", "B T2", p6, p6lo, p6hi))
lnc = R["transfer_T1_to_T2"]["Schelleng Z^1 (slope fixed)"]["coef"][0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, j, xl in ((axes[0], 0, "ln Z (공개값, kg/s)"), (axes[1], 1, "ln EI (공개값, 1e-4 N m²)")):
    for lab, g, y, lo, hi in pts:
        x = np.log(PROP[g][j]); c = COL[g[0]]
        jit = ({"A1": -0.003, "A2": 0.004, "C1": -0.004, "B_T2 s1": -0.004, "B_T2 P6": 0.004} if j == 0 else
               {"A1": -0.012, "A2": 0.012, "A_T2": -0.012, "C_T2": 0.012, "B_T2 s1": -0.012, "B_T2 P6": 0.012}).get(lab, 0.0)
        ax.errorbar(x + jit, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=1.2, capsize=0, alpha=0.8, zorder=2)
        ax.plot(x + jit, y, "o", ms=8, mfc=(c if g.endswith("T2") else "white"), mec=c, mew=2, zorder=3)
        off = {"A1": (-14, 9), "C1": (-22, -14) if j == 0 else (8, 3), "A2": (6, -11) if j == 0 else (-4, -16), "B_T2 P6": (6, -11),
               "A_T2": (-36, 8) if j == 1 else (6, -14),
               "C_T2": (6, -12) if j == 1 else (6, 4)}.get(lab, (6, 4))
        ax.annotate(lab, (x + jit, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK)
    for a, b in (("A T1", "A T2"), ("B T1", "B T2"), ("C T1", "C T2")):
        ya = np.mean([p[2] for p in pts if p[1] == a]); yb = np.mean([p[2] for p in pts if p[1] == b])
        ax.annotate("", xy=(np.log(PROP[b][j]), yb), xytext=(np.log(PROP[a][j]), ya),
                    arrowprops=dict(arrowstyle="-|>", color=COL[a[0]], lw=1.2, alpha=0.55, shrinkA=7, shrinkB=7), zorder=1)
    if j == 0:
        xx = np.linspace(0.03, 0.19, 10)
        ax.plot(xx, lnc + xx, color=MUTED, lw=2, ls="--", zorder=0)
        ax.text(0.135, lnc + 0.135 - 0.03, "Schelleng: 기울기 1, 상수 1개\n(T1 현 5개로 보정)", fontsize=8, color=MUTED, ha="left", va="top")
    ax.set_xlabel(xl, fontsize=10)
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("최대 활 힘 수준 (A1+A2 법칙 대비 log 이동)", fontsize=10)
axes[0].plot([], [], "o", mfc="white", mec=INK, mew=2, label="저장력 T1"); axes[0].plot([], [], "o", mfc=INK, mec=INK, label="고장력 T2")
axes[0].legend(fontsize=8.5, loc="upper left", frameon=False)
yd = R["yardsticks"]
fig.suptitle(f"최대 활 힘 수준 — 도표 9개(현 8개), t = 0.2 s, 막대 = 열 부트스트랩 95% 구간 | 반복 잣대: rep1-rep2 {yd['B_T2r1-B_T2r2']['diff']:+.3f}, "
             f"s1-P6 {yd['B_T2s1-B_T2p6']['diff']:+.3f}, A1-A2 {yd['A1-A2']['diff']:+.3f}", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_upper_law.png", dpi=160)
print("saved", HERE / "fig_upper_law.png")
