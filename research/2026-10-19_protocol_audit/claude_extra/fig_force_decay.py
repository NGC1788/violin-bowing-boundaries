"""Figure: bow force during IWK A2 strokes by nominal force rank, and the attack-acceleration design (Claude, 2026-10-18)."""
import csv, gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["mathtext.fontset"] = "stixsans"
HERE = Path(__file__).parent
meta = {}
for r in csv.DictReader((HERE / "ramp_meta_A2.csv").open()):
    if not r["error"]:
        meta[(r["folder"].split("/")[-1], int(r["trial"]))] = r
series = defaultdict(lambda: defaultdict(list))
level = {}
with gzip.open(Path.home() / "violin-rne/timeres/windows_full.csv.gz", "rt") as fh:
    for r in csv.DictReader(fh):
        if r["error"]:
            continue
        key = (r["condition"], int(r["trial"]))
        rank = int(r["force_rank"]); level[key] = (int(r["beta_level"]), rank)
        series[(r["condition"], (rank - 1) // 10)][round(float(r["t"]), 2)].append(float(r["force"]))
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), gridspec_kw={"hspace": 0.45})
colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
for j, (cond, v) in enumerate((("v1", 0.05), ("v2", 0.10), ("v3", 0.20))):
    ax = axes[0, j]
    for b in range(5):
        d = series[(cond, b)]
        ts = sorted(t for t in d if len(d[t]) >= 50)
        med = [np.median(d[t]) for t in ts]
        onset = np.median([float(meta[k]["force_at_onset"]) for k in meta if k[0] == cond and (level.get(k, (0, 0))[1] - 1) // 10 == b])
        ax.plot(ts, np.maximum(med, 1e-3), color=colors[b], lw=2, label=f"힘 순위 {10*b+1}–{10*b+10}")
        ax.plot([-0.15], [onset], "o", color=colors[b])
    ax.set_yscale("log"); ax.set_ylim(5e-3, 5); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3], ["0.01", "0.03", "0.1", "0.3", "1", "3"]); ax.minorticks_off(); ax.set_title(f"{v} m/s: 활 힘 (중앙값)", fontsize=11)
    ax.set_xlabel("평탄 구간 시작 후 시간 (s)\n● = 활이 움직이기 직전 힘", fontsize=9); ax.axvline(0, color="#999", lw=0.8)
    if j == 0:
        ax.set_ylabel("활 힘 (N)"); ax.legend(fontsize=8)
    ax2 = axes[1, j]
    A = np.full((40, 50), np.nan)
    for k, r in meta.items():
        if k[0] == cond and k in level:
            bl, rk = level[k]; A[bl, rk - 1] = float(r["accel"])
    im = ax2.imshow(np.log10(A), origin="lower", aspect="auto", cmap="magma", vmin=-1.7, vmax=1)
    ax2.set_xlabel("힘 순위 (1 = 가장 약함)"); ax2.set_title(f"{v} m/s: 출발 가속도 log10(m/s²)", fontsize=11)
    if j == 0:
        ax2.set_ylabel("β 열 (0 = 0.02, 39 = 0.20)")
fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8)
fig.suptitle("IWK A2 표본: 약한 힘의 긋기는 힘이 일정하지 않고 계속 줄어든다(위), 출발 가속도는 β·힘에 따라 설계상 다르다(아래)", fontsize=12)
fig.savefig(HERE / "fig_force_decay.png", dpi=150, bbox_inches="tight")
print("saved")
