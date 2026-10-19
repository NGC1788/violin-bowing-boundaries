"""Figure: bow-force change vs bow travel and vs elapsed time (A2, A1), by speed (Claude, 2026-10-18)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"; plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw={"hspace": 0.4})
for row, (name, path, rev) in enumerate((("A2", "blocks/blocks_A2.npz", False), ("A1", "blocks/blocks_A1.npz", True))):
    Z = np.load(path); D = Z["data"]; fl = list(Z["fields"]); folder = Z["folder"]; trial = Z["trial"]
    rank = (trial - 1) % 50 + 1
    rank = 51 - rank if rev else rank
    sp, f, pos = D[:, fl.index("speed")], D[:, fl.index("force_dc")], D[:, fl.index("position")]
    vmax = np.nanmax(sp, 1); mov = sp > 0.02 * vmax[:, None]; on = np.argmax(mov, 1)
    for k, s in enumerate(sorted(set(folder))):
        m = np.flatnonzero((folder == s) & (rank <= 20) & (on >= 2))
        xs, ts, ys = [], [], []
        for i in m:
            j = np.flatnonzero(mov[i]); f0 = np.nanmedian(f[i, on[i] - 2:on[i]])
            xs.append(pos[i, j] - pos[i, on[i] - 1]); ts.append((j - on[i]) * 0.05); ys.append(f[i, j] - f0)
        x, t, y = map(np.concatenate, (xs, ts, ys))
        for ax, var, bins in ((axes[row, 0], x * 100, np.linspace(0, 16, 17)), (axes[row, 1], t, np.linspace(0, 4, 21))):
            c = 0.5 * (bins[1:] + bins[:-1]); med = [np.nanmedian(y[(var >= a) & (var < b)]) if np.sum((var >= a) & (var < b)) > 50 else np.nan for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(c, med, "o-", ms=3, lw=2, label=f"{['0.05', '0.10', '0.20'][k]} m/s")
    axes[row, 0].set_xlabel("활이 움직인 거리 (cm)"); axes[row, 1].set_xlabel("움직이기 시작한 뒤 시간 (s)")
    for ax in axes[row]:
        ax.set_ylabel("활 힘 변화 (N, 출발 대비)"); ax.axhline(0, color="#999", lw=0.8); ax.legend(fontsize=8)
    axes[row, 0].set_title(f"{name}: 이동 거리 기준 — " + ("세 속도가 한 곡선에 겹친다" if name == "A2" else "시간 기준보다 잘 겹친다(0.10 m/s 약간 어긋남)"), fontsize=11)
    axes[row, 1].set_title(f"{name}: 시간 기준 — 속도마다 갈라진다", fontsize=11)
fig.suptitle("약한 힘(순위 1–20) 긋기의 활 힘 감소는 시간이 아니라 활이 이동한 거리에 따라 일어난다 → 시간에 따라 흐르는 센서 오차가 아니다", fontsize=11)
fig.savefig("fig_drift_test.png", dpi=150, bbox_inches="tight"); print("saved")
