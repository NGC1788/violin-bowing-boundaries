"""Is the low-force bow-force decrease in the IWK data a real force change or a sensor drift? (Claude, 2026-10-18)
Input: blocks/blocks_<A>.npz from stroke_blocks.py (50 ms blocks of every stroke).
Usage: python drift_tests.py blocks/blocks_A2.npz [reverse_rank]

T1  time or travel?  The force change is compared after the bow has stopped (time runs, travel frozen) and
    before it starts (time runs, travel 0): a time drift keeps changing, a travel-dependent change freezes.
    Also: do the three speeds (4x different time per cm) collapse on travel distance or on elapsed time?
T2  physical bound on the bridge channel: during constant-speed bowing with an oscillating string the
    mean lateral force on the string is strictly positive (v_b <F> = string losses + contact dissipation > 0),
    so bridge_dc < 0 on the plateau means a bridge-channel offset.
T3  nut vs bridge: the mean lateral force splits (1-beta) to the bridge and beta to the nut; the two
    independent sensors must give the same total. Calibrated on high-force blocks, checked on low-force ones.
T4  additive or proportional, and does the size of the change depend on beta (string stiffness at the bow
    ~ 1/(beta(1-beta)): a real position-controlled deflection change gives a larger force change where the
    contact is stiffer; an additive sensor offset does not care).
"""
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
reverse = len(sys.argv) > 2 and sys.argv[2] == "reverse_rank"
Z = np.load(path, allow_pickle=False)
D, fields = Z["data"], list(Z["fields"])
F = {name: D[:, i, :] for i, name in enumerate(fields)}
folder, trial, beta = Z["folder"], Z["trial"], Z["beta"]
dt = float(Z["block_s"])
rank = (trial - 1) % 50 + 1
if reverse:
    rank = 51 - rank
speeds = sorted(set(folder))
nS, nB = D.shape[0], D.shape[2]
t = (np.arange(nB) + 0.5) * dt

sp = F["speed"]
vmax = np.nanmax(sp, axis=1)
moving = sp > 0.02 * vmax[:, None]
onset = np.argmax(moving, axis=1)
last_move = nB - 1 - np.argmax(moving[:, ::-1], axis=1)
plateau = sp >= 0.97 * vmax[:, None]
f = F["force_dc"]
f_on = np.array([np.nanmedian(f[i, max(onset[i] - 3, 0):max(onset[i], 1)]) for i in range(nS)])
valid = np.isfinite(f_on) & (onset >= 2)
print(f"{path.name}: 긋기 {nS}, 유효 {valid.sum()}")


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.median(x) if x.size else np.nan


print("\nT1. 시간인가 이동거리인가")
print("  (a) 활이 멈춘 뒤(속도<2%) 힘 변화율 vs 움직이는 중 변화율 (N/s, 중앙값)  — 시간 흐름이면 멈춘 뒤에도 계속 변한다")
for s in speeds:
    for lo, hi in ((1, 20), (21, 35), (36, 50)):
        m = valid & (folder == s) & (rank >= lo) & (rank <= hi)
        mov, stop, pre = [], [], []
        for i in np.flatnonzero(m):
            fi = f[i]
            a, b = onset[i], last_move[i]
            if b - a > 4:
                mov.append((fi[b] - fi[a]) / ((b - a) * dt))
            post = fi[b + 2:]
            post = post[np.isfinite(post)]
            if post.size >= 4:
                stop.append((post[-1] - post[0]) / ((post.size - 1) * dt))
            prev = fi[:max(a - 1, 0)]
            prev = prev[np.isfinite(prev)]
            if prev.size >= 4:
                pre.append((prev[-1] - prev[0]) / ((prev.size - 1) * dt))
        print(f"    {s} 순위 {lo:2d}-{hi:2d}: 움직이는 중 {med(mov):+.4f}  멈춘 뒤 {med(stop):+.4f} (n {len(stop)})  출발 전 {med(pre):+.4f} (n {len(pre)})")
print("  (b) 세 속도의 힘 변화(출발 대비)가 이동거리로 맞춰지는가, 경과시간으로 맞춰지는가 (순위 1-20, 중앙값 N)")
for key, grid, unit in (("position", np.array([0.02, 0.05, 0.08, 0.11, 0.14]), "m"), ("time", np.array([0.3, 0.6, 1.0, 1.5, 2.0]), "s")):
    rows = {}
    for s in speeds:
        m = np.flatnonzero(valid & (folder == s) & (rank <= 20))
        vals = []
        for g in grid:
            d = []
            for i in m:
                if key == "position":
                    x = F["position"][i] - (F["position"][i, onset[i] - 1] if onset[i] > 0 else 0)
                    j = np.flatnonzero((x >= g) & moving[i])
                else:
                    x = t - t[onset[i]]
                    j = np.flatnonzero((x >= g) & moving[i])
                if j.size:
                    d.append(f[i, j[0]] - f_on[i])
            vals.append(med(d))
        rows[s] = vals
    spread = np.nanmean([np.nanstd([rows[s][k] for s in speeds]) for k in range(len(grid))])
    print(f"    {key:8s} 격자 {grid} {unit}: " + " | ".join(f"{s}: " + " ".join(f"{v:+.3f}" for v in rows[s]) for s in speeds)
          + f"   → 속도 간 평균 표준편차 {spread:.4f} N")

print("\nT2. 브리지 채널 물리 한계: 일정 속도 구간(평탄 0.15 s 이후)에서 브리지 DC < -0.02 N 인 블록 비율")
for s in speeds:
    out = []
    for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50)):
        m = np.flatnonzero(valid & (folder == s) & (rank >= lo) & (rank <= hi))
        neg, tot, mins = 0, 0, []
        for i in m:
            pl = np.flatnonzero(plateau[i])
            if pl.size < 5:
                continue
            blk = pl[3:]
            bd = F["bridge_dc"][i, blk]
            osc = F["bridge_p2p"][i, blk] > 0.03
            neg += int(np.sum((bd < -0.02) & osc)); tot += int(np.sum(osc))
            mins.append(np.nanmin(bd))
        out.append(f"순위 {lo}-{hi}: {neg / max(tot, 1):.0%} (최저 중앙 {med(mins):+.3f})")
    print(f"  {s}: " + "  ".join(out))

print("\nT3. 브리지와 너트, 두 독립 센서가 같은 평균 횡력을 주는가 (β ≥ 0.12)")
bb = beta[:, None] * np.ones((1, nB))
tot_b = F["bridge_dc"] / (1 - bb)
tot_n = F["nut_dc"] / bb
for s in speeds:
    m = valid & (folder == s) & (beta >= 0.12)
    hiF = m & (rank >= 41)
    X, Y = [], []
    for i in np.flatnonzero(hiF):
        pl = np.flatnonzero(plateau[i])[3:]
        X += list(tot_b[i, pl]); Y += list(tot_n[i, pl])
    X, Y = np.array(X), np.array(Y)
    ok = np.isfinite(X) & np.isfinite(Y)
    g = np.sum(X[ok] * Y[ok]) / np.sum(X[ok] ** 2) if ok.sum() else np.nan
    corr = np.corrcoef(X[ok], Y[ok])[0, 1] if ok.sum() > 3 else np.nan
    print(f"  {s}: 센 힘 보정 — 너트 기준 횡력 = {g:+.3f} × 브리지 기준 횡력 (상관 {corr:+.2f}, 블록 {ok.sum()})")
    for lo, hi in ((1, 15), (16, 30)):
        mm = m & (rank >= lo) & (rank <= hi)
        early_b, late_b, early_n, late_n, late_f, early_f = [], [], [], [], [], []
        for i in np.flatnonzero(mm):
            pl = np.flatnonzero(plateau[i])
            if pl.size < 8:
                continue
            e, l = pl[3:6], pl[-4:-1]
            early_b.append(np.nanmean(tot_b[i, e])); late_b.append(np.nanmean(tot_b[i, l]))
            early_n.append(np.nanmean(tot_n[i, e]) / g); late_n.append(np.nanmean(tot_n[i, l]) / g)
            early_f.append(np.nanmean(f[i, e])); late_f.append(np.nanmean(f[i, l]))
        print(f"    순위 {lo}-{hi} (n {len(early_b)}): 브리지 기준 횡력 {med(early_b):+.3f}→{med(late_b):+.3f} | 너트 기준(보정) {med(early_n):+.3f}→{med(late_n):+.3f} | 힘 {med(early_f):.3f}→{med(late_f):.3f} N")

print("\nT4. 감소량은 더해지는 꼴인가 비례하는가, β(현 강성)에 따라 달라지는가 — 이동 10 cm 지점 힘 변화")
for s in speeds:
    lines = []
    for lo, hi in ((21, 30), (31, 40), (41, 50)):
        for blo, bhi in ((0.02, 0.05), (0.05, 0.10), (0.10, 0.21)):
            m = np.flatnonzero(valid & (folder == s) & (rank >= lo) & (rank <= hi) & (beta >= blo) & (beta < bhi))
            d = []
            for i in m:
                x = F["position"][i] - (F["position"][i, onset[i] - 1] if onset[i] > 0 else 0)
                j = np.flatnonzero((x >= 0.10) & moving[i])
                if j.size:
                    d.append(f[i, j[0]] - f_on[i])
            lines.append(f"순위{lo}-{hi} β{blo}-{bhi}: {med(d):+.3f}")
    print(f"  {s}: " + "  ".join(lines))
