"""Critical slowing down: does the time a stroke needs to lock into Helmholtz motion
grow as its bow force approaches the steady-state minimum force?

For each stroke the lock time is the first window start from which at least 80 %
of the remaining windows are Helmholtz (0 if it starts locked, censored if it never
locks on the plateau). Distance to the boundary is log F - log F_late, where F_late is
the column's lower boundary from the last common window time of that speed.
"""
import csv, gzip, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

FILE = sys.argv[1] if len(sys.argv) > 1 else "windows_full.csv.gz"
rows = [r for r in csv.DictReader(gzip.open(FILE, "rt")) if not r["error"]]
strokes = defaultdict(list)
for r in rows:
    strokes[(r["condition"], int(r["trial"]))].append((float(r["t"]), r["label"] == "helmholtz", float(r["force"]), float(r["beta"]), int(r["beta_level"])))

def lock_time(seq):
    seq.sort()
    h = np.array([s[1] for s in seq]); t = np.array([s[0] for s in seq])
    for i in range(len(h)):
        if h[i] and h[i:].mean() >= 0.8:
            return t[i], False
    return t[-1], True          # never locked within the plateau: censored

late_t = {}
for (c, _), seq in strokes.items():
    late_t[c] = min(late_t.get(c, 1e9), max(s[0] for s in seq))
# late boundary per column (first H-run of 2 at the last common time)
cols = defaultdict(list)
for (c, n), seq in strokes.items():
    last = [s for s in seq if abs(s[0] - late_t[c]) < 1e-6]
    if last:
        cols[(c, last[0][4])].append((last[0][2], last[0][1]))
bound = {}
for key, cells in cols.items():
    cells.sort(); h = [x[1] for x in cells]
    i = next((k for k in range(len(h) - 1) if h[k] and h[k + 1]), None)
    if i is not None and i > 0:
        bound[key] = np.log(cells[i][0])
print(f"긋기 {len(strokes)}개, 마지막 공통 시각 {late_t}")
for c in ("v1", "v2", "v3"):
    d, T, cens = [], [], []
    for (cc, n), seq in strokes.items():
        if cc != c: continue
        key = (c, seq[0][4])
        if key not in bound: continue
        dist = np.log(np.mean([s[2] for s in seq])) - bound[key]
        if 0 < dist < 1.2:                         # strokes above the late boundary only
            tl, cen = lock_time(seq)
            d.append(dist); T.append(tl); cens.append(cen)
    d, T, cens = map(np.array, (d, T, cens))
    rho, p = spearmanr(d, T)
    bins = [(0, .15), (.15, .3), (.3, .5), (.5, .8), (.8, 1.2)]
    med = []
    for a, b in bins:
        m = (d >= a) & (d < b)
        med.append(f"{a:.2f}-{b:.2f}: 중앙 {np.median(T[m]):.2f}s (미정착 {cens[m].mean():.0%}, n={m.sum()})" if m.sum() else "-")
    print(f"\n[{c}] 경계 위 긋기 {len(d)}개 · 거리↔정착시간 Spearman ρ={rho:+.2f} (p={p:.1e})")
    for x in med: print("   ", x)
