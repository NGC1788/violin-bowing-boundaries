"""Empirical non-overlapping (lag 4 windows = 0.2 s) transition rates vs normalised force (F / column F_E,cur)
for A2 and A1 at 0.05 m/s, floor strokes of the used columns.  Hysteresis (establishment threshold F_E above the
maintenance threshold) predicts p(nonH->H) ~ 0 below F/F_E = 1 while p(H->nonH) stays ~ 0 down to F_M."""
import numpy as np
from common import load, column_ratios, entry_low

LAG = 4
for tag in ("A2", "A1"):
    S, speed = load(tag)
    fl = [s for s in S if s.folder == 0 and s.floor]
    used = {c for c, *_ in column_ratios(fl)[0]}
    FE = {c: np.quantile([entry_low(s.F, s.H)[0] for s in fl if s.col == c and entry_low(s.F, s.H) is not None], .1) for c in used}
    rows = []
    for s in fl:
        if s.col not in used:
            continue
        for j in range(0, len(s.t) - LAG, LAG):
            rows.append((s.F[j + LAG] / FE[s.col], s.t[j + LAG], s.H[j], s.H[j + LAG]))
    R = np.array(rows, float)
    print(f"{tag} 0.05 m/s: lag {LAG} windows (0.2 s, non-overlapping); force bins of F(t+0.2)/F_E,cur")
    for tlo, thi in ((0.0, 0.8), (0.8, 2.5)):
        print(f"  t+0.2 in [{tlo},{thi}):")
        for lo, hi in ((0, 0.35), (0.35, 0.65), (0.65, 1.0), (1.0, 1.5), (1.5, 3), (3, 1e9)):
            m = (R[:, 0] >= lo) & (R[:, 0] < hi) & (R[:, 1] >= tlo) & (R[:, 1] < thi)
            n0 = np.sum(m & (R[:, 2] == 0)); n1 = np.sum(m & (R[:, 2] == 1))
            up = np.sum(m & (R[:, 2] == 0) & (R[:, 3] == 1)); dn = np.sum(m & (R[:, 2] == 1) & (R[:, 3] == 0))
            print(f"     F/F_E [{lo:.2f},{hi if hi < 1e8 else 'inf'}): from nonH n={n0:4d} up {up:3d} ({up / max(n0, 1):.3f}) | from H n={n1:4d} down {dn:3d} ({dn / max(n1, 1):.3f})")
