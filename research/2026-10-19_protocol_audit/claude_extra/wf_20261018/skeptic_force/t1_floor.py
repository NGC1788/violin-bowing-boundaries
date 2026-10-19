"""T1 floor test. If the travel-dependent decrease were an additive sensor term (true force unchanged), the reading
would keep falling at the same rate per cm whatever its level. If it is a real force decrease, the reading must stop
falling when the hair leaves the string (true force 0 -> reading = sensor zero). Compare the slope dF/dx in a late
travel segment between strokes whose reading at the segment start is near zero and strokes, in the same session
and beta range, whose reading is well above zero.  Usage: python -I t1_floor.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np

SEGS = ((0.06, 0.12), (0.10, 0.16), (0.12, 0.18))
for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    for s in ("v1", "v2", "v3"):
        for (x0, x1) in SEGS:
            rows = []
            for i in np.flatnonzero(S.valid & (S.folder == s) & (S.rank <= 35)):
                j0, j1 = S.at_travel(i, x0), S.at_travel(i, x1)
                if j0 < 0 or j1 < 0 or j1 - j0 < 3:
                    continue
                # level at segment start: median of 3 blocks around j0
                lvl = np.nanmedian(S.force_dc[i, max(j0 - 1, 0):j0 + 2])
                end = np.nanmedian(S.force_dc[i, max(j1 - 1, 0):j1 + 2])
                slope = (end - lvl) / (x1 - x0) / 100  # N per cm
                rows.append((lvl, slope, S.f_on[i], S.beta[i]))
            R = np.array(rows)
            out = []
            for lo, hi in ((-1, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.12), (0.12, 0.25), (0.25, 0.6), (0.6, 2)):
                m = (R[:, 0] >= lo) & (R[:, 0] < hi)
                if m.sum() >= 15:
                    out.append(f"[{lo:+.2f},{hi:.2f}) n{m.sum():4d} {med(R[m,1])*1000:+5.1f}")
            print(f"  {s} seg {x0*100:.0f}-{x1*100:.0f} cm  slope mN/cm by reading level at segment start: " + " | ".join(out))
