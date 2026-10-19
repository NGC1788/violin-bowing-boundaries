"""T5: the bridge channel's DC at low force goes negative during the stroke (physically impossible as a mean lateral
force while bowing in one direction). Is that offset travel-dependent like the force decrease (a shared, pose-linked
measurement artifact would make the skeptic's case), time-dependent, or tied to the string state?
For ranks 1-20 (low force) and 41-50: bridge DC vs travel and vs elapsed time for the three speeds (spread between
speeds, like drift_tests T1b), and per-stroke correlation between the bridge DC change and the force change at 15 cm.
Usage: python -I t5_bridge.py"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np

XG = np.array([0.02, 0.05, 0.08, 0.11, 0.14, 0.17])
TG = np.array([0.3, 0.6, 1.0, 1.5, 2.0])
for A in ("A2", "A1"):
    S = Spec(A)
    t = (np.arange(S.nb) + 0.5) * S.dt
    print(f"\n=== {A}")
    for lo, hi in ((1, 10), (11, 20), (41, 50)):
        for key, grid in (("travel", XG), ("time", TG)):
            rows = {}
            for s in ("v1", "v2", "v3"):
                m = np.flatnonzero(S.valid & (S.folder == s) & (S.rank >= lo) & (S.rank <= hi))
                vals = []
                for g in grid:
                    d = []
                    for i in m:
                        x = S.x[i] if key == "travel" else t - t[S.onset[i]]
                        j = np.flatnonzero((x >= g) & S.moving[i])
                        if j.size:
                            d.append(S.bridge_dc[i, j[0]])
                    vals.append(med(d))
                rows[s] = vals
            spread = np.nanmean([np.nanstd([rows[s][k] for s in rows]) for k in range(len(grid))])
            print(f"  ranks {lo}-{hi} bridge DC (N) by {key:6s} {grid}: " + " | ".join(f"{s}: " + " ".join(f"{v:+.3f}" for v in rows[s]) for s in rows)
                  + f"  -> spread {spread:.4f}")
    # per-stroke association, ranks 1-20, between bridge DC change (2 cm -> 15 cm) and force change (same points)
    for s in ("v1", "v2", "v3"):
        m = np.flatnonzero(S.valid & (S.folder == s) & (S.rank <= 20))
        db, dfc = [], []
        for i in m:
            j0, j1 = S.at_travel(i, 0.02), S.at_travel(i, 0.15)
            if j0 >= 0 and j1 >= 0:
                db.append(S.bridge_dc[i, j1] - S.bridge_dc[i, j0]); dfc.append(S.force_dc[i, j1] - S.force_dc[i, j0])
        db, dfc = np.array(db), np.array(dfc)
        print(f"  {s} ranks 1-20: bridge DC change 2->15 cm median {med(db):+.3f} N, force change {med(dfc):+.3f} N, per-stroke corr {np.corrcoef(db, dfc)[0,1]:+.2f}")
