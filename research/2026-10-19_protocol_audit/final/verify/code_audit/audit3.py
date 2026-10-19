"""code_audit: why D-gbm gives lower-boundary errors > 2 in baselines.py (single-specimen training)."""
import sys, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load
T = 0.2
ALL = ("A1", "A2", "B1", "C1", "D1", "B_T2r1", "B_T2r2")
dat = {t: load(t) for t in ALL}
rows = {}
for t in ALL:
    rows[t] = C.intervals(dat[t], T)
GRID0 = np.log(0.03)
cases = [(("B_T2r1",), "B_T2r2"), (("B_T2r2",), "B_T2r1"), (("B_T2r2",), "B_T2r2"), (("B_T2r1",), "B_T2r1"), (("B1",), "B1"), (("D1",), "D1"), (("C1",), "C1"), (("A1","A2"), "B_T2r2")]
for tr, te in cases:
    dm = C.data_models([dat[t] for t in tr])
    fmin_train = min(np.min(dat[t]["force"][np.isclose(dat[t]["t"], T)]) for t in tr)
    for name in ("D-gbm", "D-logit"):
        preds = [C.band_from_prob(dm[name], r, T) for r in rows[te]]
        fin = [(r, p) for r, p in zip(rows[te], preds) if np.isfinite(r["lower"][0]) and np.isfinite(r["lower"][1])]
        at_floor = sum(1 for r, p in fin if np.isfinite(p[0]) and abs(p[0] - GRID0) < 1e-9)
        errs = np.array([abs(p[0] - 0.5 * sum(r["lower"])) if np.isfinite(p[0]) else 3.0 for r, p in fin])
        errs_nofloor = np.array([e for (r, p), e in zip(fin, errs) if not (np.isfinite(p[0]) and abs(p[0] - GRID0) < 1e-9)])
        # alternative: clamp prediction to the lowest force measured in that column (no extrapolation)
        errs_clamp = np.array([abs(max(p[0], r["fmin"]) - 0.5 * sum(r["lower"])) if np.isfinite(p[0]) else 3.0 for r, p in fin])
        print(f"train {'+'.join(tr):9s} -> {te:7s} {name:7s}: median lower err {np.median(errs):.3f}; finite cols {len(fin)}, predicted lower at grid floor 0.03 N: {at_floor}"
              f"; median excl. floor {np.median(errs_nofloor) if len(errs_nofloor) else np.nan:.3f}; clamp-to-column-min {np.median(errs_clamp):.3f}; min train force {fmin_train:.3f} N")
    if tr == ("B_T2r2",) and te == "B_T2r2":
        # P(H) of the gbm below the measured range for one floor column
        for r, p in fin:
            if abs(p[0] - GRID0) < 1e-9:
                g = np.log(np.array([0.03, 0.1, 0.2, np.exp(r["fmin"]), np.exp(r["lower"][0]), np.exp(r["lower"][1])]))
                pr = dm["D-gbm"](g, np.full(6, r["lv"]), np.full(6, r["lb"]), np.full(6, T))
                d = dat[te]; h = d["H"][r["idx"]]
                print(f"   example col v={np.exp(r['lv']):.2f} beta={np.exp(r['lb']):.3f}: column min force {np.exp(r['fmin']):.3f} N, obs lower interval ({np.exp(r['lower'][0]):.3f}, {np.exp(r['lower'][1]):.3f}) N, first labels {h[:6].astype(int)}")
                print(f"   gbm P(H) at F=0.03,0.10,0.20,colmin,lo,hi: {np.round(pr,2)}")
                break
