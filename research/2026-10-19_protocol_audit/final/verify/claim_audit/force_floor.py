import sys
sys.argv = ["x"]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import numpy as np
import compare as C
from data import load
for tag in ("A1", "A2", "B1", "C1", "D1", "B_T2r1", "B_T2r2"):
    d = load(tag); rows = C.intervals(d, 0.2)
    fmin = np.exp([r["fmin"] for r in rows]); fmax = np.exp([r["fmax"] for r in rows])
    left = [np.exp(r["fmin"]) for r in rows if not np.isfinite(r["lower"][0])]
    print(f"{tag:7s} per-column min force at t=0.2: median {np.median(fmin):.3f} N [p10 {np.percentile(fmin,10):.3f}, p90 {np.percentile(fmin,90):.3f}] | max force median {np.median(fmax):.2f} | left-censored cols {len(left)} with min force median {np.median(left) if left else float('nan'):.3f} N")
