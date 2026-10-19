"""claim_audit: censoring composition of the boundary targets per specimen, and per-specimen free power-law fits
(same censored_fit as compare.py) to see which beta/speed exponents each test specimen actually has."""
import sys
sys.argv = ["x"]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import numpy as np
import compare as C
from data import load, TRAIN, TEST

for T in (0.0, 0.2, 0.3):
    print(f"== t={T}")
    for tag in TRAIN + TEST:
        d = load(tag)
        rows = C.intervals(d, T)
        L = np.array([r["lower"] for r in rows]); U = np.array([r["upper"] for r in rows])
        lfin = np.sum(np.isfinite(L[:, 0]) & np.isfinite(L[:, 1])); lleft = np.sum(~np.isfinite(L[:, 0])); lright = np.sum(~np.isfinite(L[:, 1]))
        unone = np.sum(np.isnan(U[:, 0]) & np.isnan(U[:, 1]))
        ufin = np.sum(np.isfinite(U[:, 0]) & np.isfinite(U[:, 1])); uright = np.sum(np.isfinite(U[:, 0]) & ~np.isfinite(U[:, 1]))
        for r in rows: r["tag"] = tag
        pl = C.censored_fit(rows, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")
        ur = [r for r in rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
        pu = C.censored_fit(ur, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "upper")
        print(f"  {tag:7s} cols {len(rows)} | lower finite {lfin} left {lleft} right(no run) {lright} | upper finite {ufin} right {uright} none {unone}"
              f" | own fit lower v {pl[1]:+.2f} b {pl[2]:+.2f} s {np.exp(pl[3]):.2f} | upper v {pu[1]:+.2f} b {pu[2]:+.2f}")
