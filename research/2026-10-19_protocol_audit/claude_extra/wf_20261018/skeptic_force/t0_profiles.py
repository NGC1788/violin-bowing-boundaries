"""Median force change vs bow travel (moving blocks only), by force-rank group, speed and beta third; plus the
post-stop change. Usage: python -I t0_profiles.py"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np

XG = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.195])
for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}: median dF (N) at travel (cm) {np.round(XG*100,1)} | post-stop dF")
    for s in ("v1", "v2", "v3"):
        for lo, hi in ((1, 5), (6, 15), (16, 25), (26, 35), (36, 45), (46, 50)):
            for blo, bhi in ((0.0, 0.045), (0.045, 0.1), (0.1, 0.3)):
                m = np.flatnonzero(S.valid & (S.folder == s) & (S.rank >= lo) & (S.rank <= hi) & (S.beta >= blo) & (S.beta < bhi))
                vals = []
                for g in XG:
                    d = []
                    for i in m:
                        j = S.at_travel(i, g)
                        if j >= 0:
                            d.append(S.df[i, j])
                    vals.append(med(d))
                post = [np.nanmedian(S.df[i, S.last_move[i] + 2:S.nvalid[i]]) for i in m if S.nvalid[i] - S.last_move[i] > 3]
                fon = med(S.f_on[m])
                print(f"  {s} r{lo:2d}-{hi:2d} b[{blo:.3f},{bhi:.3f}) Fon {fon:5.2f}: " + " ".join(f"{v:+.3f}" for v in vals) + f" | {med(post):+.3f}")
