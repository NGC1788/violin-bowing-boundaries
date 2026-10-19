"""T4 (angle 1): what does the size of the travel-dependent change depend on?
Per speed session and beta column (40 columns = 40 robot positions along the string), regress the force change at a
fixed travel on the onset force over strokes not limited by the zero floor (0.25 <= F_on <= 2.5 N):
      dF(x) = a_c(x) + b_c(x) * F_on        (a: additive part, b: proportional part)
Then ask which description of a_c across the 40 columns is better:
  M0 constant;  M1 linear in robot position along the string (beta*L) = smooth pose term;
  M2 series stiffness  a = -delta * k_eff(beta),  1/k_eff = 1/k_h + L beta(1-beta)/T   (T = 145 N, L = 0.70 m; A-T1)
  M3 = M1 with a quadratic term (flexible pose term, 3 params)
Compared by residual SS and AIC across the 40 columns.  Also b_c(x) by travel (is the proportional part linear in x,
as a two-load-cell gain imbalance would give, or a dip?).  Usage: python -I t4_beta.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np
from scipy.optimize import least_squares

T, L = 145.31, 0.70
XS = (0.05, 0.10, 0.15, 0.19)


def keff(beta, kh):
    return 1.0 / (1.0 / kh + L * beta * (1 - beta) / T)


def aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    for s in ("v1", "v2", "v3"):
        for xg in XS:
            a_c, b_c, bet = [], [], []
            for c in range(40):
                m = np.flatnonzero(S.valid & (S.folder == s) & (S.col == c) & (S.f_on >= 0.25) & (S.f_on <= 2.5))
                y, f = [], []
                for i in m:
                    j = S.at_travel(i, xg)
                    if j >= 0:
                        y.append(S.df[i, j]); f.append(S.f_on[i])
                if len(y) >= 8:
                    p = np.polyfit(f, y, 1)
                    b_c.append(p[0]); a_c.append(p[1]); bet.append(med(S.beta[m]))
            a_c, b_c, bet = map(np.array, (a_c, b_c, bet))
            n = a_c.size
            rss0 = np.sum((a_c - a_c.mean()) ** 2)
            p1 = np.polyfit(bet, a_c, 1); rss1 = np.sum((a_c - np.polyval(p1, bet)) ** 2)
            p3 = np.polyfit(bet, a_c, 2); rss3 = np.sum((a_c - np.polyval(p3, bet)) ** 2)
            best = None
            for kh0 in (100, 300, 700, 1500, 4000, 2e4):
                r = least_squares(lambda q: -q[0] * keff(bet, abs(q[1])) - a_c, x0=[np.mean(-a_c) / keff(0.1, kh0), kh0])
                if best is None or r.cost < best.cost:
                    best = r
            rss2 = 2 * best.cost
            kh = abs(best.x[1]); delta = best.x[0]
            ratio = (np.mean(a_c[bet < 0.045]) / np.mean(a_c[bet > 0.1])) if np.any(bet > 0.1) else np.nan
            print(f"  {s} x={xg*100:4.1f} cm: a_c median {1000*med(a_c):+.0f} mN (small/large beta ratio {ratio:.2f}); b_c median {1000*med(b_c):+.0f} mN/N | "
                  f"AIC M0 {aic(rss0,n,1):.1f} M1(lin pose) {aic(rss1,n,2):.1f} M2(stiffness) {aic(rss2,n,2):.1f} [k_h {kh:.0f} N/m, delta {1000*delta:.2f} mm] M3(quad pose) {aic(rss3,n,3):.1f}")
