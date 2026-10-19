"""code_audit: plan-specified few-shot for D-logit (refit intercept + log F coefficient) vs the band-shift used in compare.py."""
import sys, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.special import expit
T = 0.2
dat = {t: load(t) for t in TRAIN + TEST}
rows = {t: C.intervals(dat[t], T) for t in TRAIN + TEST}
lf_ = lambda d: np.log(np.maximum(d["force"], 1e-3))
X = np.vstack([C.features(lf_(d), np.log(d["speed"]), np.log(d["beta"]), d["t"]) for d in (dat["A1"], dat["A2"])])
y = np.concatenate([dat["A1"]["H"], dat["A2"]["H"]]).astype(int)
mu, sd = X.mean(0), X.std(0)
lr = LogisticRegression(C=10.0, max_iter=5000).fit((X - mu) / sd, y)
def eta(lf, lv, lb, t):
    return lr.decision_function((C.features(lf, lv, lb, t) - mu) / sd)
for t in TEST:
    d = dat[t]; R = rows[t]
    all_idx = np.concatenate([r["idx"] for r in R])
    lv_of = {}; lb_of = {}
    for r in R:
        for i in r["idx"]:
            lv_of[i] = r["lv"]; lb_of[i] = r["lb"]
    base = C.score_boundaries(R, [C.band_from_prob(lambda a, b, c, e: expit(eta(a, b, c, e)), r, T) for r in R])
    res = {}
    for kshot in (10, 40):
        rng = np.random.default_rng(123)
        sc = []
        for rep in range(20):
            ch = rng.choice(all_idx, kshot, replace=False)
            lf = lf_(d)[ch]; e0 = eta(lf, np.array([lv_of[i] for i in ch]), np.array([lb_of[i] for i in ch]), np.full(kshot, T))
            zl = (lf - mu[0]) / sd[0]; yy = d["H"][ch].astype(float)
            def nll(x):
                p = np.clip(expit(e0 + x[0] + x[1] * zl), 1e-9, 1 - 1e-9)
                return -np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)) + 0.5 * (x[0] ** 2 + x[1] ** 2)
            xa = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
            pfun = lambda a, b, c, e, xa=xa: expit(eta(a, b, c, e) + xa[0] + xa[1] * (a - mu[0]) / sd[0])
            s2 = C.score_boundaries(R, [C.band_from_prob(pfun, r, T) for r in R])
            sc.append((s2["lower"][0], s2["upper"][0]))
        res[kshot] = np.nanmedian(np.array(sc), axis=0)
    print(f"{t:7s} D-logit zero-shot L {base['lower'][0]:.3f} U {base['upper'][0]:.3f} | plan few-shot k10 L {res[10][0]:.3f} U {res[10][1]:.3f} | k40 L {res[40][0]:.3f} U {res[40][1]:.3f}")
