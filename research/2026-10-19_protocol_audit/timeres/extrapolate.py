"""Steady-state beta exponent by extrapolating the settling of the boundary.

Per speed, every (beta column, window time t) gives a censored interval for log F_min.
Model:  log F_min(beta, t) = c + p_inf log beta + (d0 + d1 log beta) exp(-t / tau)
so the instantaneous beta exponent p(t) = p_inf + d1 exp(-t/tau) relaxes to p_inf.
Columns enter with whatever plateau length they have; no single common time is needed.
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
sys.argv = ["x", "windows_full.csv.gz"]
import analyse as A
rows = A.load()

def data_for(c, step):
    sub = [r for r in rows if r["condition"] == c]
    tmax = max(r["t"] for r in sub)
    out = []
    for t in np.round(np.arange(0, tmax + 1e-9, step), 3):
        for (v, b, lo, hi) in A.intervals_at(sub, t):
            out.append((b, t, lo, hi))
    return out

def fit(data, x0=None):
    lb = np.log([d[0] for d in data]); t = np.array([d[1] for d in data])
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])
    def nll(x):
        c, p, d0, d1, ltau, ls = x
        mu = c + p * lb + (d0 + d1 * lb) * np.exp(-t / np.exp(ltau))
        s = np.exp(ls)
        up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
        dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))
    best = None
    for tau0 in (0.2, 0.6, 1.5):
        r = minimize(nll, x0 if x0 is not None else [-5, -2, 1, 1, np.log(tau0), np.log(0.3)], method="Nelder-Mead",
                     options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun: best = r
        if x0 is not None: break
    return best.x, best.fun

rng = np.random.default_rng(2)
for c, step in (("v1", 0.25), ("v2", 0.1), ("v3", 0.05)):
    data = data_for(c, step)
    x, f = fit(data)
    betas = sorted({d[0] for d in data})
    boots = []
    for _ in range(40):
        pick = set(rng.choice(len(betas), len(betas), replace=True))
        keep = [betas[i] for i in pick]
        sample = [d for d in data if d[0] in keep]
        boots.append(fit(sample, x0=x)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"[{c} {A.NOMINAL[c]} m/s] 관측 {len(data)}개 · 정상상태 β 지수 p∞ = {x[1]:+.2f}  (열 부트스트랩 95% [{lo:+.2f}, {hi:+.2f}])"
          f" · 이완시간 τ = {np.exp(x[4]):.2f}s · t=0의 β 지수 {x[1]+x[3]:+.2f}")
