"""Within one speed, how does the beta exponent of F_min evolve over the whole plateau?"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import analyse as A
import sys
sys.argv = ["x", "windows_full.csv.gz"]
rows = A.load()

def fit_beta(data):
    lb = np.log([d[1] for d in data]); lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])
    def nll(x):
        mu = x[0] + x[1] * lb; s = np.exp(x[2])
        up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
        dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))
    return minimize(nll, [0.0, -1.5, np.log(0.4)], method="Nelder-Mead", options={"maxiter": 20000}).x

rng = np.random.default_rng(1)
for c, tmax in (("v1", 3.5), ("v2", 1.5), ("v3", 0.5)):
    sub = [r for r in rows if r["condition"] == c]
    out = []
    for t in np.round(np.arange(0, tmax + 1e-9, 0.25 if c == "v1" else 0.1 if c == "v2" else 0.05), 2):
        data = A.intervals_at(sub, t)
        if len(data) < 15: continue
        x = fit_beta(data)
        boots = [fit_beta([data[i] for i in rng.integers(0, len(data), len(data))])[1] for _ in range(60)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cens = sum(not np.isfinite(d[2]) or not np.isfinite(d[3]) for d in data)
        out.append(f"{t:.2f}s β{x[1]:+.2f}[{lo:+.2f},{hi:+.2f}] 검열{cens}/{len(data)}")
    print(f"\n[{c} = {A.NOMINAL[c]} m/s]"); print("\n".join("  " + o for o in out))
