"""Is the super-linear speed exponent uniform, or driven by the fastest speed only?
Fit beta exponent shared, one intercept per (string, speed); local exponent
between neighbouring speeds = intercept difference / ln 2."""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from fmin_censored import SOURCES, REPO, read, intervals
warnings.filterwarnings("ignore")

data = []
for s, rel in SOURCES.items():
    data += intervals(read(REPO / rel), s, 2)[0]
keys = sorted({(d[0], d[1]) for d in data})
idx = np.array([keys.index((d[0], d[1])) for d in data])
lb = np.log([d[2] for d in data]); lo = np.array([d[3] for d in data]); hi = np.array([d[4] for d in data])

def nll(t):
    mu = t[:len(keys)][idx] + t[-2] * lb
    s = np.exp(t[-1])
    up = norm.cdf((hi - mu) / s); dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0)
    return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

r = minimize(nll, np.concatenate([np.zeros(len(keys)), [-1.9, np.log(0.4)]]), method="L-BFGS-B")
c = dict(zip(keys, r.x[:len(keys)]))
print(f"공통 β 지수 {r.x[-2]:+.2f}")
print(f"{'표본':4}{'0.05→0.10 국소 지수':>22}{'0.10→0.20 국소 지수':>22}")
for s in SOURCES:
    e1 = (c[(s, 0.10)] - c[(s, 0.05)]) / np.log(2)
    e2 = (c[(s, 0.20)] - c[(s, 0.10)]) / np.log(2)
    print(f"{s:4}{e1:>22.2f}{e2:>22.2f}")
