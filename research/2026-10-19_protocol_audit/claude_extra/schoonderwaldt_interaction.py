"""Does the speed exponent of the (maintenance-type) minimum bow force change with beta in Schoonderwaldt 2008?
The elasto-plastic simulator predicts it rises from ~0 at small beta to ~+1 at large beta (Claude, 2026-10-18).
Model (interval-censored, Gaussian): log F = c + a log v + b log beta + g log v * log(beta / beta_ref)."""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location("sf", __file__.replace("schoonderwaldt_interaction.py", "schoonderwaldt_fmin.py"))
src = open(spec.origin).read().split("rng = np.random.default_rng(0)")[0]   # reuse data loading + intervals(), skip its main loop
ns = {}
exec(src, ns)
intervals = ns["intervals"]

def fit(data, beta_ref, interaction=True):
    lv = np.log([d[0] for d in data]); lb = np.log([d[1] for d in data])
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])
    lr = np.log(beta_ref)
    def nll(x):
        g = x[3] if interaction else 0.0
        mu = x[0] + x[1] * lv + x[2] * lb + g * lv * (lb - lr); s = np.exp(x[-1])
        up = norm.cdf((hi - mu) / s); dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))
    x0 = [0.0, 0.3, -1.5, 0.0, np.log(0.3)] if interaction else [0.0, 0.3, -1.5, np.log(0.3)]
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x, r.fun

rng = np.random.default_rng(1)
for gray, run in ((False, 2), (True, 2), (False, 1), (None, 2)):
    data, cens, none = intervals("D4", gray, run)
    betas = sorted({d[1] for d in data}); ref = float(np.exp(np.mean(np.log(betas))))
    x, f = fit(data, ref); x0, f0 = fit(data, ref, interaction=False)
    boots = [fit([data[i] for i in rng.integers(0, len(data), len(data))], ref)[0][3] for _ in range(300)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    small = [d for d in data if d[1] <= ref]; large = [d for d in data if d[1] > ref]
    xs, _ = fit(small, ref, interaction=False); xl, _ = fit(large, ref, interaction=False)
    tag = {False: "회색H=비H", True: "회색H=H", None: "회색H 제외"}[gray]
    print(f"D4 [{tag}, 연속{run}칸] 열 {len(data)} β {betas[0]:.3f}~{betas[-1]:.3f}: 상호작용 g = {x[3]:+.2f} [{lo:+.2f}, {hi:+.2f}] "
          f"(ΔlogL {f0 - f:.1f}); 기준 β {ref:.3f}에서 속도지수 {x[1]:+.2f}; 작은 β 절반 {xs[1]:+.2f} / 큰 β 절반 {xl[1]:+.2f}")
