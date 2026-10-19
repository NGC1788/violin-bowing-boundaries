"""Speed-specific boundary levels (separate intercept per speed, common beta slope), A1 vs A2 (Claude, 2026-10-18)."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import verify_astra_boundaries as V

def fit_levels(D):
    lv, lb, lo, hi = D.T
    levels = np.unique(lv)
    idx = np.searchsorted(levels, lv)
    def nll(x):
        mu = x[idx] + x[3] * lb
        s = np.exp(x[4])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))
    r = minimize(nll, [-2, -2, -2, -1, np.log(0.5)], method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x

for tag, stem in V.SPECIMENS.items():
    rows, beta, speed = V.load(stem)
    col = V.columns(beta)
    for t in (0.0, 0.3):
        D = V.intervals(rows, beta, col, speed, t)
        x = fit_levels(D)
        F = np.exp(x[:3] + x[3] * np.log(0.1))
        print(f"{tag} t={t}: β=0.1에서 경계 힘 v1 {F[0]:.3f} / v2 {F[1]:.3f} / v3 {F[2]:.3f} N  (β지수 {x[3]:+.2f}; v1→v2 지수 {np.log(F[1]/F[0])/np.log(2):+.2f}, v2→v3 {np.log(F[2]/F[1])/np.log(2):+.2f})")
