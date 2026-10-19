"""Thermal Schelleng maximum-bow-force law (Claude, 2026-10-19; post hoc model, pre-registered predictions).

Schelleng (1973): the Helmholtz slip can be triggered by the returning corner only while
    F_max = 2 Z v / (beta * dmu),   dmu = mu_s - mu_d.
Thermal closure (IWK friction reconstruction, one string = B T2, novelty/friction_contrast/offline):
    mu_d = A_MU * (F v)^(-A_EXP),  A_MU 0.390, A_EXP 0.181 (R2 0.95, LOO 0.042 log), mu_s = const.
So F_max solves the implicit equation
    F * (mu_s - A_MU (F v)^(-A_EXP)) = k * 2 Z v / beta^m        (k = 1, m = 1: no free parameter)
Local effective exponents at the solution (s = A_EXP mu_d / dmu, the contrast elasticity):
    d ln F / d ln v    = (1 - s) / (1 + s)
    d ln F / d ln beta = -m / (1 + s)       -> e_beta = -(1 + e_v) / 2 when m = 1
    d ln F / d ln Z    = 1 / (1 + s)
The root is unique on F v > (A_MU / mu_s)^(1/A_EXP) (dmu > 0), because d(F dmu)/dF = mu_s - (1 - A_EXP) mu_d > 0 there.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

A_MU, A_EXP = 0.390, 0.181          # mu_d = A_MU (F v)^-A_EXP   (friction_offline.log [A], structured law)
MU_PEAK = 0.839                     # median steady-window peak (stick-phase maximum) friction, B T2
MU_BREAK = 1.5                      # attack breakaway friction (task statement; see README for provenance check)
DMU_STEADY = 0.301                  # median measured steady contrast (31 Helmholtz windows)


def solve_F(G, v, mu_s, a_mu=A_MU, a_exp=A_EXP, iters=80):
    """Vectorised bisection in log F for F (mu_s - a_mu (F v)^-a_exp) = G (G > 0)."""
    G, v, mu_s = np.broadcast_arrays(np.asarray(G, float), np.asarray(v, float), np.asarray(mu_s, float))
    lf0 = np.log((a_mu / mu_s) ** (1.0 / a_exp) / v)           # dmu = 0 here
    lo = lf0 + 1e-12
    hi = np.maximum(lf0 + np.log(1e6), np.log(100.0 * G / mu_s))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        F = np.exp(mid)
        f = F * (mu_s - a_mu * (F * v) ** (-a_exp)) - G
        pos = f > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)                                       # log F


def contrast_elasticity(lF, v, mu_s, a_mu=A_MU, a_exp=A_EXP):
    mud = a_mu * (np.exp(lF) * v) ** (-a_exp)
    return a_exp * mud / (mu_s - mud), mud


def local_exponents(lF, v, mu_s, m=1.0):
    s, mud = contrast_elasticity(lF, v, mu_s)
    return dict(e_v=(1 - s) / (1 + s), e_beta=-m / (1 + s), e_Z=1 / (1 + s), s=s, mu_d=mud, dmu=mu_s - mud)


# ---------------------------------------------------------------- model family (all use the published Z)
# Each model: (n_free, init, predict(theta, lv, lb, lz) -> log F_max)

def _th(mu_s, lk, lv, lb, lz, m=1.0):
    v = np.exp(lv)
    G = np.exp(lk + np.log(2.0) + lz + lv - m * lb)
    return solve_F(G, v, mu_s)


MODELS = {
    # plain Schelleng, constant contrast; one global constant c (= 2 / dmu_eff)
    "S1_strict": (1, [1.0], lambda p, lv, lb, lz: p[0] + lz + lv - lb),
    # plain Schelleng with the measured steady contrast 0.301, no free parameter
    "S0_dmu_steady": (0, [], lambda p, lv, lb, lz: np.log(2.0 / DMU_STEADY) + lz + lv - lb),
    # thermal, zero parameters
    "T0_peak": (0, [], lambda p, lv, lb, lz: _th(MU_PEAK, 0.0, lv, lb, lz)),
    "T0_break": (0, [], lambda p, lv, lb, lz: _th(MU_BREAK, 0.0, lv, lb, lz)),
    # thermal, one parameter: global prefactor k with mu_s fixed
    "T1k_peak": (1, [0.0], lambda p, lv, lb, lz: _th(MU_PEAK, p[0], lv, lb, lz)),
    "T1k_break": (1, [0.0], lambda p, lv, lb, lz: _th(MU_BREAK, p[0], lv, lb, lz)),
    # thermal, one parameter: effective static coefficient mu_s (k = 1); mu_s = MU_MIN + exp(p)
    "T1mu": (1, [np.log(0.6)], lambda p, lv, lb, lz: _th(0.30 + np.exp(p[0]), 0.0, lv, lb, lz)),
    # thermal, two parameters (mu_s and k)
    "T2_mu_k": (2, [np.log(0.6), 0.0], lambda p, lv, lb, lz: _th(0.30 + np.exp(p[0]), p[1], lv, lb, lz)),
    # empirical reference: free speed and beta exponents with Z^1 (3 parameters)
    "E3_free_Z": (3, [0.0, 0.66, -1.1], lambda p, lv, lb, lz: p[0] + lz + p[1] * lv + p[2] * lb),
}


def mu_s_of(name, theta):
    if name in ("T0_peak", "T1k_peak"):
        return MU_PEAK
    if name in ("T0_break", "T1k_break"):
        return MU_BREAK
    if name in ("T1mu", "T2_mu_k"):
        return 0.30 + float(np.exp(theta[0]))
    return None


def arrays(rows, zof):
    lv = np.array([r["lv"] for r in rows]); lb = np.array([r["lb"] for r in rows])
    lz = np.array([np.log(zof[r["tag"]]) for r in rows])
    lo = np.array([r["upper"][0] for r in rows]); hi = np.array([r["upper"][1] for r in rows])
    return lv, lb, lz, lo, hi


def censored_nll(mu, s, lo, hi):
    pr = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
    return -np.sum(np.log(np.clip(pr, 1e-300, None)))


def fit(name, rows, zof):
    """Interval-censored normal fit of log F_max (upper boundary columns); returns (theta, log sigma, nll)."""
    n, init, f = MODELS[name]
    lv, lb, lz, lo, hi = arrays(rows, zof)
    def nll(p):
        return censored_nll(f(p[:n], lv, lb, lz), np.exp(p[n]), lo, hi)
    best = None
    starts = [np.r_[init, np.log(0.3)]]
    if n >= 1:
        starts.append(np.r_[np.array(init) + 0.3, np.log(0.3)])
        starts.append(np.r_[np.array(init) - 0.3, np.log(0.3)])
    for x0 in starts:
        r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:n], float(best.x[n]), float(best.fun)


def predict(name, theta, lv, lb, lz):
    return MODELS[name][2](np.asarray(theta, float), np.asarray(lv, float), np.asarray(lb, float), np.asarray(lz, float))


def offset_fit(mu0, lo, hi):
    """One-offset interval-censored fit with the model fixed (= final/confirm_A_T2.offset_fit)."""
    ok = ~(np.isnan(lo) & np.isnan(hi))
    mu0, lo, hi = mu0[ok], lo[ok], hi[ok]
    r = minimize(lambda p: censored_nll(mu0 + p[0], np.exp(p[1]), lo, hi), [0.0, np.log(0.5)], method="Nelder-Mead")
    return float(r.x[0]), float(np.exp(r.x[1]))


def design_exponents(name, theta, lv, lb, lz):
    """Effective exponents of a model over a design (OLS of predicted log F on [1, lv, lb])."""
    y = predict(name, theta, lv, lb, lz)
    X = np.column_stack([np.ones_like(lv), lv, lb])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(c=float(c[0]), e_v=float(c[1]), e_beta=float(c[2]))
