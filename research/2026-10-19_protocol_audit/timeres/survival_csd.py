"""Steady-state boundary and bifurcation type from lock-in times (mixture-cure survival model).

Per speed folder, every stroke contributes its lock-in time L (first window from which
>= 80 % of the remaining windows are Helmholtz), interval-censored on the 0.05 s window
grid, right-censored if it never locks on the plateau, left-censored if locked from the start.

Model (one speed):
  boundary      log F*(beta) = c + p log beta                        (steady-state lower boundary)
  susceptible   P(stroke can lock) = Phi((log F - log F*) / s_b)      (soft boundary; else never locks)
  lock time     log L ~ Normal(mu, sigma),  mu = log tau0 - theta * log(max(d, eps)),  d = log F - log F*
Compared variants: theta free; theta = 1/2 (saddle-node); theta = 1 (Hopf / period doubling);
'no divergence' mu = log tau0 + k d (finite time even at the boundary).
F* comes from the whole time course, not from one window, and its profile likelihood shows
whether it is identifiable at all.

Usage: python survival_csd.py <archive-stem> [window_s]
"""

import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import analysis_multi as M

EPS = 0.02


def lock_intervals(rows, info, folder, window_s, rule="base"):
    min_per, _ = M.RULES[rule]
    f = info["folders"][folder]
    strokes = defaultdict(list)
    for r in rows:
        if r["folder"] == folder and r["window_s"] == window_s and r["force"] > 0:
            strokes[r["trial"]].append(r)
    out = []
    for seq in strokes.values():
        seq.sort(key=lambda r: r["t"])
        h = np.array([M.label(r, info["floor"], f["ref_f0"], min_per) == "helmholtz" for r in seq])
        t = np.array([r["t"] for r in seq])
        if len(t) < 3:
            continue
        i = next((k for k in range(len(h)) if h[k] and h[k:].mean() >= 0.8), None)
        if i is None:
            lo, hi = t[-1], np.inf                       # never locked within the plateau
        elif i == 0:
            lo, hi = 0.0, 0.0                            # locked from the first window
        else:
            lo, hi = t[i - 1], t[i]
        out.append((np.log(np.mean([r["force"] for r in seq])), np.log(np.median([r["beta"] for r in seq])), lo, hi))
    return out


def nll(x, data, variant):
    c, p, log_sb, log_tau0, a, log_sig = x
    lf, lb, lo, hi = data
    d = lf - (c + p * lb)
    sus = norm.cdf(d / np.exp(log_sb))
    if variant == "free":
        mu = log_tau0 - a * np.log(np.maximum(d, EPS))
    elif variant == "half":
        mu = log_tau0 - 0.5 * np.log(np.maximum(d, EPS))
    elif variant == "one":
        mu = log_tau0 - 1.0 * np.log(np.maximum(d, EPS))
    else:  # no divergence
        mu = log_tau0 + a * d
    sig = np.exp(log_sig)
    # P(L <= x) for a susceptible stroke; L = 0 handled as P(L <= first window), i.e. tiny time
    def cdf(x):
        return np.where(x > 0, norm.cdf((np.log(np.maximum(x, 1e-6)) - mu) / sig), 0.0)
    locked0 = (hi == 0)
    p_int = np.where(np.isinf(hi), 0.0, sus * (cdf(np.where(locked0, 0.025, hi)) - cdf(lo)))
    p_right = (1 - sus) + sus * (1 - cdf(lo))
    lik = np.where(np.isinf(hi), p_right, p_int)
    return -np.sum(np.log(np.clip(lik, 1e-300, None)))


def fit(data, variant, starts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        x0 = [-5 + rng.normal(0, .5), -2 + rng.normal(0, .3), np.log(0.2), np.log(0.3), 0.7 if variant != "none" else -1.0, np.log(0.8)]
        r = minimize(nll, x0, args=(data, variant), method="Nelder-Mead", options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-6, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    return best


def profile_p(data, variant, grid):
    """Profile likelihood of the steady-state beta exponent p (identifiability check)."""
    out = []
    for p in grid:
        def nll_fixed(y):
            return nll([y[0], p, *y[1:]], data, variant)
        r = min((minimize(nll_fixed, [-5 + s, np.log(0.2), np.log(0.3), 0.7, np.log(0.8)], method="Nelder-Mead",
                          options={"maxiter": 20000, "maxfev": 20000}) for s in (-1, 0, 1)), key=lambda r: r.fun)
        out.append(r.fun)
    return np.array(out)


def main():
    stem = sys.argv[1]
    window_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    rows = M.load(stem)
    info = M.prepare(rows)
    for folder in sorted(info["folders"]):
        raw = lock_intervals(rows, info, folder, window_s)
        data = tuple(np.array([d[i] for d in raw]) for i in range(4))
        n_right = int(np.isinf(data[3]).sum()); n_zero = int((data[3] == 0).sum())
        print(f"\n[{folder} · {info['folders'][folder]['speed']:.3f} m/s] 긋기 {len(raw)} (미잠금 {n_right}, 처음부터 잠김 {n_zero})")
        fits = {v: fit(data, v) for v in ("free", "half", "one", "none")}
        for v, r in fits.items():
            k = 6 if v in ("free", "none") else 5
            extra = f" θ={r.x[4]:.2f}" if v == "free" else (f" k={r.x[4]:+.2f}" if v == "none" else "")
            print(f"  {v:5} -logL {r.fun:9.1f}  AIC {2 * r.fun + 2 * k:9.1f}  정상상태 β지수 p={r.x[1]:+.2f}  τ0={np.exp(r.x[3]):.2f}s{extra}")
        best = min(fits, key=lambda v: 2 * fits[v].fun + 2 * (6 if v in ("free", "none") else 5))
        grid = np.linspace(-3.0, -0.5, 11)
        prof = profile_p(data, best, grid)
        inside = grid[prof - prof.min() <= 1.92]
        print(f"  최선 모형 {best}: p 프로파일 95% 구간 ≈ [{inside.min():+.2f}, {inside.max():+.2f}]"
              f"{' (격자 끝에 닿음 → 식별 약함)' if inside.min() <= grid[0] or inside.max() >= grid[-1] else ''}")


if __name__ == "__main__":
    main()
