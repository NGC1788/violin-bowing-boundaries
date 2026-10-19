"""Re-analyse Schoonderwaldt et al. (2008) published violin-string grids for the minimum-bow-force law,
with interval censoring (the authors fitted borders by eye / without a censoring model).
Data: Codex's vector extraction of their Figures 4 and 7 (published author classifications)."""
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
SRC = "/Users/jeongjihoon/violin-bowing-boundaries/research/2026-10-13/schoonderwaldt2008/published_cells.csv"
rows = list(csv.DictReader(open(SRC)))

def intervals(string, gray_is_h, run):
    cols = {}
    for r in rows:
        if r["string"] != string or r["published_category"] == "unplotted_unknown":
            continue
        if r["published_category"] == "helmholtz_not_systematically_analyzed" and gray_is_h is None:
            continue
        h = r["published_category"] == "helmholtz" or (gray_is_h and r["published_category"] == "helmholtz_not_systematically_analyzed")
        cols.setdefault((float(r["bow_speed_nominal_m_s"]), float(r["beta_from_reported_rounded_distance"])), []).append((float(r["bow_force_nominal_N"]), h))
    out, cens, none = [], 0, 0
    for (v, b), cells in cols.items():
        cells.sort(); hs = [c[1] for c in cells]; lf = np.log([c[0] for c in cells])
        s = next((i for i in range(len(hs) - run + 1) if all(hs[i:i + run])), None)
        if s is None: none += 1; continue
        lo, hi = (-np.inf, lf[0]) if s == 0 else (lf[s - 1], lf[s])
        cens += s == 0
        out.append((v, b, lo, hi))
    return out, cens, none

def fit(data, fixed_a=None):
    lv = np.log([d[0] for d in data]); lb = np.log([d[1] for d in data])
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])
    def nll(x):
        a = fixed_a if fixed_a is not None else x[1]
        mu = x[0] + a * lv + x[-2] * lb; s = np.exp(x[-1])
        up = norm.cdf((hi - mu) / s); dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))
    x0 = [0.0, -1.5, np.log(0.3)] if fixed_a is not None else [0.0, 0.5, -1.5, np.log(0.3)]
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
    return r.x, r.fun

rng = np.random.default_rng(0)
for string in ("D4", "E5"):
    for gray, run in ((False, 2), (True, 2), (False, 1), (None, 2)):
        data, cens, none = intervals(string, gray, run)
        if len({d[0] for d in data}) < 2: continue
        x, f = fit(data); x0, f0 = fit(data, fixed_a=0.0); x1, f1 = fit(data, fixed_a=1.0)
        boots = []
        for _ in range(300):
            s = [data[i] for i in rng.integers(0, len(data), len(data))]
            boots.append(fit(s)[0][1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        tag = {False: "회색H=비H", True: "회색H=H", None: "회색H 제외"}[gray]
        print(f"{string} [{tag}, 연속{run}칸] 열 {len(data)} (왼쪽검열 {cens}, 경계없음 {none})  "
              f"속도지수 {x[1]:+.2f} [{lo:+.2f},{hi:+.2f}]  β지수 {x[2]:+.2f}  "
              f"ΔlogL(자유-속도0) {f0 - f:.1f}  ΔlogL(자유-속도1) {f1 - f:.1f}")
