"""Step 1: the minimum-bow-force law with censoring handled, not dropped.

Every bow-position column of every (string, speed) panel contributes an
*interval* for its lower boundary instead of a point:

  - normal column:    last non-Helmholtz force below the first Helmholtz run
                      < F_min <= first force of that run
  - left-censored:    the lowest force on the grid is already Helmholtz,
                      so only F_min <= F_lowest is known
  - no Helmholtz run: the boundary is not located; the column is reported, not used

The earlier pilot simply discarded left-censored columns. Those columns sit at
large beta where F_min falls below the force grid, so discarding them removes
exactly the low-force end of the law and can bias the beta exponent.

Model: log F_min = c_string + a log v + p log beta + e,  e ~ N(0, s^2),
fitted by interval-censored maximum likelihood.
"""

import csv
import gzip
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path.home() / "violin-bowing-boundaries"
SOURCES = {
    "A": "reports/collection/2024-03-25_TypeA_sample1/regime_map/20260917T135154Z_3982fcc7/regimes.csv",
    "B": "reports/collection/2024-04-10_TypeB_sample1__2024-04-10/regime_map/20260917T154209Z_aa60a05c/regimes.csv",
    "C": "reports/collection/2024-04-12_TypeC_sample1__2024-04-15/regime_map/20260917T163822Z_54d11e71/regimes.csv",
    "D": "reports/collection/2024-04-15_TypeD_sample1__2024-04-17/regime_map/20260917T172527Z_6dbcf479/regimes.csv",
    "A2": "research/2026-10-14/lockbox_A2/regimes.csv.gz",
}
NOMINAL = {"1": 0.05, "2": 0.10, "3": 0.20}


def read(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def intervals(rows: list[dict], string: str, run: int):
    """One lower-boundary interval per (speed, beta column)."""
    out, missing = [], 0
    groups: dict = {}
    for r in rows:
        if str(r["window_inside_plateau"]) not in ("True", "true", "1"):
            continue
        force = float(r["c1_window_mean"])
        if force <= 0:
            continue
        key = (r["condition"], r["beta_level"])
        groups.setdefault(key, []).append((force, float(r["beta"]), float(r["c2_window_mean"]), r["label"]))
    for (condition, _), cells in groups.items():
        cells.sort()
        forces = np.array([c[0] for c in cells])
        h = np.array([c[3] == "helmholtz" for c in cells])
        beta = float(np.median([c[1] for c in cells]))
        speed = NOMINAL[condition[-1]]
        start = next((i for i in range(len(h) - run + 1) if h[i:i + run].all()), None)
        if start is None:
            missing += 1
            continue
        lo = np.log(forces[start - 1]) if start > 0 else -np.inf   # -inf: left-censored
        hi = np.log(forces[start])
        out.append((string, speed, beta, lo, hi))
    return out, missing


def negloglik(theta, s_idx, lv, lb, lo, hi, n_strings):
    c = theta[:n_strings]
    a, p, log_sigma = theta[n_strings:n_strings + 3]
    sigma = np.exp(log_sigma)
    mu = c[s_idx] + a * lv + p * lb
    upper = norm.cdf((hi - mu) / sigma)
    lower = np.where(np.isfinite(lo), norm.cdf((lo - mu) / sigma), 0.0)
    return -np.sum(np.log(np.clip(upper - lower, 1e-300, None)))


def fit(data, fixed=None):
    """fixed: dict like {'a': 1.0, 'p': -2.0} to test a published law."""
    strings = sorted({d[0] for d in data})
    s_idx = np.array([strings.index(d[0]) for d in data])
    lv = np.log([d[1] for d in data])
    lb = np.log([d[2] for d in data])
    lo = np.array([d[3] for d in data])
    hi = np.array([d[4] for d in data])
    k = len(strings)
    fixed = fixed or {}

    def unpack(free):
        theta, j = list(free[:k]), k
        for name in ("a", "p"):
            if name in fixed:
                theta.append(fixed[name])
            else:
                theta.append(free[j]); j += 1
        theta.append(free[j])
        return np.array(theta)

    n_free = k + sum(name not in fixed for name in ("a", "p")) + 1
    start = np.concatenate([np.full(k, np.median(hi)), [v for name, v in (("a", 1.0), ("p", -1.5)) if name not in fixed], [np.log(0.4)]])
    result = minimize(lambda f: negloglik(unpack(f), s_idx, lv, lb, lo, hi, k), start, method="BFGS")
    theta = unpack(result.x)
    se = {}
    if not fixed and result.success:
        cov = result.hess_inv
        se = {"a": float(np.sqrt(cov[k, k])), "p": float(np.sqrt(cov[k + 1, k + 1]))}
    return {"nll": float(result.fun), "params": n_free, "a": theta[k], "p": theta[k + 1],
            "sigma": float(np.exp(theta[k + 2])), "se": se, "ok": bool(result.success)}


LAWS = {
    "자유 거듭제곱": None,
    "Schelleng v/β²": {"a": 1.0, "p": -2.0},
    "Beigi v/β": {"a": 1.0, "p": -1.0},
    "Schoonderwaldt 속도무관": {"a": 0.0},
}


def report(data, title):
    n = len(data)
    left = sum(not np.isfinite(d[3]) for d in data)
    print(f"\n=== {title}: 열 {n}개 (왼쪽 검열 {left}개 = {left / n:.0%}) ===")
    print(f"{'법칙':22}{'속도 지수':>14}{'β 지수':>14}{'AIC':>10}{'속도 하나 빼고 -logL/열':>24}")
    for name, fixed in LAWS.items():
        res = fit(data, fixed)
        aic = 2 * res["nll"] + 2 * res["params"]
        cv = []
        for held in (0.05, 0.10, 0.20):
            train = [d for d in data if d[1] != held]
            test = [d for d in data if d[1] == held]
            r = fit(train, fixed)
            strings = sorted({d[0] for d in train})
            # evaluate the held speed with the trained parameters (string intercepts reused)
            refit = fit(train, fixed)
            theta_c = {}
            # recover intercepts by a constrained refit on train only
            k = len(strings)
            s_idx = np.array([strings.index(d[0]) for d in test])
            lv = np.log([d[1] for d in test]); lb = np.log([d[2] for d in test])
            lo = np.array([d[3] for d in test]); hi = np.array([d[4] for d in test])
            c = _intercepts(train, refit, strings)
            mu = c[s_idx] + refit["a"] * lv + refit["p"] * lb
            sig = refit["sigma"]
            upper = norm.cdf((hi - mu) / sig)
            lower = np.where(np.isfinite(lo), norm.cdf((lo - mu) / sig), 0.0)
            cv.append(-np.log(np.clip(upper - lower, 1e-300, None)))
        cv = float(np.mean(np.concatenate(cv)))
        a = f"{res['a']:+.3f}" + (f"±{res['se']['a']:.3f}" if res["se"] else " (고정)" if fixed and "a" in fixed else "")
        p = f"{res['p']:+.3f}" + (f"±{res['se']['p']:.3f}" if res["se"] else " (고정)" if fixed and "p" in fixed else "")
        print(f"{name:22}{a:>14}{p:>14}{aic:>10.1f}{cv:>24.3f}")


def _intercepts(train, refit, strings):
    """String intercepts given fixed (a, p, sigma): 1-D censored MLE per string."""
    out = np.zeros(len(strings))
    for i, s in enumerate(strings):
        sub = [d for d in train if d[0] == s]
        lv = np.log([d[1] for d in sub]); lb = np.log([d[2] for d in sub])
        lo = np.array([d[3] for d in sub]); hi = np.array([d[4] for d in sub])
        base = refit["a"] * lv + refit["p"] * lb

        def nll(c):
            mu = c[0] + base
            up = norm.cdf((hi - mu) / refit["sigma"])
            dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / refit["sigma"]), 0.0)
            return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

        out[i] = minimize(nll, [np.median(hi - base)], method="Nelder-Mead").x[0]
    return out


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    data, missing = [], {}
    for string, rel in SOURCES.items():
        got, miss = intervals(read(REPO / rel), string, run)
        data += got
        missing[string] = miss
    print(f"헬름홀츠 구간 정의: 연속 {run}칸 이상 · 경계를 못 찾은 열: {missing}")
    report([d for d in data if d[0] != "A2"], "A·B·C·D")
    report(data, "A·B·C·D + A2")
    # The earlier pilot's bias: drop left-censored columns and refit the free law
    kept = [d for d in data if np.isfinite(d[3]) and d[0] != "A2"]
    res = fit(kept)
    print(f"\n[비교] 왼쪽 검열 열을 버리면(이전 방식): 속도 {res['a']:+.3f}±{res['se'].get('a', float('nan')):.3f}, "
          f"β {res['p']:+.3f}±{res['se'].get('p', float('nan')):.3f}  (열 {len(kept)}개)")


if __name__ == "__main__":
    main()
