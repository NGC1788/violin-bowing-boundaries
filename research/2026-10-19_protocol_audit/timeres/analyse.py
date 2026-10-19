"""Does the super-linear speed exponent survive when speeds are compared at the
same time since the bow reached full speed?

For each settling time t, every (speed, beta column) gets a censored interval
for its lower boundary from the window labels at that t, and the law
log F_min = c + a log v + p log beta is fitted by interval-censored ML.
Only the strokes around the original boundary were extracted, so a boundary
that moves outside that force range is censored at the subset edge.
"""

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

OUT = Path.home() / "violin-rne/timeres"
NOMINAL = {"v1": 0.05, "v2": 0.10, "v3": 0.20}
RUN = 2


def load():
    import sys as _s; _f = _s.argv[1] if len(_s.argv) > 1 else "windows.csv.gz"; rows = [r for r in csv.DictReader(gzip.open(OUT / _f, "rt")) if not r["error"]]
    for r in rows:
        r["t"] = float(r["t"]); r["force"] = float(r["force"]); r["beta"] = float(r["beta"])
    return rows


def intervals_at(rows, t):
    """Lower-boundary interval per (speed, beta column) from the labels of windows starting at time t."""
    cols = defaultdict(list)
    for r in rows:
        if abs(r["t"] - t) < 1e-6 and r["force"] > 0:
            cols[(r["condition"], r["beta_level"])].append(r)
    out = []
    for (condition, _), cells in cols.items():
        cells.sort(key=lambda r: r["force"])
        if len(cells) < 4:
            continue
        h = [c["label"] == "helmholtz" for c in cells]
        f = np.log([c["force"] for c in cells])
        beta = float(np.median([c["beta"] for c in cells]))
        start = next((i for i in range(len(h) - RUN + 1) if all(h[i:i + RUN])), None)
        if start is None:
            lo, hi = f[-1], np.inf                 # boundary above the extracted force range
        elif start == 0:
            lo, hi = -np.inf, f[0]                 # boundary at or below the extracted range
        else:
            lo, hi = f[start - 1], f[start]
        out.append((NOMINAL[condition], beta, lo, hi))
    return out


def fit(data, fixed_a=None):
    lv = np.log([d[0] for d in data]); lb = np.log([d[1] for d in data])
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])

    def nll(t):
        c, a, p, ls = (t[0], fixed_a if fixed_a is not None else t[1], t[-2], t[-1])
        mu = c + a * lv + p * lb
        s = np.exp(ls)
        up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
        dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

    x0 = [0.0, -1.9, np.log(0.4)] if fixed_a is not None else [0.0, 1.0, -1.9, np.log(0.4)]
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    a = fixed_a if fixed_a is not None else r.x[1]
    return {"a": a, "p": r.x[-2], "sigma": float(np.exp(r.x[-1])), "nll": float(r.fun)}


def boot(data, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for d in data:
        by[d[0]].append(d)
    draws = []
    for _ in range(reps):
        sample = []
        for rows in by.values():
            idx = rng.integers(0, len(rows), len(rows))
            sample += [rows[i] for i in idx]
        r = fit(sample)
        draws.append((r["a"], r["p"]))
    return np.percentile(np.array(draws), [2.5, 97.5], axis=0)


def main():
    rows = load()
    max_t = {c: max(r["t"] for r in rows if r["condition"] == c) for c in NOMINAL}
    print(f"창 {len(rows)}개 · 속도별 마지막 창 시작 시각 " + ", ".join(f"{c} {t:.2f}s" for c, t in max_t.items()))
    common = sorted({r["t"] for r in rows if r["t"] <= min(max_t.values()) + 1e-9})
    print(f"\n같은 정착 시간 t에서 세 속도를 비교 (0.2초 창의 시작 시각)")
    print(f"{'t (s)':>6}{'열':>5}{'속도 지수':>12}{'β 지수':>10}   95% 구간 (속도, β)")
    for t in common:
        data = intervals_at(rows, t)
        if len({d[0] for d in data}) < 3:
            continue
        r = fit(data)
        ci = boot(data)
        print(f"{t:6.2f}{len(data):5}{r['a']:+12.2f}{r['p']:+10.2f}   [{ci[0,0]:+.2f}, {ci[1,0]:+.2f}]  [{ci[0,1]:+.2f}, {ci[1,1]:+.2f}]")

    print("\n속도별 정착 곡선: 시간에 따른 F_min (β 지수는 공통 추정, 기준 t=0 대비 log 변화)")
    for c in NOMINAL:
        ts = sorted({r["t"] for r in rows if r["condition"] == c})
        ts = ts[: min(len(ts), 12)]
        base = None
        line = []
        for t in ts:
            data = [d for d in intervals_at([r for r in rows if r["condition"] == c], t)]
            if len(data) < 8:
                continue
            lb = np.log([d[1] for d in data]); lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])

            def nll(x):
                mu = x[0] - 1.95 * lb
                up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / np.exp(x[1])), 1.0)
                dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / np.exp(x[1])), 0.0)
                return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

            level = minimize(nll, [0.0, np.log(0.4)], method="Nelder-Mead").x[0]
            base = level if base is None else base
            line.append(f"{t:.2f}s:{level - base:+.2f}")
        print(f"  {c} ({NOMINAL[c]} m/s): " + "  ".join(line))


if __name__ == "__main__":
    main()
