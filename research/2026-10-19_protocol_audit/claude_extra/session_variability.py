"""Session-to-session variation of the measured boundary at fixed speed (Part 6: two repeated Schelleng diagrams
under the same conditions), compared with the A1/A2 specimen difference (Claude, 2026-10-18).

For each speed folder (= one recording session) and window start t: boundary intervals per beta column from the
current window force (first K=2 consecutive Helmholtz windows, as in verify_astra_boundaries.py), then a joint
interval-censored fit  log F* = c_session + b * log(beta)  with one level per session and a common beta slope.
Reports: (1) level differences between the two repeats at the same speed (the session effect), (2) the speed
exponent of each repeat, (3) the same quantities for A1 vs A2, and (4) how large a speed-exponent difference
session noise alone produces: SD(exponent) ~ sigma_session / sqrt(sum (log v - mean)^2).
Usage: python session_variability.py [features stem ...]   (default: Part 6, A1, A2)
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import verify_astra_boundaries as V

STEMS = sys.argv[1:] or ["2024-10-23_variability_setup", "2024-03-25_TypeA_sample1", "2024-04-19_TypeA_sample2"]


def session_intervals(rows, beta, col, t):
    """Per folder (= session), the boundary interval in each beta column."""
    groups = defaultdict(list)
    for k, d in rows.items():
        if t in d:
            groups[(k[0], col[k])].append((d[t][0], d[t][1], beta[k]))
    out = defaultdict(list)
    for (folder, c), g in groups.items():
        g.sort()
        h = [x[1] for x in g]
        lf = np.log(np.maximum([x[0] for x in g], 1e-4))
        s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
        lo, hi = (lf[-1], np.inf) if s is None else ((-np.inf, lf[0]) if s == 0 else (lf[s - 1], lf[s]))
        out[folder].append((np.log(np.median([x[2] for x in g])), lo, hi))
    return {f: np.array(v) for f, v in out.items()}


def fit_levels(ints):
    names = sorted(ints)
    lb = np.concatenate([ints[n][:, 0] for n in names]); lo = np.concatenate([ints[n][:, 1] for n in names])
    hi = np.concatenate([ints[n][:, 2] for n in names]); idx = np.concatenate([[i] * len(ints[n]) for i, n in enumerate(names)])
    k = len(names)
    def nll(x):
        mu = x[idx] + x[k] * lb
        s = np.exp(x[k + 1])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))
    r = minimize(nll, np.r_[np.full(k, -2.0), -1.0, np.log(0.5)], method="Powell", options={"maxiter": 200000, "xtol": 1e-8, "ftol": 1e-10})
    return dict(zip(names, r.x[:k])), r.x[k], np.exp(r.x[k + 1])


def main():
    for stem in STEMS:
        try:
            rows, beta, speed = V.load(stem)
        except FileNotFoundError:
            print(f"{stem}: 특징 파일 없음 (아직 처리 전)")
            continue
        col = V.columns(beta)
        print(f"\n{stem}: 긋기 {len(rows)}, 세션(속도 폴더) {len(speed)}")
        for t in (0.0, 0.3):
            ints = session_intervals(rows, beta, col, t)
            levels, bslope, sigma = fit_levels(ints)
            by_speed = defaultdict(list)
            for f, c in levels.items():
                by_speed[round(speed[f], 3)].append((f, c))
            lines = []
            for v, items in sorted(by_speed.items()):
                txt = ", ".join(f"{f.split('/')[-2] + '/' if f.count('/') > 1 else ''}{f.split('/')[-1]}: {np.exp(c + bslope * np.log(0.1)):.3f} N" for f, c in items)
                if len(items) == 2:
                    txt += f"  (반복 간 log 차이 {abs(items[0][1] - items[1][1]):.2f})"
                lines.append(f"v={v}: {txt}")
            print(f"  t={t}: β 기울기 {bslope:+.2f}, 잔차 σ {sigma:.2f} | β=0.1 경계 — " + " | ".join(lines))
            repeats = defaultdict(dict)
            for f, c in levels.items():
                repeats[f.rsplit("/", 1)[0]][round(speed[f], 3)] = c
            for rep, d in repeats.items():
                if len(d) >= 3:
                    v = np.log(sorted(d)); c = np.array([d[x] for x in sorted(d)])
                    print(f"    {rep}: 속도 지수(수준 기준) {np.polyfit(v, c, 1)[0]:+.2f}")


if __name__ == "__main__":
    main()
