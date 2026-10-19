"""Same interval-censored UPPER-boundary estimator applied to Schoonderwaldt, Guettler & Askenfelt (2008)
published author classifications (violin D4 steel string, rigid monochord, 4 speeds; Codex's vector extraction,
~/violin-bowing-boundaries/research/2026-10-13/schoonderwaldt2008/published_cells.csv).
Upper boundary per (speed, beta): first run of K Helmholtz cells from the top force; right-censored if the run
includes the top analysed cell; columns without a run are excluded. Unplotted/unknown cells are skipped.
Also: their Table I c_upper (fit with slope fixed at -1) -> implied speed exponent."""
import csv

import numpy as np
from scipy.stats import chi2

import ub

SRC = "/Users/jeongjihoon/violin-bowing-boundaries/research/2026-10-13/schoonderwaldt2008/published_cells.csv"
rows = list(csv.DictReader(open(SRC)))


def table(string, K, what="upper"):
    cols = {}
    for r in rows:
        if r["string"] != string or r["published_category"] in ("unplotted_unknown", "helmholtz_not_systematically_analyzed"):
            continue
        key = (float(r["bow_speed_nominal_m_s"]), float(r["beta_from_reported_rounded_distance"]))
        cols.setdefault(key, []).append((float(r["bow_force_nominal_N"]), r["published_category"] == "helmholtz", r["published_category"]))
    speeds = sorted({k[0] for k in cols})
    out, stat = [], {}
    for (v, b), cells in sorted(cols.items()):
        cells.sort()
        f = [c[0] for c in cells]; h = [c[1] for c in cells]
        lo, hi, st = (ub.upper_interval(f, h, K) if what == "upper" else ub.lower_interval(f, h, K))
        stat.setdefault(v, []).append(st)
        if st == "none":
            continue
        out.append((speeds.index(v), 0, np.log(v), np.log(b), lo, hi))
    return np.array(out), stat


def report(D, label, nb=1000):
    full, nll, z = ub.fit(D)
    _, bs = ub.boot(D, nboot=nb, seed=0, stat=lambda c: c[:3])
    _, n_s, _ = ub.fit(D, fixed={1: 1.0, 2: -1.0})
    _, n_a1, _ = ub.fit(D, fixed={1: 1.0})
    _, n_b1, _ = ub.fit(D, fixed={2: -1.0})
    _, n_a0, _ = ub.fit(D, fixed={1: 0.0})
    a_lo, a_hi = ub.ci(bs[:, 1]); b_lo, b_hi = ub.ci(bs[:, 2])
    print(f"{label}: cols {len(D)} (right-censored {np.sum(~np.isfinite(D[:, 5]))}, left {np.sum(~np.isfinite(D[:, 4]))})  "
          f"a {full[1]:+.3f} [{a_lo:+.2f},{a_hi:+.2f}]  b {full[2]:+.3f} [{b_lo:+.2f},{b_hi:+.2f}]  sigma {np.exp(full[3]):.2f}  "
          f"p(a=1,b=-1) {chi2.sf(2 * (n_s - nll), 2):.1e}  p(a=1) {chi2.sf(2 * (n_a1 - nll), 1):.1e}  "
          f"p(b=-1) {chi2.sf(2 * (n_b1 - nll), 1):.1e}  p(a=0) {chi2.sf(2 * (n_a0 - nll), 1):.1e}")
    return full, bs


def per_speed(D, speeds):
    for f, v in enumerate(speeds):
        Df = D[D[:, 0] == f]
        if len(Df) < 4:
            continue
        X = np.column_stack([np.ones(len(Df)), Df[:, 3]])
        c, _, z = ub.fit(Df, X=X)
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(500):
            idx = rng.integers(0, len(Df), len(Df))
            bs.append(ub.fit(Df[idx], X=X[idx], x0=z)[0][1])
        print(f"    v {v:.2f}: beta slope {c[1]:+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]  "
              f"F(beta 0.1) {np.exp(c[0] + c[1] * np.log(0.1)):.3f} N  (n {len(Df)}, right {np.sum(~np.isfinite(Df[:, 5]))})")


print("== Table I of Schoonderwaldt 2008 (c_upper with the beta slope fixed at -1): implied speed exponent")
v = np.array([0.05, 0.10, 0.15, 0.20]); c = np.array([66, 86, 125, 166.0])
A = np.column_stack([np.ones(4), np.log(v)])
coef = np.linalg.lstsq(A, np.log(c), rcond=None)[0]
print(f"   OLS slope of log c_upper on log v: {coef[1]:+.3f}; pairwise 5->10 {np.log(86/66)/np.log(2):+.2f}, 10->20 {np.log(166/86)/np.log(2):+.2f}")
for K in (2, 1, 3):
    D, stat = table("D4", K)
    print(f"\n== D4, K={K}: column status per speed " + "; ".join(f"{v}: int {s.count('interval')}, right {s.count('right')}, none {s.count('none')}" for v, s in stat.items()))
    report(D, f"D4 upper K={K}")
    if K == 2:
        per_speed(D, [0.05, 0.10, 0.15, 0.20])
        Dl, statl = table("D4", K, "lower")
        report(Dl, f"D4 lower K={K} (same code; cf. claude_extra/schoonderwaldt_fmin.py -0.09)")
D, stat = table("E5", 2)
print("\n== E5 (2 speeds, sparse systematic analysis): status " + "; ".join(f"{v}: {s}" for v, s in stat.items()))
if len(D) > 4:
    report(D, "E5 upper K=2", nb=300)
