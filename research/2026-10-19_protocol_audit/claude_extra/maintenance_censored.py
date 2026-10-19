"""Maintenance threshold from exits, interval-censored (Claude, 2026-10-18; A1/A2 development data only).
For each stroke that entered Helmholtz (K=2): if Helmholtz is later lost to a K=2 run of multiple-slip windows
while the force is falling, the maintenance threshold lies between the force of the first lost window and the
force of the last Helmholtz window before it (interval); if Helmholtz is never lost, the threshold is below the
lowest force at which Helmholtz was still observed (left-censored); strokes that lose Helmholtz to other regimes
(aperiodic/ambiguous/no-oscillation: upper-limit or noise exits) are dropped and counted. Establishment threshold
per stroke from onset force: entered -> threshold <= onset force (left-censored); never entered -> threshold >
onset force (right-censored). Both fitted with log F* = c + a log v + b log beta, normal errors; column-cluster
bootstrap (resample beta columns within speed)."""
import csv
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import verify_astra_boundaries as V
from entry_exit import load_all, RAMP


def per_stroke(s, K=2):
    h = [x[2] for x in s]
    e = next((i for i in range(len(h) - K + 1) if all(h[i:i + K])), None)
    if e is None:
        return None
    i = e + K
    last_h = s[e + K - 1][1]
    while i < len(s) - 1:
        if not s[i][2] and not s[i + 1][2]:
            kind = s[i][3]
            if kind == "multiple_slip" and s[i + 1][3] == "multiple_slip":
                return ("exit", np.log(max(s[i][1], 1e-4)), np.log(max(last_h, 1e-4)))
            return ("other", None, None)
        if s[i][2]:
            last_h = s[i][1]
        i += 1
    low = min(x[1] for x in s[e:] if x[2])
    return ("kept", -np.inf, np.log(max(low, 1e-4)))


def fit(lv, lb, lo, hi):
    def nll(x):
        mu = x[0] + x[1] * lv + x[2] * lb
        s = np.exp(x[3])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))
    return min((minimize(nll, [-1, a, -1.5, np.log(0.5)], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
                for a in (0.0, 1.0)), key=lambda r: r.fun).x


for tag, stem in V.SPECIMENS.items():
    seq, beta, speed = load_all(stem)
    col = V.columns(beta)
    onset = {(r["folder"], int(r["trial"])): float(r["force_at_onset"]) for r in csv.DictReader(open(RAMP[tag])) if not r["error"]}
    M, E, counts = [], [], defaultdict(int)
    for k, s in seq.items():
        if k not in onset or onset[k] <= 0.01:
            continue
        lv, lb, cid = np.log(speed[k[0]]), np.log(beta[k]), (k[0], col[k])
        r = per_stroke(s)
        lf0 = np.log(onset[k])
        E.append((lv, lb, -np.inf, lf0, cid) if r is not None else (lv, lb, lf0, np.inf, cid))
        if r is None:
            continue
        counts[r[0]] += 1
        if r[0] in ("exit", "kept"):
            M.append((lv, lb, r[1], r[2], cid))
    print(f"\n{tag}: 진입한 긋기 중 출구(다중 미끄럼으로 상실) {counts['exit']}, 끝까지 유지 {counts['kept']}, 다른 상실(제외) {counts['other']}")
    for name, rows in (("수립(출발 힘)", E), ("유지(상실 힘)", M)):
        A = np.array([r[:4] for r in rows], float)
        cid = [r[4] for r in rows]
        x = fit(*A.T)
        cols = sorted(set(cid)); idx = defaultdict(list)
        for i, c in enumerate(cid): idx[c].append(i)
        rng = np.random.default_rng(0); boots = []
        by_speed = defaultdict(list)
        for c in cols: by_speed[c[0]].append(c)
        for _ in range(100):
            pick = [c for f, cs in by_speed.items() for c in rng.choice(len(cs), len(cs))  for c in [cs[c]]]
            ii = np.concatenate([idx[c] for c in pick])
            boots.append(fit(*A[ii].T))
        b = np.array(boots)
        lvls = {v: float(np.exp(x[0] + x[1] * np.log(v) + x[2] * np.log(0.1))) for v in (0.05, 0.1, 0.2)}
        print(f"  {name}: 속도 지수 {x[1]:+.2f} [{np.percentile(b[:, 1], 2.5):+.2f}, {np.percentile(b[:, 1], 97.5):+.2f}], β 지수 {x[2]:+.2f} "
              f"[{np.percentile(b[:, 2], 2.5):+.2f}, {np.percentile(b[:, 2], 97.5):+.2f}], β=0.1 경계 " + " / ".join(f"{v}: {f:.3f} N" for v, f in lvls.items()) + f"  (긋기 {len(A)})")
