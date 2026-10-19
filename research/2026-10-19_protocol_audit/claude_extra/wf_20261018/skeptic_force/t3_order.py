"""T3 recording order and cross-stroke continuity (angles 2 and 3).
Trials are recorded in index order inside each speed folder (one session). A2: beta ascending, force ascending within
a column; A1 ('r' folders): beta descending, force descending. A tuning break with plucks precedes every column.
 (a) two-way additive design fit  y = rank effect + column effect  for  y = log F_on  and  y = dF10 (force change at
     10 cm travel) and dFend (at the last moving block); residual lag-1 autocorrelation in recording order (within a
     column) vs a shuffled null; residual trend vs trial index.
 (b) carry-over: does the residual of F_on(n) depend on what happened to the force reading in stroke n-1
     (its end-of-stroke change) beyond the design?
 (c) pre-onset settling: reading change across the pre-onset blocks (bow placed, not moving).
 (d) first stroke of each column (after the tuning break; in A2 also right after a ~4 N stroke) vs its neighbours.
 (e) sensor zero: pre-onset bridge/nut DC across the session; floor reading of contact-lost strokes vs trial index/beta.
Usage: python -I t3_order.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np

rng = np.random.default_rng(1)


def additive_fit(y, rank, col):
    """least squares y ~ rank factor + col factor (main effects); returns residuals (nan kept)."""
    ok = np.isfinite(y)
    R = np.zeros((y.size, 50)); C = np.zeros((y.size, 40))
    R[np.arange(y.size), rank - 1] = 1; C[np.arange(y.size), col] = 1
    X = np.hstack([R, C[:, 1:]])
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res = np.full(y.size, np.nan); res[ok] = y[ok] - X[ok] @ beta
    return res


def lag1(res, order_key):
    """lag-1 correlation of residuals between consecutive trials inside the same column."""
    a, b = [], []
    for k in range(len(res) - 1):
        if order_key[k] == order_key[k + 1] and np.isfinite(res[k]) and np.isfinite(res[k + 1]):
            a.append(res[k]); b.append(res[k + 1])
    return np.corrcoef(a, b)[0, 1] if len(a) > 10 else np.nan, len(a)


for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    # per-stroke quantities
    dF10 = np.full(S.n, np.nan); dFend = np.full(S.n, np.nan); fend = np.full(S.n, np.nan); pre_drift = np.full(S.n, np.nan)
    for i in range(S.n):
        if not S.valid[i]:
            continue
        j = S.at_travel(i, 0.10)
        if j >= 0:
            dF10[i] = S.df[i, j]
        lm = S.last_move[i]
        dFend[i] = np.nanmedian(S.df[i, max(lm - 1, 0):lm + 1])
        post = S.force_dc[i, lm + 1:S.nvalid[i]]
        fend[i] = np.nanmedian(post) if np.isfinite(post).sum() else np.nan
        pre = S.force_dc[i, :S.onset[i]]
        pre = pre[np.isfinite(pre)]
        if pre.size >= 6:
            pre_drift[i] = np.median(pre[-3:]) - np.median(pre[:3])
    for s in ("v1", "v2", "v3"):
        m = np.flatnonzero(S.folder == s)
        m = m[np.argsort(S.trial[m])]  # recording order
        rank, col, tr = S.rank[m], S.col[m], S.trial[m]
        print(f"  --- {s}")
        for name, y in (("log F_on", np.log(np.clip(S.f_on[m], 1e-3, None))), ("dF10", dF10[m]), ("dFend", dFend[m])):
            res = additive_fit(y, rank, col)
            r1, n1 = lag1(res, col)
            null = []
            for _ in range(200):
                perm = res.copy()
                for c in range(40):
                    q = np.flatnonzero(col == c)
                    perm[q] = perm[rng.permutation(q)]
                null.append(lag1(perm, col)[0])
            # trend of residual with trial index inside columns (time within column, after removing rank effect)
            ok = np.isfinite(res)
            sd = np.nanstd(res)
            # residual mean in first/second half of the session
            half = tr <= 1000
            print(f"    {name:8s}: resid sd {sd:.4f}; lag-1 r {r1:+.3f} (null 95% {np.percentile(null,2.5):+.3f}..{np.percentile(null,97.5):+.3f}, n {n1}); "
                  f"mean resid 1st/2nd half {np.nanmean(res[half]):+.4f}/{np.nanmean(res[~half]):+.4f}")
            if name == "log F_on":
                res_on = res
            if name == "dFend":
                res_end = res
        # (b) carry-over: residual log F_on(n) vs dFend residual and raw dFend of stroke n-1 (same column, consecutive)
        a, b, c2 = [], [], []
        for k in range(1, len(m)):
            if col[k] == col[k - 1] and np.isfinite(res_on[k]) and np.isfinite(res_end[k - 1]):
                a.append(res_on[k]); b.append(res_end[k - 1]); c2.append(dFend[m[k - 1]])
        a, b, c2 = map(np.array, (a, b, c2))
        print(f"    carry-over: corr(resid log F_on[n], resid dFend[n-1]) {np.corrcoef(a, b)[0,1]:+.3f}; "
              f"corr(resid log F_on[n], dFend[n-1]) {np.corrcoef(a, c2)[0,1]:+.3f} (n {a.size})")
        # (c) pre-onset settling
        for lo, hi in ((1, 20), (21, 40), (41, 50)):
            q = (rank >= lo) & (rank <= hi)
            print(f"    pre-onset reading change (last-first pre-onset blocks, ~0.4 s) ranks {lo}-{hi}: median {1000*med(pre_drift[m][q]):+.1f} mN, "
                  f"p10/p90 {1000*np.nanpercentile(pre_drift[m][q],10):+.1f}/{1000*np.nanpercentile(pre_drift[m][q],90):+.1f}")
        # (d) first stroke of the column vs neighbours (in recording order): residuals
        first = np.array([k for k in range(len(m)) if k == 0 or col[k] != col[k - 1]])
        second = first + 1
        print(f"    first stroke of column (recorded after the tuning break): resid log F_on median {med(res_on[first]):+.4f} vs 2nd {med(res_on[second]):+.4f}; "
              f"resid dFend {med(res_end[first]):+.4f} vs 2nd {med(res_end[second]):+.4f}")
        # (e) sensor zeros
        bpre = np.array([np.nanmedian(S.bridge_dc[i, :S.onset[i]]) for i in m])
        npre = np.array([np.nanmedian(S.nut_dc[i, :S.onset[i]]) for i in m])
        print(f"    pre-onset bridge DC: median {1000*med(bpre):+.2f} mN, sd {1000*np.nanstd(bpre):.2f}, corr with trial {np.corrcoef(tr, bpre)[0,1]:+.2f}; "
              f"nut DC median {1000*med(npre):+.2f} mN sd {1000*np.nanstd(npre):.2f}")
        # bridge pre-onset vs previous stroke's end bridge DC (persistence of the bridge offset?)
        bend = np.array([np.nanmedian(S.bridge_dc[i, S.last_move[i] + 1:S.nvalid[i]]) for i in m])
        okk = np.isfinite(bend[:-1]) & np.isfinite(bpre[1:])
        print(f"    bridge: corr(pre-onset[n], end-of-record[n-1]) {np.corrcoef(bend[:-1][okk], bpre[1:][okk])[0,1]:+.3f}; "
              f"end-of-record bridge DC for ranks 1-5 median {med(bend[rank<=5]):+.3f}")
        floor = (rank <= 5)
        fl = fend[m]
        okf = floor & np.isfinite(fl)
        print(f"    floor reading (post-stop, ranks 1-5): median {1000*med(fl[okf]):+.1f} mN; corr with trial {np.corrcoef(tr[okf], fl[okf])[0,1]:+.2f}, "
              f"with beta {np.corrcoef(S.beta[m][okf], fl[okf])[0,1]:+.2f}; first/last 10 columns {1000*med(fl[okf & (col<10)]):+.1f}/{1000*med(fl[okf & (col>=30)]):+.1f} mN")
