"""T3b carry-over with a flexible design: within each column, y is detrended by a running local-quadratic fit over
force rank (window 9), so only stroke-to-stroke (high-frequency) residuals remain. Then: lag-1 autocorrelation of the
residuals in recording order, and does the onset-force residual of stroke n follow the residual end-of-stroke change of
stroke n-1 (a sensor offset carried over without re-zeroing)?  Also: floor level of strokes that clearly lost contact
(reading flat and < 0.03 N over the last 6 moving blocks) vs recording order and beta.  Usage: python -I t3b_carry.py"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np


def hf_resid(y, rank, col, w=9):
    res = np.full(y.size, np.nan)
    for c in np.unique(col):
        q = np.flatnonzero(col == c)
        q = q[np.argsort(rank[q])]
        r = rank[q].astype(float); v = y[q]
        for k in range(q.size):
            lo, hi = max(0, k - w // 2), min(q.size, k + w // 2 + 1)
            rr, vv = r[lo:hi], v[lo:hi]
            ok = np.isfinite(vv)
            ok[k - lo] = False  # leave-one-out
            if ok.sum() >= 5 and np.isfinite(v[k]):
                p = np.polyfit(rr[ok], vv[ok], 2)
                res[q[k]] = v[k] - np.polyval(p, r[k])
    return res


for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    dFend = np.full(S.n, np.nan); dF10 = np.full(S.n, np.nan)
    flat_floor = np.full(S.n, np.nan)
    for i in np.flatnonzero(S.valid):
        lm = S.last_move[i]
        dFend[i] = np.nanmedian(S.df[i, max(lm - 1, 0):lm + 1])
        j = S.at_travel(i, 0.10)
        if j >= 0:
            dF10[i] = S.df[i, j]
        seg = S.force_dc[i, lm - 7:lm - 1]
        post = S.force_dc[i, lm + 1:S.nvalid[i]]
        if np.isfinite(seg).sum() == 6 and np.all(seg < 0.03):
            sl = np.polyfit(np.arange(6), seg, 1)[0]
            if abs(sl) < 0.0015:
                flat_floor[i] = np.nanmedian(np.concatenate([seg, post]))
    for s in ("v1", "v2", "v3"):
        m = np.flatnonzero(S.folder == s)
        m = m[np.argsort(S.trial[m])]
        rank, col, tr = S.rank[m], S.col[m], S.trial[m]
        ron = hf_resid(np.log(np.clip(S.f_on[m], 1e-3, None)), rank, col)
        rend = hf_resid(dFend[m], rank, col)
        r10 = hf_resid(dF10[m], rank, col)
        def l1(x):
            a = x[:-1]; b = x[1:]; same = col[:-1] == col[1:]
            ok = same & np.isfinite(a) & np.isfinite(b)
            return np.corrcoef(a[ok], b[ok])[0, 1], ok.sum()
        a = ron[1:]; b = rend[:-1]; same = col[1:] == col[:-1]
        ok = same & np.isfinite(a) & np.isfinite(b)
        cc = np.corrcoef(a[ok], b[ok])[0, 1]
        # regression slope: how many N of onset change per N of previous end change (convert log resid to N)
        fon = S.f_on[m][1:][ok]
        slope = np.polyfit(b[ok], a[ok] * fon, 1)[0]
        bb = rend[1:]; okk = same & np.isfinite(bb) & np.isfinite(ron[1:])
        cc_same = np.corrcoef(ron[1:][okk], bb[okk])[0, 1]
        print(f"  {s}: hf-resid sd: log F_on {np.nanstd(ron):.4f}, dF10 {1000*np.nanstd(r10):.1f} mN, dFend {1000*np.nanstd(rend):.1f} mN; "
              f"lag-1 r: F_on {l1(ron)[0]:+.3f}, dF10 {l1(r10)[0]:+.3f}, dFend {l1(rend)[0]:+.3f}")
        print(f"      carry-over corr(resid F_on[n], resid dFend[n-1]) {cc:+.3f} (n {ok.sum()}), slope {slope:+.3f} N/N; "
              f"same-stroke corr(resid F_on[n], resid dFend[n]) {cc_same:+.3f}")
        ff = flat_floor[m]
        okf = np.isfinite(ff)
        if okf.sum() > 20:
            print(f"      contact-lost floor: n {okf.sum()}, median {1000*med(ff[okf]):+.1f} mN, IQR {1000*np.nanpercentile(ff,25):+.1f}..{1000*np.nanpercentile(ff,75):+.1f}; "
                  f"corr with trial {np.corrcoef(tr[okf], ff[okf])[0,1]:+.2f}, beta {np.corrcoef(S.beta[m][okf], ff[okf])[0,1]:+.2f}; "
                  f"by thirds of session {1000*med(ff[okf & (tr<=667)]):+.1f}/{1000*med(ff[okf & (tr>667) & (tr<=1333)]):+.1f}/{1000*med(ff[okf & (tr>1333)]):+.1f} mN")
        else:
            print(f"      contact-lost floor strokes: {okf.sum()}")
