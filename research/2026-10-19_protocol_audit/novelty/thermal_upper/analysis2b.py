"""Fix for analysis2.py (b1-pooled, b2): the 14-16 parameter Nelder-Mead fits did not converge (point estimates outside
their own bootstrap intervals). Re-fit with L-BFGS-B from an OLS start + Powell polish; cluster bootstrap over diagrams.
(Claude, 2026-10-19, post hoc, 11 processed diagrams only.) Usage: python analysis2b.py -> analysis2b.json"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import rows as R
import thermal_law as TL
import thermal_law2 as TL2

HERE = Path(__file__).parent
NB = 200


def cfit(X, lo, hi, x0):
    f = lambda p: TL.censored_nll(X @ p[:-1], np.exp(p[-1]), lo, hi)
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 5000})
    r2 = minimize(f, r.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-9})
    r3 = minimize(f, r2.x, method="L-BFGS-B", options={"maxiter": 5000})
    return r3.x, r3.fun


def ols_start(X, lo, hi):
    y = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo + 0.2, hi - 0.2))
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.r_[c, np.log(0.3)]


def main():
    RW = R.all_rows()
    allr = [r for t in R.TAGS for r in RW[t]]
    lv, lb, lz, lo, hi = TL.arrays(allr, R.ZPUB)
    tags = np.array([r["tag"] for r in allr])
    D = np.column_stack([(tags == t).astype(float) for t in R.TAGS[1:]])
    idx = {t: np.flatnonzero(tags == t) for t in R.TAGS}
    rng = np.random.default_rng(5)
    out = {}
    sp = sorted(set(np.round(lv, 6)))
    # b1 pooled speed curvature
    X1 = np.column_stack([np.isclose(lv, s).astype(float) for s in sp] + [lb, D])
    p1, f1 = cfit(X1, lo, hi, ols_start(X1, lo, hi))
    e = lambda p: ((p[1] - p[0]) / (sp[1] - sp[0]), (p[2] - p[1]) / (sp[2] - sp[1]))
    # b2 interaction / beta curvature
    X2 = np.column_stack([lv, lb, lv * lb, lb ** 2, np.ones_like(lv), D])
    p2, f2 = cfit(X2, lo, hi, ols_start(X2, lo, hi))
    b1, b2 = [], []
    for _ in range(NB):
        tt = rng.choice(R.TAGS, 11)
        ii = np.concatenate([idx[t] for t in tt])
        q1, _ = cfit(X1[ii], lo[ii], hi[ii], p1); b1.append(e(q1))
        q2, _ = cfit(X2[ii], lo[ii], hi[ii], p2); b2.append(q2[:4])
    b1, b2 = np.array(b1), np.array(b2)
    e12, e23 = e(p1)
    ci = lambda a: np.percentile(a, [2.5, 97.5]).tolist()
    out["speed_curvature_pooled"] = dict(e12=e12, e23=e23, curvature=e23 - e12, e12_ci=ci(b1[:, 0]), e23_ci=ci(b1[:, 1]),
                                         curvature_ci=ci(b1[:, 1] - b1[:, 0]), nll=float(f1), boot=f"cluster over diagrams ({NB})")
    lv0 = lb0 = np.log(0.1)
    loc = lambda c: (c[0] + c[2] * lb0, c[1] + c[2] * lv0 + 2 * c[3] * lb0)
    out["beta_curvature_measured"] = dict(a=p2[0], b=p2[1], g=p2[2], q=p2[3], g_ci=ci(b2[:, 2]), q_ci=ci(b2[:, 3]),
                                          e_v_ref=loc(p2)[0], e_beta_ref=loc(p2)[1], nll=float(f2))
    print(f"(b1) pooled speed-category exponents (diagram offsets, common beta slope): e12 {e12:+.3f} {np.round(ci(b1[:, 0]), 3)} "
          f"e23 {e23:+.3f} {np.round(ci(b1[:, 1]), 3)} curvature {e23 - e12:+.3f} {np.round(ci(b1[:, 1] - b1[:, 0]), 3)}")
    print(f"(b2) measured g(lv*lb) {p2[2]:+.3f} {np.round(ci(b2[:, 2]), 3)} q(lb^2) {p2[3]:+.3f} {np.round(ci(b2[:, 3]), 3)} | "
          f"local e_v {loc(p2)[0]:+.2f} e_b {loc(p2)[1]:+.2f} at v=beta=0.1")
    # model predictions through the same regressions (noise-free; OLS on predicted values)
    full = json.loads((HERE / "analysis2.json").read_text())["full"]
    out["model_pred"] = {}
    for m in ("S1_strict", "T1k_break", "T1k_peak", "T2km_break", "T2km_peak", "E3_free_Z"):
        th = full[m]["theta"]
        y = TL2.predict(m, th, lv, lb, lz)
        c1 = np.linalg.lstsq(X1, y, rcond=None)[0]; c2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        a12, a23 = e(c1)
        out["model_pred"][m] = dict(e12=a12, e23=a23, curvature=a23 - a12, g=c2[2], q=c2[3], e_v_ref=loc(c2)[0], e_beta_ref=loc(c2)[1])
        print(f"   {m:11s} e12 {a12:+.3f} e23 {a23:+.3f} curvature {a23 - a12:+.3f} | g {c2[2]:+.3f} q {c2[3]:+.3f}")
    (HERE / "analysis2b.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
