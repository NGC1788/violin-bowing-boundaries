"""Post hoc follow-ups (Claude, 2026-10-19), 11 processed diagrams only:
(a) LODO/LOSO for the post hoc models of thermal_law2.py (T2km_peak, T2km_break, S2m) next to S1_strict, T1k_break, E3_free_Z;
(b) curvature tests that separate a thermal (power-controlled) contrast from the data:
    - speed curvature: thermal law => local speed exponent RISES with v (more power -> smaller contrast elasticity s);
      measured: speed-category fit, e(v1->v2) vs e(v2->v3);
    - force/beta curvature: thermal law => e_v rises and |e_beta| rises toward small beta (higher F v);
      measured: censored fit with lv*lb and lb^2 terms and diagram fixed offsets.
Usage: OPENBLAS_NUM_THREADS=1 python analysis2.py -> analysis2.json, stdout log"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import rows as R
import thermal_law as TL
import thermal_law2 as TL2

HERE = Path(__file__).parent
STEP = np.log(40.0) / 49
NB = 200
MODELS = ["S1_strict", "T1k_break", "T1k_peak", "E3_free_Z", "T2km_peak", "T2km_break", "S2m"]


def fit(name, rows, x0=None):
    n, init, f = TL2.MODELS[name]
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    nll = lambda p: TL.censored_nll(f(p[:n], lv, lb, lz), np.exp(p[n]), lo, hi)
    starts = [np.r_[init, np.log(0.3)]] + ([np.asarray(x0, float)] if x0 is not None else [])
    best = None
    for s in starts:
        r = minimize(nll, s, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        best = r if best is None or r.fun < best.fun else best
    return best.x


def score(name, p, rows):
    n = TL2.MODELS[name][0]
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    mu = TL2.predict(name, p[:n], lv, lb, lz)
    fin = np.isfinite(lo) & np.isfinite(hi)
    ex = TL2.design_exponents(name, p[:n], lv, lb, lz)
    return dict(nll=float(TL.censored_nll(mu, np.exp(p[n]), lo, hi) / len(rows)), level_err=TL.offset_fit(mu, lo, hi)[0],
                med_abs_finite=float(np.median(np.abs(mu[fin] - 0.5 * (lo[fin] + hi[fin])))), e_v=ex["e_v"], e_beta=ex["e_beta"])


def cfit(X, lo, hi, x0):
    f = lambda p: TL.censored_nll(X @ p[:-1], np.exp(p[-1]), lo, hi)
    return minimize(f, x0, method="Nelder-Mead", options={"maxiter": 60000, "xatol": 1e-8, "fatol": 1e-10}).x


def speedcat(rows):
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    sp = sorted(set(np.round(lv, 6)))
    X = np.column_stack([np.isclose(lv, s).astype(float) for s in sp] + [lb])
    return X, lo, hi, sp


def main():
    t0 = time.time()
    A = json.loads((HERE / "analysis.json").read_text())
    RW = R.all_rows()
    allr = [r for t in R.TAGS for r in RW[t]]
    rng = np.random.default_rng(11)
    out = {}
    full = {m: fit(m, allr) for m in MODELS}
    out["full"] = {m: dict(theta=[float(x) for x in full[m][:-1]], sigma=float(np.exp(full[m][-1]))) for m in MODELS}
    print("full fits:", {m: np.round(full[m], 3).tolist() for m in MODELS}, flush=True)
    # ------------------------------------------------ (a) CV of post hoc models
    for lab, groups in (("LODO", {t: [t] for t in R.TAGS}),
                        ("LOSO", {s: [t for t in R.TAGS if R.STRING[t] == s] for s in sorted(set(R.STRING.values()))})):
        res = {m: {} for m in MODELS}
        for g, test in groups.items():
            tr = [r for t in R.TAGS if t not in test for r in RW[t]]
            for m in MODELS:
                p = fit(m, tr, full[m])
                for t in test:
                    sc = score(m, p, RW[t])
                    sc.update(err_e_v=sc["e_v"] - A["per_diagram"][t]["e_v"], err_e_beta=sc["e_beta"] - A["per_diagram"][t]["e_beta"],
                              theta=[float(x) for x in p[:-1]])
                    res[m][t] = sc
        print(f"\n(a) {lab}: mean NLL | RMS level | median finite | RMS e_v err | RMS e_b err | dNLL vs S1 [95% boot over diagrams] wins", flush=True)
        summ = {}
        for m in MODELS:
            v = [res[m][t] for t in R.TAGS]
            dn = np.array([res[m][t]["nll"] - res["S1_strict"][t]["nll"] for t in R.TAGS])
            bb = [np.mean(dn[rng.integers(0, 11, 11)]) for _ in range(2000)]
            summ[m] = dict(mean_nll=float(np.mean([x["nll"] for x in v])), rms_level=float(np.sqrt(np.mean([x["level_err"] ** 2 for x in v]))),
                           med_finite=float(np.median([x["med_abs_finite"] for x in v])),
                           rms_e_v=float(np.sqrt(np.mean([x["err_e_v"] ** 2 for x in v]))), rms_e_beta=float(np.sqrt(np.mean([x["err_e_beta"] ** 2 for x in v]))),
                           dnll_vs_S1=float(dn.mean()), dnll_ci=np.percentile(bb, [2.5, 97.5]).tolist(), wins=int(np.sum(dn < 0)))
            s = summ[m]
            print(f"  {m:11s} {s['mean_nll']:.3f} | {s['rms_level']:.3f} | {s['med_finite']:.3f} | {s['rms_e_v']:.2f} | {s['rms_e_beta']:.2f} | "
                  f"{s['dnll_vs_S1']:+.4f} [{s['dnll_ci'][0]:+.4f},{s['dnll_ci'][1]:+.4f}] {s['wins']}/11", flush=True)
        out[lab] = res; out[lab + "_summary"] = summ
    # T2km_break vs E3 (both near the free exponents): paired
    for lab in ("LODO", "LOSO"):
        dn = np.array([out[lab]["T2km_break"][t]["nll"] - out[lab]["E3_free_Z"][t]["nll"] for t in R.TAGS])
        print(f"  {lab} T2km_break - E3_free_Z: mean dNLL {dn.mean():+.4f}, wins {int(np.sum(dn < 0))}/11")
        out[lab + "_T2km_break_vs_E3"] = dict(mean=float(dn.mean()), wins=int(np.sum(dn < 0)))
    print(f"[{time.time() - t0:.0f} s]", flush=True)

    # ------------------------------------------------ (b1) speed curvature
    print("\n(b1) speed curvature: measured speed-category exponents e(v1->v2), e(v2->v3) [boot 95%] vs thermal/plain predictions on the same columns")
    out["speed_curvature"] = {}
    curv_obs, curv_th = [], []
    for t in R.TAGS:
        X, lo, hi, sp = speedcat(RW[t])
        p = cfit(X, lo, hi, np.r_[np.zeros(len(sp)), -1.0, np.log(0.3)])
        e12 = (p[1] - p[0]) / (sp[1] - sp[0]); e23 = (p[2] - p[1]) / (sp[2] - sp[1])
        bs = []
        for _ in range(NB):
            ii = rng.integers(0, len(lo), len(lo))
            q = cfit(X[ii], lo[ii], hi[ii], p)
            bs.append(((q[1] - q[0]) / (sp[1] - sp[0]), (q[2] - q[1]) / (sp[2] - sp[1])))
        bs = np.array(bs)
        lv, lb, lz, _, _ = TL.arrays(RW[t], R.ZPUB)
        pr = {}
        for m in ("T1k_break", "T1k_peak", "T2km_break", "T2km_peak"):
            n = TL2.MODELS[m][0]
            y = TL2.predict(m, full[m][:n], lv, lb, lz)
            # predicted speed-category means on the same beta columns (OLS with speed dummies + lb)
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            pr[m] = dict(e12=float((c[1] - c[0]) / (sp[1] - sp[0])), e23=float((c[2] - c[1]) / (sp[2] - sp[1])))
        d = dict(e12=float(e12), e23=float(e23), e12_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), e23_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                 curvature=float(e23 - e12), curvature_ci=np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5]).tolist(), pred=pr)
        out["speed_curvature"][t] = d
        curv_obs.append(e23 - e12); curv_th.append(pr["T1k_peak"]["e23"] - pr["T1k_peak"]["e12"])
        print(f"  {t:7s} e12 {e12:+.2f} [{d['e12_ci'][0]:+.2f},{d['e12_ci'][1]:+.2f}] e23 {e23:+.2f} [{d['e23_ci'][0]:+.2f},{d['e23_ci'][1]:+.2f}] "
              f"curv {e23 - e12:+.2f} [{d['curvature_ci'][0]:+.2f},{d['curvature_ci'][1]:+.2f}] | T-break {pr['T1k_break']['e12']:+.2f}/{pr['T1k_break']['e23']:+.2f} "
              f"T-peak {pr['T1k_peak']['e12']:+.2f}/{pr['T1k_peak']['e23']:+.2f}", flush=True)
    # pooled speed curvature with diagram-specific intercepts and beta slopes common
    lv, lb, lz, lo, hi = TL.arrays(allr, R.ZPUB)
    sp = sorted(set(np.round(lv, 6)))
    tags = np.array([r["tag"] for r in allr])
    D = np.column_stack([(tags == t).astype(float) for t in R.TAGS[1:]])
    X = np.column_stack([np.isclose(lv, s).astype(float) for s in sp] + [lb, D])
    p = cfit(X, lo, hi, np.r_[np.zeros(len(sp)), -1.0, np.zeros(D.shape[1]), np.log(0.3)])
    e12 = (p[1] - p[0]) / (sp[1] - sp[0]); e23 = (p[2] - p[1]) / (sp[2] - sp[1])
    idx = {t: np.flatnonzero(tags == t) for t in R.TAGS}
    bs = []
    for _ in range(100):
        tt = rng.choice(R.TAGS, 11)
        ii = np.concatenate([idx[t] for t in tt])
        q = cfit(X[ii], lo[ii], hi[ii], p)
        bs.append(((q[1] - q[0]) / (sp[1] - sp[0]), (q[2] - q[1]) / (sp[2] - sp[1])))
    bs = np.array(bs)
    out["speed_curvature_pooled"] = dict(e12=float(e12), e23=float(e23), curvature=float(e23 - e12),
                                         curvature_ci=np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5]).tolist(), boot="cluster over diagrams (100)",
                                         n_diagrams_curv_negative=int(np.sum(np.array(curv_obs) < 0)),
                                         thermal_peak_curvature_mean=float(np.mean(curv_th)))
    print(f"  pooled (diagram offsets): e12 {e12:+.3f} e23 {e23:+.3f} curvature {e23 - e12:+.3f} "
          f"{np.round(out['speed_curvature_pooled']['curvature_ci'], 3)}; diagrams with negative curvature {int(np.sum(np.array(curv_obs) < 0))}/11; "
          f"thermal (T1k_peak) predicted curvature mean {np.mean(curv_th):+.3f}", flush=True)

    # ------------------------------------------------ (b2) beta curvature / interaction
    print("\n(b2) pooled censored fit log F = diag offset + a lv + b lb + g lv*lb + q lb^2 (measured) vs the same regression on model predictions")
    Xq = np.column_stack([lv, lb, lv * lb, lb ** 2, np.ones_like(lv), D])
    p = cfit(Xq, lo, hi, np.r_[0.7, -1.0, 0.0, 0.0, 0.0, np.zeros(D.shape[1]), np.log(0.3)])
    bs = []
    for _ in range(100):
        tt = rng.choice(R.TAGS, 11)
        ii = np.concatenate([idx[t] for t in tt])
        bs.append(cfit(Xq[ii], lo[ii], hi[ii], p)[:4])
    bs = np.array(bs)
    # centre for interpretability: report at lv = log 0.1, lb = log 0.1 the local exponents
    lv0, lb0 = np.log(0.1), np.log(0.1)
    loc = lambda c: (c[0] + c[2] * lb0, c[1] + c[2] * lv0 + 2 * c[3] * lb0)
    meas = dict(a=float(p[0]), b=float(p[1]), g=float(p[2]), q=float(p[3]), g_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
                q_ci=np.percentile(bs[:, 3], [2.5, 97.5]).tolist(), e_v_at_ref=float(loc(p)[0]), e_beta_at_ref=float(loc(p)[1]))
    pred = {}
    Xo = np.column_stack([lv, lb, lv * lb, lb ** 2, np.ones_like(lv), D])
    for m in ("S1_strict", "T1k_break", "T1k_peak", "T2km_break", "T2km_peak", "E3_free_Z"):
        n = TL2.MODELS[m][0]
        y = TL2.predict(m, full[m][:n], lv, lb, lz)
        c = np.linalg.lstsq(Xo, y, rcond=None)[0]
        pred[m] = dict(g=float(c[2]), q=float(c[3]), e_v_at_ref=float(loc(c)[0]), e_beta_at_ref=float(loc(c)[1]))
    out["beta_curvature"] = dict(measured=meas, predicted=pred, ref="v 0.1, beta 0.1")
    print(f"  measured: g(lv*lb) {p[2]:+.3f} {np.round(meas['g_ci'], 3)} q(lb^2) {p[3]:+.3f} {np.round(meas['q_ci'], 3)} | local e_v {meas['e_v_at_ref']:+.2f} e_b {meas['e_beta_at_ref']:+.2f} at v,beta 0.1")
    for m, c in pred.items():
        print(f"  {m:11s} g {c['g']:+.3f} q {c['q']:+.3f} | local e_v {c['e_v_at_ref']:+.2f} e_b {c['e_beta_at_ref']:+.2f}")
    (HERE / "analysis2.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote analysis2.json [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
