"""Thermal Schelleng upper law vs plain Schelleng on the 11 processed diagrams (post hoc analysis; Claude, 2026-10-19).
(1) per-diagram measured free exponents vs the thermal law's local exponents (structural identity e_beta = -(1+e_v)/2);
(2) leave-one-diagram-out (LODO) and leave-one-string-out (LOSO) predictive comparison of zero/one/two/three-parameter
    upper laws; (3) D_T2 replay (post hoc: T1-only training as in the frozen D_T2 plan); (4) tension-pair Z exponents.
Usage: OPENBLAS_NUM_THREADS=1 python analysis.py   -> analysis.json, analysis.log (stdout)"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import rows as R
import thermal_law as TL

HERE = Path(__file__).parent
STEP = np.log(40.0) / 49
NB = 200
MODELS = ["S0_dmu_steady", "T0_peak", "T0_break", "S1_strict", "T1k_peak", "T1k_break", "T1mu", "T2_mu_k", "E3_free_Z"]
ONE_PARAM = ["S1_strict", "T1k_peak", "T1k_break", "T1mu"]


def fit_x0(name, rows, x0=None):
    n, init, f = TL.MODELS[name]
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    nll = lambda p: TL.censored_nll(f(p[:n], lv, lb, lz), np.exp(p[n]), lo, hi)
    x0 = np.r_[init, np.log(0.3)] if x0 is None else np.asarray(x0, float)
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
    if n >= 1:  # one restart from the init to guard against a poor local start
        r2 = minimize(nll, np.r_[init, np.log(0.3)], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        r = r2 if r2.fun < r.fun else r
    return r.x


def free_fit(rows, x0=None):
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    X = np.column_stack([np.ones_like(lv), lv, lb])
    f = lambda p: TL.censored_nll(X @ p[:3], np.exp(p[3]), lo, hi)
    x0 = [0.0, 0.7, -1.0, np.log(0.3)] if x0 is None else x0
    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x


def score(name, p, rows):
    n = TL.MODELS[name][0]
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    mu = TL.predict(name, p[:n], lv, lb, lz)
    s = np.exp(p[n])
    nll = TL.censored_nll(mu, s, lo, hi) / len(rows)
    lev = TL.offset_fit(mu, lo, hi)[0]
    fin = np.isfinite(lo) & np.isfinite(hi)
    med = float(np.median(np.abs(mu[fin] - 0.5 * (lo[fin] + hi[fin])))) if fin.any() else np.nan
    hit = np.mean([(m >= a - STEP if np.isfinite(a) else True) and (m <= b + STEP if np.isfinite(b) else True)
                   for m, a, b in zip(mu, lo, hi)])
    ex = TL.design_exponents(name, p[:n], lv, lb, lz)
    return dict(nll=float(nll), level_err=lev, med_abs_finite=med, hit=float(hit), e_v=ex["e_v"], e_beta=ex["e_beta"], n=len(rows))


def main():
    t0 = time.time()
    RW = R.all_rows()
    allr = [r for t in R.TAGS for r in RW[t]]
    out = {"tags": R.TAGS}
    # ---------------------------------------------------------------- full fits
    full = {m: fit_x0(m, allr) for m in MODELS}
    out["full_fit"] = {m: dict(theta=[float(x) for x in full[m][:-1]], sigma=float(np.exp(full[m][-1])),
                               mu_s=TL.mu_s_of(m, full[m][:-1])) for m in MODELS}
    print("full fits (all 11):", {m: (np.round(full[m][:-1], 3).tolist(), round(float(np.exp(full[m][-1])), 3)) for m in MODELS}, flush=True)

    # ---------------------------------------------------------------- (1) measured vs thermal exponents per diagram
    rng = np.random.default_rng(7)
    out["per_diagram"] = {}
    print("\n(1) per diagram: measured free censored exponents [boot 95%] vs thermal local exponents (T1k_break, T1k_peak, T1mu fitted on all 11)")
    print("    and the thermal identity e_beta + (1 + e_v)/2 = 0 evaluated with the MEASURED e_v")
    for t in R.TAGS:
        rs = RW[t]
        p = free_fit(rs)
        bs = []
        for _ in range(NB):
            ii = rng.integers(0, len(rs), len(rs))
            bs.append(free_fit([rs[i] for i in ii], p))
        bs = np.array(bs)
        ident = bs[:, 2] + (1 + bs[:, 1]) / 2
        lv, lb, lz, lo, hi = TL.arrays(rs, R.ZPUB)
        loc = {}
        for m in ("T1k_break", "T1k_peak", "T1mu"):
            n = TL.MODELS[m][0]
            yF = TL.predict(m, full[m][:n], lv, lb, lz)
            le = TL.local_exponents(yF, np.exp(lv), TL.mu_s_of(m, full[m][:n]))
            ex = TL.design_exponents(m, full[m][:n], lv, lb, lz)
            loc[m] = dict(design_e_v=ex["e_v"], design_e_beta=ex["e_beta"], local_s_median=float(np.median(le["s"])),
                          local_e_v_range=[float(np.min(le["e_v"])), float(np.max(le["e_v"]))],
                          local_e_beta_range=[float(np.min(le["e_beta"])), float(np.max(le["e_beta"]))],
                          dmu_median=float(np.median(le["dmu"])), mu_d_median=float(np.median(le["mu_d"])))
        d = dict(e_v=float(p[1]), e_v_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), e_beta=float(p[2]),
                 e_beta_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
                 identity=float(p[2] + (1 + p[1]) / 2), identity_ci=np.percentile(ident, [2.5, 97.5]).tolist(), thermal=loc,
                 n_cols=len(rs), n_right_censored=int(np.sum(np.isfinite(lo) & ~np.isfinite(hi))), n_finite=int(np.sum(np.isfinite(lo) & np.isfinite(hi))))
        out["per_diagram"][t] = d
        tb = loc["T1k_break"]; tp = loc["T1k_peak"]
        print(f"  {t:7s} v {p[1]:+.2f} [{d['e_v_ci'][0]:+.2f},{d['e_v_ci'][1]:+.2f}] b {p[2]:+.2f} [{d['e_beta_ci'][0]:+.2f},{d['e_beta_ci'][1]:+.2f}] "
              f"| identity {d['identity']:+.2f} [{d['identity_ci'][0]:+.2f},{d['identity_ci'][1]:+.2f}] "
              f"| T-break v {tb['design_e_v']:+.2f} b {tb['design_e_beta']:+.2f} (s {tb['local_s_median']:.2f}) "
              f"| T-peak v {tp['design_e_v']:+.2f} b {tp['design_e_beta']:+.2f} (s {tp['local_s_median']:.2f}, dmu {tp['dmu_median']:.2f}) "
              f"| fin/right {d['n_finite']}/{d['n_right_censored']}", flush=True)
    # pooled identity test
    pf = free_fit(allr)
    tags_idx = {t: [i for i, r in enumerate(allr) if r["tag"] == t] for t in R.TAGS}
    bsp = []
    for _ in range(NB):  # cluster bootstrap over diagrams
        tt = rng.choice(R.TAGS, len(R.TAGS))
        bsp.append(free_fit([allr[i] for t in tt for i in tags_idx[t]], pf))
    bsp = np.array(bsp)
    idp = bsp[:, 2] + (1 + bsp[:, 1]) / 2
    out["pooled_free"] = dict(e_v=float(pf[1]), e_beta=float(pf[2]), e_v_ci=np.percentile(bsp[:, 1], [2.5, 97.5]).tolist(),
                              e_beta_ci=np.percentile(bsp[:, 2], [2.5, 97.5]).tolist(), identity=float(pf[2] + (1 + pf[1]) / 2),
                              identity_ci=np.percentile(idp, [2.5, 97.5]).tolist(), boot="cluster over diagrams")
    print(f"  pooled (11 diagrams, cluster boot): v {pf[1]:+.3f} {np.round(out['pooled_free']['e_v_ci'], 2)} b {pf[2]:+.3f} "
          f"{np.round(out['pooled_free']['e_beta_ci'], 2)} identity {out['pooled_free']['identity']:+.3f} {np.round(out['pooled_free']['identity_ci'], 3)}", flush=True)
    print(f"  [{time.time() - t0:.0f} s]", flush=True)

    # ---------------------------------------------------------------- (2) LODO and LOSO
    def cv(groups, label):
        res = {m: {} for m in MODELS}
        for g, test_tags in groups.items():
            tr = [r for t in R.TAGS if t not in test_tags for r in RW[t]]
            for m in MODELS:
                p = fit_x0(m, tr, full[m])
                for t in test_tags:
                    sc = score(m, p, RW[t])
                    ob = out["per_diagram"][t]
                    sc.update(err_e_v=sc["e_v"] - ob["e_v"], err_e_beta=sc["e_beta"] - ob["e_beta"],
                              theta=[float(x) for x in p[:-1]], sigma=float(np.exp(p[-1])))
                    res[m][t] = sc
            print(f"  {label} fold {g} done [{time.time() - t0:.0f} s]", flush=True)
        summ = {}
        for m in MODELS:
            v = res[m].values()
            summ[m] = dict(mean_nll=float(np.mean([x["nll"] for x in v])), rms_level=float(np.sqrt(np.mean([x["level_err"] ** 2 for x in v]))),
                           mean_abs_level=float(np.mean([abs(x["level_err"]) for x in v])),
                           median_med_abs_finite=float(np.nanmedian([x["med_abs_finite"] for x in v])),
                           mean_hit=float(np.mean([x["hit"] for x in v])),
                           rms_err_e_v=float(np.sqrt(np.mean([x["err_e_v"] ** 2 for x in v]))),
                           rms_err_e_beta=float(np.sqrt(np.mean([x["err_e_beta"] ** 2 for x in v]))))
        return res, summ

    lodo_groups = {t: [t] for t in R.TAGS}
    strings = sorted(set(R.STRING.values()))
    loso_groups = {s: [t for t in R.TAGS if R.STRING[t] == s] for s in strings}
    out["LODO"], out["LODO_summary"] = cv(lodo_groups, "LODO")
    out["LOSO"], out["LOSO_summary"] = cv(loso_groups, "LOSO")
    for lab in ("LODO", "LOSO"):
        print(f"\n(2) {lab}: held-out mean NLL/column | RMS level error | mean |level| | median finite |log err| | hit | RMS e_v err | RMS e_beta err")
        for m in MODELS:
            s = out[f"{lab}_summary"][m]
            print(f"  {m:14s} {s['mean_nll']:.3f} | {s['rms_level']:.3f} | {s['mean_abs_level']:.3f} | {s['median_med_abs_finite']:.3f} | "
                  f"{s['mean_hit']:.2f} | {s['rms_err_e_v']:.2f} | {s['rms_err_e_beta']:.2f}", flush=True)
        # paired comparisons vs plain Schelleng
        ref = out[lab]["S1_strict"]
        pc = {}
        for m in ("T1k_peak", "T1k_break", "T1mu", "T2_mu_k", "E3_free_Z", "T0_break"):
            dn = np.array([out[lab][m][t]["nll"] - ref[t]["nll"] for t in R.TAGS])
            dl = np.array([abs(out[lab][m][t]["level_err"]) - abs(ref[t]["level_err"]) for t in R.TAGS])
            # bootstrap over held-out diagrams for the mean NLL difference
            bb = [np.mean(dn[rng.integers(0, len(dn), len(dn))]) for _ in range(2000)]
            pc[m] = dict(nll_diff_mean=float(dn.mean()), nll_diff_ci=np.percentile(bb, [2.5, 97.5]).tolist(),
                         wins_nll=int(np.sum(dn < 0)), wins_level=int(np.sum(dl < 0)), level_absdiff_mean=float(dl.mean()),
                         per_diagram_nll_diff={t: float(x) for t, x in zip(R.TAGS, dn)})
            print(f"    {m:12s} - S1: mean dNLL {dn.mean():+.4f} [{pc[m]['nll_diff_ci'][0]:+.4f},{pc[m]['nll_diff_ci'][1]:+.4f}] "
                  f"wins(NLL) {pc[m]['wins_nll']}/11 | mean d|level| {dl.mean():+.4f} wins(level) {pc[m]['wins_level']}/11", flush=True)
        out[f"{lab}_paired_vs_S1"] = pc
        print("    per-diagram level errors (held out):")
        for t in R.TAGS:
            print("     " + f"{t:7s} " + " ".join(f"{m}:{out[lab][m][t]['level_err']:+.3f}" for m in ("S1_strict", "T1k_peak", "T1k_break", "T1mu", "E3_free_Z")))

    # ---------------------------------------------------------------- (3) D_T2 replay (post hoc)
    print("\n(3) D_T2 replay (post hoc; D_T2 already scored once with S1 trained on T1 strings)")
    out["D_T2_replay"] = {}
    for lab, train_tags in (("T1_only", ["A1", "A2", "B1", "C1", "D1"]), ("all_but_D_T2", [t for t in R.TAGS if t != "D_T2"])):
        tr = [r for t in train_tags for r in RW[t]]
        rr = {}
        for m in MODELS:
            p = fit_x0(m, tr, full[m])
            sc = score(m, p, RW["D_T2"])
            sc["theta"] = [float(x) for x in p[:-1]]
            rr[m] = sc
            print(f"  {lab:12s} {m:14s} theta {np.round(p[:-1], 3)} | D_T2 level err {sc['level_err']:+.4f} | NLL {sc['nll']:.3f} | "
                  f"med finite {sc['med_abs_finite']:.3f} | e_v {sc['e_v']:+.2f} e_b {sc['e_beta']:+.2f} (obs {out['per_diagram']['D_T2']['e_v']:+.2f} / {out['per_diagram']['D_T2']['e_beta']:+.2f})", flush=True)
        out["D_T2_replay"][lab] = rr

    # ---------------------------------------------------------------- (4) tension pairs: predicted vs observed level shift
    print("\n(4) T1 -> T2 upper level shifts (observed = censored offset vs base_A; predicted = mean model difference on the T1 design)")
    lvlA = {t: TL.offset_fit(R.base_A(*TL.arrays(RW[t], R.ZPUB)[:2]), *TL.arrays(RW[t], R.ZPUB)[3:])[0] for t in R.TAGS}
    out["level_A"] = lvlA
    pairs = {"A": (["A1", "A2"], ["A_T2"]), "B": (["B1"], ["B_T2s1", "B_T2r1", "B_T2r2"]), "C": (["C1"], ["C_T2"]), "D": (["D1"], ["D_T2"])}
    out["pairs"] = {}
    for k, (a, b) in pairs.items():
        dobs = np.mean([lvlA[t] for t in b]) - np.mean([lvlA[t] for t in a])
        za, zb = R.ZPUB[a[0]], R.ZPUB[b[0]]
        des = [r for t in a for r in RW[t]]
        lv = np.array([r["lv"] for r in des]); lb = np.array([r["lb"] for r in des])
        pr = {}
        for m in ("S1_strict", "T1k_peak", "T1k_break", "T1mu", "E3_free_Z"):
            n = TL.MODELS[m][0]
            pr[m] = float(np.mean(TL.predict(m, full[m][:n], lv, lb, np.full(len(lv), np.log(zb))) -
                                  TL.predict(m, full[m][:n], lv, lb, np.full(len(lv), np.log(za)))))
        out["pairs"][k] = dict(dlnZ=float(np.log(zb / za)), dlevel_obs=float(dobs), z_exp_obs=float(dobs / np.log(zb / za)), pred=pr,
                               z_exp_pred={m: v / np.log(zb / za) for m, v in pr.items()})
        print(f"  {k}: dlnZ {np.log(zb / za):+.3f} | observed dlevel {dobs:+.3f} (Z exp {dobs / np.log(zb / za):+.2f}) | predicted " +
              " ".join(f"{m} {v:+.3f}" for m, v in pr.items()), flush=True)
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote analysis.json [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
