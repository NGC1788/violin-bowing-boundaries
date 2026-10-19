"""Maximum bow force (upper boundary) law across all local diagrams (post hoc; Claude, 2026-10-19, offline).

Statistic for the string level = final/tension_pairs.py level: interval-censored offset of a diagram's upper
boundary columns from the A1+A2 P-free law without Z (t = 0.2 s windows, K = 2, measured window force,
right-censored at the highest measured force of the column). Also: per-diagram free fit of the speed and beta
exponents, tension pairs, cross-string regression on published properties (Lampis et al. POMA 58, 035013 (2025)
Table 1), and a strict Schelleng law F_max = c Z v / beta with one global constant.
Usage: python upper_law.py [t_eval]   (environment: OPENBLAS_NUM_THREADS=1)
Writes upper_law_t<t>.json next to this file. Reads final/cache/*.npz (built by final/data.py)."""
import itertools
import json
import sys
from pathlib import Path

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
sys.argv = sys.argv[:1]  # final/compare.py reads sys.argv at import
sys.path.insert(0, str(Path.home() / "violin-rne/final"))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

import compare as C  # noqa: E402
import confirm_A_T2 as K  # noqa: E402  (offset_fit; sets STEMS['A_T2'])
import data as D  # noqa: E402

D.STEMS["B_T2s1"] = "2024-06-13_TypeB_T2_sample1"
D.STEMS["C_T2"] = "2024-06-11_TypeC_T2_sample1"
OUT = Path(__file__).parent
NB = int(__import__("os").environ.get("NB", 300))
TAGS = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2"]
GROUP = {"A1": "A T1", "A2": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "A_T2": "A T2",
         "B_T2s1": "B T2", "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2"}
# POMA 2025 Table 1: T (N), Z (kg/s), EI (1e-4 N m^2), d (mm), Zto
PROP = {"A T1": (145.31, 1.0592, 3.032, 0.947, 0.039), "B T1": (150.87, 1.0998, 4.397, 0.927, 0.055),
        "C T1": (144.40, 1.0525, 3.142, 1.077, 0.039), "D T1": (142.67, 1.0398, 5.246, 1.123, 0.064),
        "A T2": (163.03, 1.1884, 3.969, 1.001, 0.027), "B T2": (163.23, 1.1899, 7.295, 0.973, 0.060),
        "C T2": (153.04, 1.1154, 3.969, 1.083, 0.040), "D T2": (164.82, 1.2021, 5.766, 1.201, 0.039)}
ZPUB = {t: PROP[GROUP[t]][1] for t in TAGS}
LN01 = np.log(0.1)


def rows_for(tag):
    rs = C.intervals(D.load(tag), T)
    for r in rs:
        r["tag"] = tag
    return rs


def upper_rows(rows):
    return [r for r in rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]


def cfit(X, off, lo, hi, x0):
    """Fast interval-censored normal fit (same likelihood as compare.censored_fit), started at x0."""
    def nll(p):
        mu = X @ p[:-1] + off
        s = np.exp(p[-1])
        pr = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(pr, 1e-300, None)))
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
    return r.x


def design_free(rows):
    X = np.array([[1.0, r["lv"], r["lb"]] for r in rows])
    return X


def design_speedcat(rows):
    sp = sorted({round(r["lv"], 4) for r in rows})
    return np.array([[1.0 * (round(r["lv"], 4) == s) for s in sp] + [r["lb"]] for r in rows]), sp


def lohi(rows):
    return np.array([r["upper"][0] for r in rows]), np.array([r["upper"][1] for r in rows])


def boot_idx(n, rng):
    return rng.integers(0, n, n)


def censor_counts(rows):
    out = {}
    for s in sorted({round(float(np.exp(r["lv"])), 3) for r in rows}):
        rs = [r for r in rows if abs(np.exp(r["lv"]) - s) < 1e-3]
        lo, hi = lohi(rs)
        out[str(s)] = dict(n=len(rs), finite=int(np.sum(np.isfinite(lo) & np.isfinite(hi))),
                           right=int(np.sum(np.isfinite(lo) & ~np.isfinite(hi))),
                           left=int(np.sum(~np.isfinite(lo) & np.isfinite(hi))), none=int(np.sum(np.isnan(lo) & np.isnan(hi))),
                           ceiling_median_N=float(np.median([np.exp(r["fmax"]) for r in rs])))
    return out


def main():
    rng = np.random.default_rng(1)
    R = {t: rows_for(t) for t in TAGS}
    tr = R["A1"] + R["A2"]
    noz = C.physics_models(tr, {t: 1.0 for t in TAGS})
    base_A = lambda r: noz["P-free"][0](r, "A1")[1]  # A1+A2 upper law, no Z
    pA = noz["P-free"][1][1]  # (c, a_v, b_beta, log sigma)
    res = {"t": T, "A1A2_upper_law": dict(c=pA[0], a_v=pA[1], b_beta=pA[2], sigma=float(np.exp(pA[3]))), "diagrams": {}}
    print(f"t = {T}: A1+A2 upper law log F = {pA[0]:+.3f} {pA[1]:+.3f} log v {pA[2]:+.3f} log beta, sigma {np.exp(pA[3]):.3f}")

    lev, lev_b, lsch, lsch_b, lref = {}, {}, {}, {}, {}
    sch_base = lambda r: r["lv"] - r["lb"]
    for tag in TAGS:
        ur = upper_rows(R[tag])
        lo, hi = lohi(ur)
        X = design_free(ur)
        p = C.censored_fit(ur, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "upper")
        Xc, sp = design_speedcat(ur)
        pc = cfit(Xc, np.zeros(len(ur)), lo, hi, np.r_[np.full(len(sp), 1.0), -1.0, np.log(0.3)])
        lev[tag] = K.offset_fit(R[tag], base_A, "upper")
        lsch[tag] = K.offset_fit(R[tag], sch_base, "upper")
        lref[tag] = (p[0] + (p[1] + p[2]) * LN01) - (pA[0] + (pA[1] + pA[2]) * LN01)
        bp, bc, bl, bs = [], [], [], []
        for _ in range(NB):
            ii = boot_idx(len(ur), rng)
            bp.append(cfit(X[ii], np.zeros(len(ii)), lo[ii], hi[ii], p))
            bc.append(cfit(Xc[ii], np.zeros(len(ii)), lo[ii], hi[ii], pc))
            sub = [ur[i] for i in ii]
            bl.append(K.offset_fit(sub, base_A, "upper"))
            bs.append(K.offset_fit(sub, sch_base, "upper"))
        bp, bc = np.array(bp), np.array(bc)
        lev_b[tag], lsch_b[tag] = np.array(bl), np.array(bs)
        e12 = (pc[1] - pc[0]) / np.log(2); e23 = (pc[2] - pc[1]) / np.log(2)
        be12 = (bc[:, 1] - bc[:, 0]) / np.log(2); be23 = (bc[:, 2] - bc[:, 1]) / np.log(2)
        ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
        res["diagrams"][tag] = dict(
            group=GROUP[tag], censoring=censor_counts(R[tag]),
            free=dict(c=p[0], a_v=p[1], a_v_ci=ci(bp[:, 1]), b_beta=p[2], b_beta_ci=ci(bp[:, 2]), sigma=float(np.exp(p[3]))),
            speedcat=dict(e_v1v2=e12, e_v1v2_ci=ci(be12), e_v2v3=e23, e_v2v3_ci=ci(be23), b_beta=pc[len(sp)], b_beta_ci=ci(bc[:, len(sp)])),
            level=lev[tag], level_ci=ci(lev_b[tag]), level_se=float(np.std(lev_b[tag])),
            level_sch_shape=lsch[tag], level_sch_ci=ci(lsch_b[tag]), level_ref_v01_b01=lref[tag])
        d = res["diagrams"][tag]
        cc = d["censoring"]
        print(f"{tag:7s} v-exp {p[1]:+.2f} [{d['free']['a_v_ci'][0]:+.2f},{d['free']['a_v_ci'][1]:+.2f}]  b-exp {p[2]:+.2f} "
              f"[{d['free']['b_beta_ci'][0]:+.2f},{d['free']['b_beta_ci'][1]:+.2f}] | v1v2 {e12:+.2f} v2v3 {e23:+.2f} | "
              f"level {lev[tag]:+.3f} [{d['level_ci'][0]:+.3f},{d['level_ci'][1]:+.3f}] | Sch-shape level {lsch[tag]:+.3f} | ref-pt {lref[tag]:+.3f} | "
              f"fin/right per v: " + " ".join(f"{cc[s]['finite']}/{cc[s]['right']}" for s in cc), flush=True)

    # recordings -> strings: B_T2 Part 6 = mean of rep_1, rep_2; B_T2 string = mean of s1 and Part 6
    def comb(store, store_b, name, parts):
        store[name] = float(np.mean([store[p] for p in parts]))
        store_b[name] = np.mean([store_b[p] for p in parts], axis=0)
    for st, sb in ((lev, lev_b), (lsch, lsch_b)):
        comb(st, sb, "B_T2p6", ["B_T2r1", "B_T2r2"])
        comb(st, sb, "B_T2", ["B_T2s1", "B_T2p6"])
        comb(st, sb, "A_T1", ["A1", "A2"])
    noise = {}
    for a, b in (("B_T2r1", "B_T2r2"), ("B_T2s1", "B_T2p6"), ("A1", "A2")):
        dd = lev[a] - lev[b]; db = lev_b[a] - lev_b[b]
        noise[f"{a}-{b}"] = dict(diff=dd, ci=[float(x) for x in np.percentile(db, [2.5, 97.5])], se=float(np.std(db)),
                                 diff_sch_shape=lsch[a] - lsch[b])
        print(f"yardstick {a}-{b}: level diff {dd:+.3f} (boot SE {np.std(db):.3f}); Schelleng-shape level diff {lsch[a] - lsch[b]:+.3f}")
    res["yardsticks"] = noise

    # (1) tension pairs
    lnZ = lambda g: np.log(PROP[g][1]); lnEI = lambda g: np.log(PROP[g][2])
    pairs = [("A", "A_T1", "A_T2", "A T1", "A T2"), ("A(A1)", "A1", "A_T2", "A T1", "A T2"), ("A(A2)", "A2", "A_T2", "A T1", "A T2"),
             ("B", "B1", "B_T2", "B T1", "B T2"), ("B(s1)", "B1", "B_T2s1", "B T1", "B T2"), ("B(Part6)", "B1", "B_T2p6", "B T1", "B T2"),
             ("C", "C1", "C_T2", "C T1", "C T2")]
    res["pairs"] = {}
    print("\n(1) tension pairs T1 -> T2 (upper level shift; Schelleng exponent 1)")
    for name, a, b, ga, gb in pairs:
        dz, de = lnZ(gb) - lnZ(ga), lnEI(gb) - lnEI(ga)
        dl, dlb = lev[b] - lev[a], lev_b[b] - lev_b[a]
        ds = lsch[b] - lsch[a]
        ci = np.percentile(dlb, [2.5, 97.5])
        res["pairs"][name] = dict(dlnZ=dz, dlnEI=de, dlevel=dl, dlevel_ci=list(ci), z_exp=dl / dz, z_exp_ci=list(ci / dz),
                                  ei_exp=dl / de, dlevel_sch_shape=ds, z_exp_sch_shape=ds / dz)
        print(f"  {name:9s} dlnZ {dz:+.3f} dlnEI {de:+.3f} | dlevel {dl:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] -> Z exponent {dl / dz:+.2f} "
              f"[{ci[0] / dz:+.2f},{ci[1] / dz:+.2f}] | EI exponent {dl / de:+.2f} | Sch-shape dlevel {ds:+.3f} (Z exp {ds / dz:+.2f})")
    # pooled Z exponent through the origin over the three independent pairs A, B, C (inverse-variance weights)
    main3 = ["A", "B", "C"]
    x = np.array([res["pairs"][p]["dlnZ"] for p in main3])
    yb = np.array([lev_b[pp[2]] - lev_b[pp[1]] for pp in pairs if pp[0] in main3])
    y = np.array([res["pairs"][p]["dlevel"] for p in main3]); w = 1 / np.var(yb, axis=1)
    g = np.sum(w * x * y) / np.sum(w * x * x)
    gb = np.sum(w[:, None] * x[:, None] * yb, 0) / np.sum(w * x * x)
    chi2 = float(np.sum(w * (y - g * x) ** 2))
    xe = np.array([res["pairs"][p]["dlnEI"] for p in main3]); ge = np.sum(w * xe * y) / np.sum(w * xe * xe)
    chi2e = float(np.sum(w * (y - ge * xe) ** 2))
    res["pairs_pooled"] = dict(z_exp=g, z_exp_ci=list(np.percentile(gb, [2.5, 97.5])), chi2_common_Z=chi2, dof=2,
                               ei_exp=ge, chi2_common_EI=chi2e, chi2_schelleng_g1=float(np.sum(w * (y - x) ** 2)))
    print(f"  pooled A,B,C (through origin, weights 1/var): Z exponent {g:+.2f} [{np.percentile(gb, 2.5):+.2f},{np.percentile(gb, 97.5):+.2f}],"
          f" chi2(common slope) {chi2:.1f} on 2 dof; chi2 for g=1 {np.sum(w * (y - x) ** 2):.1f} on 3 dof | EI exponent {ge:+.2f}, chi2 {chi2e:.1f}")
    # add between-recording variance (from the three same-type yardsticks) to each pair difference
    tau2 = float(np.mean([v["diff"] ** 2 / 2 for v in noise.values()]))
    w2 = 1 / (np.var(yb, axis=1) + 2 * tau2)
    g2 = np.sum(w2 * x * y) / np.sum(w2 * x * x); se2 = float(np.sqrt(1 / np.sum(w2 * x * x)))
    chi2b = float(np.sum(w2 * (y - g2 * x) ** 2)); chi2b1 = float(np.sum(w2 * (y - x) ** 2))
    res["pairs_pooled"].update(tau_between_recording=float(np.sqrt(tau2)), z_exp_tau=g2, z_exp_tau_se=se2, chi2_common_Z_tau=chi2b, chi2_g1_tau=chi2b1)
    print(f"  with between-recording SD tau {np.sqrt(tau2):.3f}: Z exponent {g2:+.2f} +- {se2:.2f}, chi2(common) {chi2b:.1f}/2 dof, chi2(g=1) {chi2b1:.1f}/3 dof")

    # (2) cross-string regression on published properties
    STR8 = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2", "C_T2"]
    STR9 = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2p6", "C_T2"]
    grp = lambda s: GROUP.get(s, "B T2")
    def feats(names):
        P = np.array([PROP[grp(s)] for s in names])
        return {"lnZ": np.log(P[:, 1]), "lnEI": np.log(P[:, 2]), "Zto": P[:, 4], "d": P[:, 3]}
    def fit(Xc, yv):
        A = np.column_stack([np.ones(len(yv))] + Xc) if Xc else np.ones((len(yv), 1))
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        return b, A
    def evaluate(names, y_vec, F, cols, offset=None):
        yv = y_vec - (offset if offset is not None else 0)
        Xc = [F[c] for c in cols]
        b, A = fit(Xc, yv)
        resid = yv - A @ b
        n, k = len(yv), A.shape[1]
        rss = float(resid @ resid); tss = float(np.sum((y_vec - y_vec.mean()) ** 2))
        r2 = 1 - rss / tss
        aicc = n * np.log(rss / n) + 2 * (k + 1) + 2 * (k + 1) * (k + 2) / max(n - k - 2, 1e-9)
        loso, logo = [], []
        groups = [grp(s) for s in names]
        for i in range(n):
            m = np.arange(n) != i
            bi, _ = fit([x[m] for x in Xc], yv[m])
            loso.append(yv[i] - (np.r_[1.0, [x[i] for x in Xc]] @ bi))
        for gname in sorted(set(groups)):
            m = np.array([gg != gname for gg in groups])
            bi, _ = fit([x[m] for x in Xc], yv[m])
            for i in np.flatnonzero(~m):
                logo.append(yv[i] - (np.r_[1.0, [x[i] for x in Xc]] @ bi))
        se = np.sqrt(np.diag(rss / max(n - k, 1) * np.linalg.pinv(A.T @ A))) if n > k else np.full(k, np.nan)
        return dict(coef=[float(v) for v in b], se=[float(v) for v in se], r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / max(n - k, 1),
                    aicc=float(aicc), rmse_in=float(np.sqrt(rss / n)), loso_rmse=float(np.sqrt(np.mean(np.square(loso)))),
                    logo_rmse=float(np.sqrt(np.mean(np.square(logo)))), loso_resid=[float(v) for v in loso])
    preds = ["lnZ", "lnEI", "Zto", "d"]
    models = {"null": ([], None), "Schelleng Z^1 (slope fixed)": ([], "lnZ")}
    for c in preds:
        models[c] = ([c], None)
    for c1, c2 in itertools.combinations(preds, 2):
        models[f"{c1}+{c2}"] = ([c1, c2], None)
    for c in ("lnEI", "Zto", "d"):
        models[f"Schelleng Z^1 + {c}"] = ([c], "lnZ")
    res["regression"] = {}
    for label, names in (("n8", STR8), ("n9", STR9)):
        F = feats(names)
        yv = np.array([lev[s] for s in names])
        out = {"strings": names, "levels": [float(v) for v in yv],
               "pred_corr": {f"{a}~{b}": float(np.corrcoef(F[a], F[b])[0, 1]) for a, b in itertools.combinations(preds, 2)}}
        print(f"\n(2) cross-string regression ({label}: {names})\n    levels " + " ".join(f"{s} {v:+.3f}" for s, v in zip(names, yv)))
        for mname, (cols, off) in models.items():
            o = evaluate(names, yv, F, cols, F[off] if off else None)
            out[mname] = o
            print(f"    {mname:28s} coef {' '.join(f'{c:+.3f}±{s:.3f}' for c, s in zip(o['coef'], o['se']))} | R2 {o['r2']:+.2f} adjR2 {o['adj_r2']:+.2f} "
                  f"AICc {o['aicc']:+.1f} | in {o['rmse_in']:.3f} LOSO {o['loso_rmse']:.3f} LOGO {o['logo_rmse']:.3f}")
        # selection-bias permutation test: min LOSO RMSE over the 10 free single/pair models, labels permuted
        cand = [m for m in models if m in preds or "+" in m and not m.startswith("Schelleng")]
        def best(yy):
            return min(evaluate(names, yy, F, models[m][0])["loso_rmse"] for m in cand)
        obs = best(yv)
        prm = np.random.default_rng(7)
        nulls = np.array([best(prm.permutation(yv)) for _ in range(1000)])
        single = {c: out[c]["loso_rmse"] for c in preds}
        null_single = {c: np.array([evaluate(names, prm.permutation(yv), F, [c])["loso_rmse"] for _ in range(1000)]) for c in ("lnZ", "lnEI")}
        out["perm"] = dict(best_loso_obs=obs, p_best=float(np.mean(nulls <= obs)), null_median=float(np.median(nulls)),
                           p_lnZ=float(np.mean(null_single["lnZ"] <= single["lnZ"])), p_lnEI=float(np.mean(null_single["lnEI"] <= single["lnEI"])))
        print(f"    permutation (1000): best-of-10 LOSO {obs:.3f}, null median {np.median(nulls):.3f}, p = {out['perm']['p_best']:.3f}; "
              f"single lnZ p = {out['perm']['p_lnZ']:.3f}, lnEI p = {out['perm']['p_lnEI']:.3f}")
        allm = [m for m in models if m != "null"]
        def best_all(yy):
            return min(evaluate(names, yy, F, models[m][0], F[models[m][1]] if models[m][1] else None)["loso_rmse"] for m in allm)
        obs_all = best_all(yv)
        nulls_all = np.array([best_all(prm.permutation(yv)) for _ in range(1000)])
        out["perm"].update(best_all_obs=obs_all, p_best_all=float(np.mean(nulls_all <= obs_all)))
        print(f"    permutation over all 14 non-null models: best LOSO {obs_all:.3f}, p = {np.mean(nulls_all <= obs_all):.3f}")
        res["regression"][label] = out

    # (2b) transfer test: calibrate level models on the five T1 strings only, predict the T2 strings
    T1n = ["A1", "A2", "B1", "C1", "D1"]; T2n = ["A_T2", "B_T2s1", "B_T2p6", "C_T2"]
    F1, F2 = feats(T1n), feats(T2n)
    y1 = np.array([lev[s] for s in T1n]); y2 = np.array([lev[s] for s in T2n])
    res["transfer_T1_to_T2"] = {}
    print("\n(2b) calibrate on T1 strings (A1 A2 B1 C1 D1), predict T2 levels (A_T2, B_T2s1, B_T2 Part6, C_T2)")
    for mname in ["null", "Schelleng Z^1 (slope fixed)", "lnZ", "lnEI", "Zto", "d", "Schelleng Z^1 + lnEI", "Schelleng Z^1 + Zto", "lnEI+Zto"]:
        cols, off = models[mname]
        o1 = F1[off] if off else 0.0; o2 = F2[off] if off else 0.0
        b, _ = fit([F1[c] for c in cols], y1 - o1)
        pred = np.column_stack([np.ones(len(T2n))] + [F2[c] for c in cols]) @ b + o2
        err = y2 - pred
        rm = float(np.sqrt(np.mean(np.square([err[0], 0.5 * (err[1] + err[2]), err[3]]))))
        res["transfer_T1_to_T2"][mname] = dict(coef=[float(v) for v in b], err={t: float(e) for t, e in zip(T2n, err)}, rmse_3strings=rm)
        print(f"    {mname:28s} coef {' '.join(f'{c:+.3f}' for c in b)} | obs-pred " + " ".join(f"{t} {e:+.3f}" for t, e in zip(T2n, err)) + f" | RMSE(3 strings) {rm:.3f}")

    # (3) strict Schelleng F_max = c Z v / beta, one global constant
    print("\n(3) strict Schelleng upper law F = c Z v / beta (one global constant)")
    T1 = ["A1", "A2", "B1", "C1", "D1"]
    T2d = ["A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2"]
    s3 = {}
    for shape in ("strict v/beta", "A1+A2 shape"):
        bfun = (lambda r, z: np.log(z) + r["lv"] - r["lb"]) if shape == "strict v/beta" else (lambda r, z: np.log(z) + base_A(r))
        pooled = [(r, ZPUB[t]) for t in T1 for r in R[t]]
        # calibrate ln c on all T1 columns pooled
        lnc = K.offset_fit([r for r, _ in pooled], lambda r, _m={id(r): z for r, z in pooled}: bfun(r, _m[id(r)]), "upper")
        resid = {t: K.offset_fit(R[t], lambda r, z=ZPUB[t]: lnc + bfun(r, z), "upper") for t in TAGS}
        colerr = {}
        for t in TAGS:
            fin = [r for r in R[t] if np.isfinite(r["upper"][0]) and np.isfinite(r["upper"][1])]
            colerr[t] = float(np.median([abs(lnc + bfun(r, ZPUB[t]) - 0.5 * (r["upper"][0] + r["upper"][1])) for r in fin]))
        sd_T1 = float(np.std([resid[t] for t in T1], ddof=1))
        s3[shape] = dict(lnc=lnc, resid_level=resid, col_med_abs_err=colerr, sd_T1_resid=sd_T1)
        print(f"  [{shape}] ln c = {lnc:+.3f}; T1 residual levels " + " ".join(f"{t} {resid[t]:+.3f}" for t in T1) + f" (SD {sd_T1:.3f})")
        print(f"     T2 residual levels (obs - pred): " + " ".join(f"{t} {resid[t]:+.3f}" for t in T2d) +
              f" | column median |log err|: " + " ".join(f"{t} {colerr[t]:.3f}" for t in T2d))
    # repeat noise at column level: rep_1 vs rep_2 (same speed, same column index)
    def mids(t):
        d = D.load(t); out = {}
        for r in R[t]:
            if np.isfinite(r["upper"][0]) and np.isfinite(r["upper"][1]):
                i = r["idx"][0]
                out[(round(float(d["speed"][i]), 3), int(d["column"][i]))] = (0.5 * (r["upper"][0] + r["upper"][1]), r["lb"])
        return out
    m1, m2 = mids("B_T2r1"), mids("B_T2r2")
    common = [k for k in m1 if k in m2]
    dcol = np.array([m1[k][0] - m2[k][0] for k in common]); dlb = np.array([abs(m1[k][1] - m2[k][1]) for k in common])
    ms = mids("B_T2s1")
    common2 = [k for k in m1 if k in ms]
    dcol2 = np.array([0.5 * (m1[k][0] + m2.get(k, m1[k])[0]) - ms[k][0] for k in common2])
    s3["repeat_column"] = dict(n=len(common), med_abs=float(np.median(np.abs(dcol))), mean=float(np.mean(dcol)), max_dlb=float(dlb.max()),
                               s1_vs_p6_n=len(common2), s1_vs_p6_med_abs=float(np.median(np.abs(dcol2))))
    print(f"  repeat noise, column level: rep_1 vs rep_2 median |d log Fmax| {np.median(np.abs(dcol)):.3f} (n {len(common)}, mean {np.mean(dcol):+.3f}, max |d log beta| {dlb.max():.3f});"
          f" B_T2s1 vs Part 6 {np.median(np.abs(dcol2)):.3f} (n {len(common2)})")
    res["schelleng_global"] = s3
    res["levels"] = {k: float(v) for k, v in lev.items()}
    res["levels_sch_shape"] = {k: float(v) for k, v in lsch.items()}
    with open(OUT / f"upper_law_t{T}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
