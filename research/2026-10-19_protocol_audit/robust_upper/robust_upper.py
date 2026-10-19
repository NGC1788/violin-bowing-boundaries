"""Robustness of the maximum-force (upper boundary) results on the 11 development diagrams (Claude, 2026-10-19; POST HOC).
Plan: README.md (PLAN section written before this script was run). Items U1 (no 0.20 m/s), U2 (K = 3), U3 (canonical
beta-position cluster bootstrap, B 2000, seed 20261019), U4 (torsion-corrected impedance, defensibility rule), U5
(random-effects pooled Z exponent, HKSJ).
Reads ONLY the 11 development caches listed in PATHS (never the TGT_* files that sit next to D_T2.npz) and existing json.
Usage: OPENBLAS_NUM_THREADS=1 python -B robust_upper.py   [env NB_CAN=2000 NB_DIAG=1000 for a quick test set smaller]
Writes results.json in this folder; stdout is the log."""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist

HERE = Path(__file__).resolve().parent
RNE = Path.home() / "violin-rne"
TAGS = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2", "D_T2"]
PATHS = {t: RNE / "final/cache" / f"{t}.npz" for t in TAGS if t != "D_T2"}
PATHS["D_T2"] = RNE / "novelty/thermal_upper/cache/D_T2.npz"
T1 = ["A1", "A2", "B1", "C1", "D1"]
GROUP = {"A1": "A T1", "A2": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "A_T2": "A T2", "B_T2s1": "B T2",
         "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
# Lampis, Chatziioannou & Scavone, POMA 58, 035013 (2025) Table 1 (papers/Lampis2025_POMA_58_035013.txt lines 226-241):
# d (mm), T (N), mu (g/m), Z (kg/s), Zto (printed unit g m^2/s^2, manufacturer value), EI (1e-4 N m^2)
POMA = {"A T1": (0.947, 145.31, 7.721, 1.0592, 0.039, 3.032), "B T1": (0.927, 150.87, 8.017, 1.0998, 0.055, 4.397),
        "C T1": (1.077, 144.40, 7.671, 1.0525, 0.039, 3.142), "D T1": (1.123, 142.67, 7.578, 1.0398, 0.064, 5.246),
        "A T2": (1.001, 163.03, 8.662, 1.1884, 0.027, 3.969), "B T2": (0.973, 163.23, 8.674, 1.1899, 0.060, 7.295),
        "C T2": (1.083, 153.04, 8.128, 1.1154, 0.040, 3.969), "D T2": (1.201, 164.82, 8.767, 1.2021, 0.039, 5.766)}
LNZ = {t: float(np.log(POMA[GROUP[t]][3])) for t in TAGS}
BASE_A = (-0.1764277369756792, 0.6606815556587429, -1.1107153335353168)  # frozen A1+A2 law (novelty/upper_final/common.py)
PAIRS = {"A": ({"A1": 0.5, "A2": 0.5}, {"A_T2": 1.0}), "B": ({"B1": 1.0}, {"B_T2s1": 0.5, "B_T2r1": 0.25, "B_T2r2": 0.25}),
         "C": ({"C1": 1.0}, {"C_T2": 1.0}), "D": ({"D1": 1.0}, {"D_T2": 1.0})}
DLNZ = {p: float(np.log(POMA[f"{p} T2"][3] / POMA[f"{p} T1"][3])) for p in PAIRS}
T_EVAL = 0.2
NB_CAN = int(os.environ.get("NB_CAN", 2000))
NB_DIAG = int(os.environ.get("NB_DIAG", 1000))
SEED_CAN, SEED_DIAG = 20261019, 3


# ------------------------------------------------------------------------------------------------ data
def load(tag):
    return dict(np.load(PATHS[tag]))


def intervals(d, t, K):
    """= final/compare.intervals with K as a parameter; also returns the beta column index (cluster key)."""
    m = np.isclose(d["t"], t)
    groups = defaultdict(list)
    for i in np.flatnonzero(m):
        groups[(int(d["folder"][i]), int(d["column"][i]))].append(i)
    out = []
    for key, idx in groups.items():
        idx = sorted(idx, key=lambda i: d["force"][i])
        f = np.log(np.maximum(d["force"][idx], 1e-3)); h = d["H"][idx]
        runs = [i for i in range(len(h) - K + 1) if all(h[i:i + K])]
        if runs:
            s = runs[0]
            llo, lhi = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s])
            e = runs[-1] + K - 1
            ulo, uhi = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
        else:
            llo, lhi, ulo, uhi = f[-1], np.inf, np.nan, np.nan
        out.append(dict(lv=float(np.log(d["speed"][idx[0]])), lb=float(np.log(np.median(d["beta"][idx]))),
                        lo=float(ulo), hi=float(uhi), col=key[1], fmax=float(f[-1])))
    return out


class Rows:
    """Upper-boundary rows of one diagram as arrays (only rows with an upper interval)."""
    def __init__(self, rows):
        rows = [r for r in rows if not (np.isnan(r["lo"]) and np.isnan(r["hi"]))]
        self.lv = np.array([r["lv"] for r in rows]); self.lb = np.array([r["lb"] for r in rows])
        self.lo = np.array([r["lo"] for r in rows]); self.hi = np.array([r["hi"] for r in rows])
        self.col = np.array([r["col"] for r in rows])
        cols = np.unique(self.col)
        self.members = [np.flatnonzero(self.col == c) for c in cols]
        self.n = len(rows)

    def take(self, ii):
        new = Rows.__new__(Rows)
        new.lv, new.lb, new.lo, new.hi, new.col = self.lv[ii], self.lb[ii], self.lo[ii], self.hi[ii], self.col[ii]
        new.n = len(ii); new.members = None
        return new

    def boot(self, rng):
        pick = rng.integers(0, len(self.members), len(self.members))
        return self.take(np.concatenate([self.members[c] for c in pick]))


def cat(rs):
    new = Rows.__new__(Rows)
    for k in ("lv", "lb", "lo", "hi", "col"):
        setattr(new, k, np.concatenate([getattr(r, k) for r in rs]))
    new.n = len(new.lv); new.members = None
    return new


def census(r):
    fin = np.isfinite(r.lo) & np.isfinite(r.hi)
    right = np.isfinite(r.lo) & ~np.isfinite(r.hi)
    left = ~np.isfinite(r.lo) & np.isfinite(r.hi)
    out = {}
    for sp in sorted(set(np.round(np.exp(r.lv), 3))):
        m = np.isclose(np.exp(r.lv), sp, atol=1e-3)
        out[f"{sp:.2f}"] = dict(n=int(m.sum()), finite=int((fin & m).sum()), right=int((right & m).sum()), left=int((left & m).sum()))
    return out


# ------------------------------------------------------------------------------------------------ likelihood
def _ll_terms(mu, s, lo, hi):
    a = (lo - mu) / s; b = (hi - mu) / s
    flo, fhi = np.isfinite(lo), np.isfinite(hi)
    ll = np.empty(len(mu))
    both = flo & fhi
    if both.any():
        aa, bb = a[both], b[both]
        up = aa > 0
        v = np.empty(both.sum())
        v[up] = np.log(np.clip(norm.sf(aa[up]) - norm.sf(bb[up]), 1e-300, None))
        v[~up] = np.log(np.clip(norm.cdf(bb[~up]) - norm.cdf(aa[~up]), 1e-300, None))
        ll[both] = v
    r = flo & ~fhi; ll[r] = norm.logsf(a[r])
    l = ~flo & fhi; ll[l] = norm.logcdf(b[l])
    ll[~flo & ~fhi] = 0.0
    za = np.where(flo, a, 0.0); zb = np.where(fhi, b, 0.0)
    pa = np.where(flo, np.exp(np.minimum(norm.logpdf(za) - ll, 700.0)), 0.0)
    pb = np.where(fhi, np.exp(np.minimum(norm.logpdf(zb) - ll, 700.0)), 0.0)
    return ll, (pa - pb) / s, za * pa - zb * pb  # ll, dll/dmu, dll/dlogs


def fit(X, off, lo, hi, x0=None):
    """MLE of log F = X theta + off + N(0, s^2), interval censored. Returns (theta, s, ok)."""
    X = np.asarray(X, float)
    if x0 is None:
        x0 = np.zeros(X.shape[1])
    def f(p):
        ll, dmu, dls = _ll_terms(X @ p[:-1] + off, np.exp(p[-1]), lo, hi)
        return -ll.sum(), -np.r_[X.T @ dmu, dls.sum()]
    p0 = np.r_[x0, np.log(0.3)]
    r = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-8})
    ok = bool(r.success) or np.max(np.abs(r.jac)) < 1e-4
    return r.x[:-1], float(np.exp(r.x[-1])), ok


def offset(base, r):
    th, s, ok = fit(np.ones((r.n, 1)), base, r.lo, r.hi, x0=[np.median(np.where(np.isfinite(r.lo), r.lo, r.hi) - base)])
    return float(th[0]), ok


def free3(r, x0=None):
    th, s, ok = fit(np.column_stack([np.ones(r.n), r.lv, r.lb]), np.zeros(r.n), r.lo, r.hi, x0=x0)
    return th, ok


def pooled(rs, x0=None):
    """rs: list of Rows (one per diagram slot). log F = delta_slot + e_v lv + e_b lb."""
    al = cat(rs)
    D = np.zeros((al.n, len(rs)))
    k = 0
    for j, r in enumerate(rs):
        D[k:k + r.n, j] = 1.0; k += r.n
    X = np.column_stack([D, al.lv, al.lb])
    th, s, ok = fit(X, np.zeros(al.n), al.lo, al.hi, x0=x0)
    return th, s, ok


# ------------------------------------------------------------------------------------------------ estimates
def estimates(R, fixed_law=None, x0=None):
    """All per-variant quantities from a dict tag -> Rows. fixed_law: use this (c, a, b) instead of refitting R."""
    x0 = x0 or {}
    oks = []
    if fixed_law is None:
        law, ok = free3(cat([R["A1"], R["A2"]]), x0=x0.get("law", list(BASE_A))); oks.append(ok)
    else:
        law = np.asarray(fixed_law)
    lev = {}
    for t in TAGS:
        r = R[t]
        lev[t], ok = offset(law[0] + law[1] * r.lv + law[2] * r.lb, r); oks.append(ok)
    pair = {}
    for p, (w1, w2) in PAIRS.items():
        dl = sum(w * lev[t] for t, w in w2.items()) - sum(w * lev[t] for t, w in w1.items())
        pair[p] = dict(dlevel=dl, g=dl / DLNZ[p])
    t1 = cat([R[t] for t in T1])
    lz1 = np.concatenate([np.full(R[t].n, LNZ[t]) for t in T1])
    lnc, ok = offset(lz1 + t1.lv - t1.lb, t1); oks.append(ok)
    rD = R["D_T2"]
    dt2, ok = offset(lnc + LNZ["D_T2"] + rD.lv - rD.lb, rD); oks.append(ok)
    return dict(law=[float(x) for x in law], levels=lev, pairs=pair, lnc=lnc, D_T2_strict_err=dt2,
                A_T2_shift=lev["A_T2"], A_T2_g=lev["A_T2"] / DLNZ["A"], ok=all(oks))


def pct(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [np.nan, np.nan]


def sd(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if len(a) > 1 else np.nan


# ------------------------------------------------------------------------------------------------ variants
def build_variants():
    data = {t: load(t) for t in TAGS}
    raw = {}
    for name, K, drop02 in (("base", 2, False), ("U1_no020", 2, True), ("U2_K3", 3, False)):
        R = {}
        for t in TAGS:
            rows = intervals(data[t], T_EVAL, K)
            if drop02:
                rows = [r for r in rows if abs(np.exp(r["lv"]) - 0.2) > 0.01]
            R[t] = Rows(rows)
        raw[name] = R
    return data, raw


def check_against_compare(data):
    """K = 2 intervals must equal final/compare.intervals exactly (imported read-only; python -B => no pycache)."""
    argv = sys.argv; sys.argv = sys.argv[:1]
    sys.path.insert(0, str(RNE / "final"))
    import compare as C  # noqa
    sys.argv = argv
    worst, n = 0.0, 0
    for t in TAGS:
        a = C.intervals(data[t], T_EVAL)
        b = intervals(data[t], T_EVAL, 2)
        for x, y in zip(a, b):
            for u, v in zip(x["upper"], (y["lo"], y["hi"])):
                if np.isfinite(u) or np.isfinite(v):
                    worst = max(worst, abs(u - v)); n += 1
                elif not ((np.isnan(u) and np.isnan(v)) or (u == v)):
                    worst = np.inf
    return dict(n_bounds=n, max_abs_diff=worst)


def diag_boot_pooled(R, th0, rng, B):
    out, fails = [], 0
    for _ in range(B):
        pick = rng.integers(0, len(TAGS), len(TAGS))
        rs = [R[TAGS[i]] for i in pick]
        x0 = np.r_[th0[:-2][pick], th0[-2:]]
        th, s, ok = pooled(rs, x0=x0)
        if not ok:
            fails += 1
        out.append(th[-2:])
    return np.array(out), fails


def canonical(R, est0, rng, B, with_pooled, pooled0=None):
    """Stratified beta-position cluster bootstrap; all quantities from the same draws."""
    keys = ["A_T2_shift_fixedlaw", "A_T2_shift_refit", "D_T2_fixedc", "D_T2_joint", "lnc"] + \
           [f"dlevel_{p}" for p in PAIRS] + [f"g_{p}" for p in PAIRS] + ["e_v", "e_b"]
    res = {k: [] for k in keys}
    fails = 0
    law0 = est0["law"]
    for b in range(B):
        Rb = {t: R[t].boot(rng) for t in TAGS}
        e = estimates(Rb, x0={"law": law0})
        if not e["ok"]:
            fails += 1
        rA = Rb["A_T2"]
        sA, _ = offset(law0[0] + law0[1] * rA.lv + law0[2] * rA.lb, rA)
        rD = Rb["D_T2"]
        dF, _ = offset(est0["lnc"] + LNZ["D_T2"] + rD.lv - rD.lb, rD)
        res["A_T2_shift_fixedlaw"].append(sA); res["A_T2_shift_refit"].append(e["A_T2_shift"])
        res["D_T2_fixedc"].append(dF); res["D_T2_joint"].append(e["D_T2_strict_err"]); res["lnc"].append(e["lnc"])
        for p in PAIRS:
            res[f"dlevel_{p}"].append(e["pairs"][p]["dlevel"]); res[f"g_{p}"].append(e["pairs"][p]["g"])
        if with_pooled:
            th, s, ok = pooled([Rb[t] for t in TAGS], x0=pooled0)
            res["e_v"].append(th[-2]); res["e_b"].append(th[-1])
            fails += int(not ok)
        if (b + 1) % 250 == 0:
            print(f"    canonical {b + 1}/{B} ({time.time() - T0:.0f} s)", flush=True)
    return {k: np.array(v) for k, v in res.items() if len(v)}, fails


# ------------------------------------------------------------------------------------------------ U4 torsion
def torsion(base_draws, base_est):
    out = dict(formula="Z_tot = Z0 Z_R / (Z0 + Z_R) [Schoonderwaldt 2008 Sec. 1.1; Mansour 2016 eq. (2)], "
                       "Z_R = K_R c_R / r^2 = sqrt(K_R I_R) / r^2 [Mansour 2016 Sec. 2.2]")
    # absolute interpretations of POMA 'Zto' (printed unit g m^2/s^2), I_R = mu r^2 / 2 (uniform solid cylinder)
    interp = {}
    for lab, conv in (("K_R = Zto x 1e-3 N m^2 (g->kg, unit missing one m)", lambda z, r, I: np.sqrt(z * 1e-3 * I) / r ** 2),
                      ("K_R = Zto N m^2 (number taken as N m^2)", lambda z, r, I: np.sqrt(z * I) / r ** 2),
                      ("Zto = rotational impedance sqrt(K_R I_R) in g m^2/s (s^2 a typo)", lambda z, r, I: z * 1e-3 / r ** 2)):
        rat = {}
        for g in ("A T1", "B T1", "C T1", "D T1", "A T2", "B T2", "C T2", "D T2"):
            d, T, mu, Z0, Zto, EI = POMA[g]
            r = d * 1e-3 / 2; I = mu * 1e-3 * r ** 2 / 2
            rat[g] = float(conv(Zto, r, I) / Z0)
        interp[lab] = dict(Z_R_over_Z0=rat, T1_in_literature_range_1_to_4=all(1 <= rat[g] <= 4 for g in rat if g.endswith("T1")))
    lit = dict(vanWalstijn2026_cello_G=dict(K_R=2.9e-4, I_R=4.2e-10, r=0.487e-3),
               Matusiak2024_cello_G=dict(K_R=3.0312e-4, I_R=4.2e-10, r=0.5e-3))
    for k, v in lit.items():
        v["Z_R"] = float(np.sqrt(v["K_R"] * v["I_R"]) / v["r"] ** 2)
    out["absolute_interpretations"] = interp
    out["literature_cello_G_for_scale"] = lit
    out["defensibility"] = dict(
        D1_formula_in_local_text=True,
        D2_Zto_to_K_R_without_unit_assumption=False,
        D2_reason="printed unit g m^2/s^2 is not a torsional stiffness unit (N m^2 = kg m^3/s^2) nor an impedance unit; "
                  "manufacturer value without definition or SD",
        D3_I_R_available=False,
        D3_reason="POMA gives no polar moment of inertia; wound strings have a 'complicated distribution of stiffness and "
                  "mass' (Mansour 2016), so mu r^2/2 is an uncited approximation",
        absolute_Z_tot_defensible=False)
    # bracketed ratio sensitivity (pre-declared): Z_R(T1) = k Z0(T1); Z_R(T2)/Z_R(T1) = sqrt(Zto2 mu2/(Zto1 mu1)) d1/d2
    br = {}
    for kfac in (1.25, 2.0, 4.0):
        row = {}
        for p in PAIRS:
            d1, _, mu1, Z1, Zt1, _ = POMA[f"{p} T1"]; d2, _, mu2, Z2, Zt2, _ = POMA[f"{p} T2"]
            ZR1 = kfac * Z1
            ZR2 = ZR1 * np.sqrt(Zt2 * mu2 / (Zt1 * mu1)) * d1 / d2
            tot = lambda z, zr: z * zr / (z + zr)
            dln = float(np.log(tot(Z2, ZR2) / tot(Z1, ZR1)))
            ident = abs(dln) >= 0.02
            gd = base_draws[f"dlevel_{p}"] / dln
            row[p] = dict(dlnZ0=DLNZ[p], Z_R_ratio_T2_T1=float(ZR2 / ZR1), dlnZtot=dln, identified=ident,
                          g_tot=(base_est["pairs"][p]["dlevel"] / dln) if ident else None,
                          g_tot_ci95=pct(gd) if ident else None, g_tot_se=sd(gd) if ident else None)
        br[f"k={kfac}"] = row
    out["bracket_ratio_sensitivity"] = br
    return out


# ------------------------------------------------------------------------------------------------ U5 random effects
def meta(g, se):
    g, se = np.asarray(g, float), np.asarray(se, float)
    k = len(g); v = se ** 2
    w = 1 / v
    gf = np.sum(w * g) / np.sum(w)
    Q = float(np.sum(w * (g - gf) ** 2))
    tau_dl = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    def Qt(t2):
        ww = 1 / (v + t2); m = np.sum(ww * g) / np.sum(ww)
        return np.sum(ww * (g - m) ** 2) - (k - 1)
    if Qt(0.0) <= 0:
        tau_pm = 0.0
    else:
        lo, hi = 0.0, 1.0
        while Qt(hi) > 0:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if Qt(mid) > 0 else (lo, mid)
        tau_pm = 0.5 * (lo + hi)
    out = dict(k=k, fixed=dict(g=float(gf), se=float(np.sqrt(1 / np.sum(w)))), Q=Q, df=k - 1,
               p_Q=float(1 - __import__("scipy.stats", fromlist=["chi2"]).chi2.cdf(Q, k - 1)),
               chi2_g1=float(np.sum(w * (g - 1) ** 2)), p_g1=float(1 - __import__("scipy.stats", fromlist=["chi2"]).chi2.cdf(np.sum(w * (g - 1) ** 2), k)))
    tq = tdist.ppf(0.975, k - 1)
    for lab, t2 in (("PM", tau_pm), ("DL", tau_dl)):
        ww = 1 / (v + t2); m = float(np.sum(ww * g) / np.sum(ww))
        se_re = float(np.sqrt(1 / np.sum(ww)))
        q = float(np.sum(ww * (g - m) ** 2) / (k - 1))
        se_hk = float(np.sqrt(q / np.sum(ww))); se_hk_tr = float(np.sqrt(max(1.0, q) / np.sum(ww)))
        out[lab] = dict(tau=float(np.sqrt(t2)), g=m, se_normal=se_re, ci95_normal=[m - 1.96 * se_re, m + 1.96 * se_re],
                        hksj_q=q, se_hksj=se_hk, ci95_hksj=[m - tq * se_hk, m + tq * se_hk],
                        se_hksj_trunc=se_hk_tr, ci95_hksj_trunc=[m - tq * se_hk_tr, m + tq * se_hk_tr], t_crit=float(tq))
    return out


# ------------------------------------------------------------------------------------------------ main
def summarize_variant(name, R, est, can, cfails, dboot, dfails, pooled_th):
    rep = dict(denominators={t: dict(n_rows=R[t].n, n_clusters=len(R[t].members), census=census(R[t])) for t in TAGS},
               n_rows_total=int(sum(R[t].n for t in TAGS)), n_diagrams=len(TAGS),
               reference_law=dict(zip(("c", "a_v", "b_beta"), est["law"])), lnc_T1_strict=est["lnc"],
               levels=est["levels"], fit_ok=est["ok"])
    rep["pooled_shape"] = dict(e_v=float(pooled_th[-2]), e_beta=float(pooled_th[-1]),
                               diag_boot=dict(B=NB_DIAG, seed=SEED_DIAG, unit="diagram (11 drawn with replacement)", fails=dfails,
                                              e_v_ci95=pct(dboot[:, 0]), e_beta_ci95=pct(dboot[:, 1]),
                                              e_v_se=sd(dboot[:, 0]), e_beta_se=sd(dboot[:, 1])))
    cb = dict(B=NB_CAN, seed=SEED_CAN, unit="beta-position cluster within diagram (paired across speeds), stratified by diagram",
              fails=cfails)
    if "e_v" in can:
        rep["pooled_shape"]["canonical"] = dict(e_v_ci95=pct(can["e_v"]), e_beta_ci95=pct(can["e_b"]), e_v_se=sd(can["e_v"]),
                                                e_beta_se=sd(can["e_b"]), **cb)
    rep["pairs"] = {p: dict(dlnZ=DLNZ[p], dlevel=est["pairs"][p]["dlevel"], dlevel_ci95=pct(can[f"dlevel_{p}"]),
                            dlevel_se=sd(can[f"dlevel_{p}"]), g=est["pairs"][p]["g"], g_ci95=pct(can[f"g_{p}"]),
                            g_se=sd(can[f"g_{p}"]), one_inside=bool(pct(can[f"g_{p}"])[0] <= 1 <= pct(can[f"g_{p}"])[1]))
                    for p in PAIRS}
    rep["D_T2_strict"] = dict(err=est["D_T2_strict_err"], within_0p12=bool(abs(est["D_T2_strict_err"]) < 0.12),
                              ci95_fixed_c=pct(can["D_T2_fixedc"]), ci95_joint=pct(can["D_T2_joint"]),
                              se_joint=sd(can["D_T2_joint"]))
    rep["A_T2"] = dict(shift=est["A_T2_shift"], g=est["A_T2_g"],
                       shift_ci95_fixed_law=pct(can["A_T2_shift_fixedlaw"]), g_ci95_fixed_law=pct(can["A_T2_shift_fixedlaw"] / DLNZ["A"]),
                       shift_ci95_refit=pct(can["A_T2_shift_refit"]), g_ci95_refit=pct(can["A_T2_shift_refit"] / DLNZ["A"]),
                       g_se_fixed_law=sd(can["A_T2_shift_fixedlaw"] / DLNZ["A"]))
    rep["canonical"] = cb
    print(f"\n== {name}: rows {rep['n_rows_total']} | law c {est['law'][0]:+.3f} a {est['law'][1]:+.3f} b {est['law'][2]:+.3f} | ln c {est['lnc']:+.4f}")
    ps = rep["pooled_shape"]
    print(f"   pooled e_v {ps['e_v']:+.3f} diag{np.round(ps['diag_boot']['e_v_ci95'], 3)} | e_b {ps['e_beta']:+.3f} diag{np.round(ps['diag_boot']['e_beta_ci95'], 3)}"
          + (f" | canonical e_v {np.round(ps['canonical']['e_v_ci95'], 3)} e_b {np.round(ps['canonical']['e_beta_ci95'], 3)}" if "canonical" in ps else ""))
    for p, v in rep["pairs"].items():
        print(f"   pair {p}: dlevel {v['dlevel']:+.3f} {np.round(v['dlevel_ci95'], 3)} -> g {v['g']:+.2f} {np.round(v['g_ci95'], 2)} (SE {v['g_se']:.2f}; 1 inside {v['one_inside']})")
    d = rep["D_T2_strict"]
    print(f"   D_T2 strict err {d['err']:+.4f} fixed-c {np.round(d['ci95_fixed_c'], 3)} joint {np.round(d['ci95_joint'], 3)}")
    a = rep["A_T2"]
    print(f"   A_T2 shift {a['shift']:+.3f} fixed-law {np.round(a['shift_ci95_fixed_law'], 3)} (g {a['g']:+.2f} {np.round(a['g_ci95_fixed_law'], 2)}) | refit {np.round(a['shift_ci95_refit'], 3)} (g {np.round(a['g_ci95_refit'], 2)})")
    print(f"   fails: canonical {cfails}/{NB_CAN}, diagram boot {dfails}/{NB_DIAG}", flush=True)
    return rep


def originals():
    up = json.loads((RNE / "novelty/upper_law/upper_law_t0.2.json").read_text())
    lv = json.loads((RNE / "novelty/upper_final/levels.json").read_text())
    hz = json.loads((RNE / "novelty/upper_final/hier_z.json").read_text())
    dt = json.loads((RNE / "astra_20261019/D_T2_confirm_result.json").read_text())
    dpair = hz["pairs"]["E"]["noise_mom"]["per_type"]["D"]
    return dict(
        pooled_shape=dict(e_v=lv["pooled_shape"]["e_v"], e_v_ci95=lv["pooled_shape"]["e_v_ci95"], e_beta=lv["pooled_shape"]["e_beta"],
                          e_beta_ci95=lv["pooled_shape"]["e_beta_ci95"], source="novelty/upper_final/levels.json (200 unstratified beta-cluster)"),
        pairs={**{p: dict(g=up["pairs"][p]["z_exp"], g_ci95=up["pairs"][p]["z_exp_ci"], source="upper_law_t0.2.json (300 unpaired row resamples)")
                  for p in ("A", "B", "C")},
               "D": dict(g=dpair["gamma"], g_ci95=[dpair["gamma"] - 1.96 * dpair["se_gamma"], dpair["gamma"] + 1.96 * dpair["se_gamma"]],
                         source="hier_z.json pairs.E.noise_mom (normal, incl. session-noise MoM)")},
        A_T2=dict(shift=0.197, shift_ci95=[0.106, 0.286], g=1.71, g_ci95=[0.92, 2.49], source="final/confirm_A_T2.log (frozen, 200 unpaired row resamples)"),
        D_T2=dict(err=dt["level_fits"]["upper_strict"]["level"], ci95=dt["bootstrap"]["upper_strict"]["percentile_95_conditional_successes"],
                  source="astra_20261019/D_T2_confirm_result.json (sealed)"))


def main():
    global T0
    T0 = time.time()
    data, raw = build_variants()
    res = dict(status="POST HOC on 11 development diagrams (A_T2, D_T2 already scored once = seen)", t_eval=T_EVAL,
               inputs={t: str(PATHS[t]) for t in TAGS}, check_K2_vs_compare=check_against_compare(data),
               originals=originals(), variants={}, secondary_frozen_law={})
    print("check K=2 intervals vs final/compare.intervals:", res["check_K2_vs_compare"], flush=True)
    draws = {}
    for vi, (name, R) in enumerate(raw.items()):
        est = estimates(R)
        th, s, ok = pooled([R[t] for t in TAGS])
        rngd = np.random.default_rng(SEED_DIAG)
        dboot, dfails = diag_boot_pooled(R, th, rngd, NB_DIAG)
        print(f"  {name}: diagram bootstrap done ({time.time() - T0:.0f} s)", flush=True)
        rngc = np.random.default_rng(SEED_CAN)
        can, cfails = canonical(R, est, rngc, NB_CAN, with_pooled=(name == "base"), pooled0=th)
        draws[name] = can
        res["variants"][name] = summarize_variant(name, R, est, can, cfails, dboot, dfails, th)
        res["variants"][name]["pooled_shape"]["fit_ok"] = ok
        if name != "base":  # secondary: frozen BASE_A law instead of the refitted one (point estimates only)
            e2 = estimates(R, fixed_law=BASE_A)
            res["secondary_frozen_law"][name] = dict(pairs={p: e2["pairs"][p] for p in PAIRS}, A_T2_shift=e2["A_T2_shift"],
                                                     A_T2_g=e2["A_T2_g"])
            print(f"   [{name}, frozen BASE_A law] " + " ".join(f"{p} g {e2['pairs'][p]['g']:+.2f}" for p in PAIRS) +
                  f" | A_T2 shift {e2['A_T2_shift']:+.3f}", flush=True)

    # U4
    res["U4_torsion"] = torsion(draws["base"], res["variants"]["base"])
    print("\nU4 torsion: absolute interpretations (Z_R/Z0, T1 strings):")
    for k, v in res["U4_torsion"]["absolute_interpretations"].items():
        print(f"   {k}: " + " ".join(f"{g} {x:.2f}" for g, x in v["Z_R_over_Z0"].items()) + f" | T1 in 1-4: {v['T1_in_literature_range_1_to_4']}")
    for k, row in res["U4_torsion"]["bracket_ratio_sensitivity"].items():
        print(f"   bracket {k}: " + " | ".join(
            f"{p} dlnZ0 {v['dlnZ0']:+.3f} dlnZtot {v['dlnZtot']:+.4f} " + (f"g_tot {v['g_tot']:+.1f} {np.round(v['g_tot_ci95'], 1)}" if v["identified"] else "not identified")
            for p, v in row.items()))

    # U5
    b = res["variants"]["base"]["pairs"]
    res["U5_random_effects"] = {}
    for lab, ps in (("A,B,C", ["A", "B", "C"]), ("A,B,C,D", ["A", "B", "C", "D"])):
        m = meta([b[p]["g"] for p in ps], [b[p]["g_se"] for p in ps])
        m["inputs"] = {p: dict(g=b[p]["g"], se=b[p]["g_se"]) for p in ps}
        m["one_in_PM_hksj"] = bool(m["PM"]["ci95_hksj"][0] <= 1 <= m["PM"]["ci95_hksj"][1])
        m["PM_hksj_width"] = m["PM"]["ci95_hksj"][1] - m["PM"]["ci95_hksj"][0]
        res["U5_random_effects"][lab] = m
        print(f"\nU5 {lab}: fixed {m['fixed']['g']:+.2f} ± {m['fixed']['se']:.2f}; Q {m['Q']:.1f}/{m['df']} (p {m['p_Q']:.4f}); chi2(g=1) {m['chi2_g1']:.1f}/{m['k']} (p {m['p_g1']:.4f})")
        for e in ("PM", "DL"):
            q = m[e]
            print(f"   {e}: tau {q['tau']:.2f} g {q['g']:+.2f} normal {np.round(q['ci95_normal'], 2)} HKSJ {np.round(q['ci95_hksj'], 2)} (q {q['hksj_q']:.2f}) HKSJ-trunc {np.round(q['ci95_hksj_trunc'], 2)}")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(HERE / "canonical_draws.npz", **{f"{v}__{k}": a for v, d in draws.items() for k, a in d.items()})
    print(f"\nwrote results.json, canonical_draws.npz [{time.time() - T0:.0f} s]")


T0 = time.time()
if __name__ == "__main__":
    main()
