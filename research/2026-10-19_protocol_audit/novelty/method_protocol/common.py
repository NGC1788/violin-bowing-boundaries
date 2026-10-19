"""Protocol-aware estimator of the minimum-force (lower) boundary -- shared code.
Claude, 2026-10-19, novelty/method_protocol (post hoc development on the 11 EXPOSED diagrams).

Nothing here downloads data, touches running processes or writes outside this folder.
Label features (amplitude_std, periodicity, f0_hz, flybacks_per_period) are read ONLY for the
EXPOSED tags below.  For unseen archives use `controls_only()` (control columns only).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.csv as pc
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import spearmanr, theilslopes

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                      # ~/violin-rne
FEAT = BASE / "timeres/features"
CACHE = HERE / "cache"

# ----------------------------------------------------------------------------------------- inventory
EXPOSED = {  # tag -> (feature stem, replicate sub-folder or None)
    "A1": ("2024-03-25_TypeA_sample1", None), "A2": ("2024-04-19_TypeA_sample2", None),
    "B1": ("2024-04-10_TypeB_sample1", None), "C1": ("2024-04-12_TypeC_sample1", None),
    "D1": ("2024-04-15_TypeD_sample1", None), "A_T2": ("2024-06-19_TypeA_T2_sample1", None),
    "B_T2s1": ("2024-06-13_TypeB_T2_sample1", None),
    "B_T2r1": ("2024-10-23_variability_setup", "rep_1"), "B_T2r2": ("2024-10-23_variability_setup", "rep_2"),
    "C_T2": ("2024-06-11_TypeC_T2_sample1", None), "D_T2": ("2024-06-17_TypeD_T2_sample1", None)}
TAGS = list(EXPOSED)
# Published string properties (Lampis et al. POMA 58 035013 (2025) Table 1 as transcribed in
# novelty/lower_protocol/analysis.py; D_T2 from final/CONFIRM_PLAN_D_T2.md): (Z kg/s, EI 1e-4 N m^2)
PROPS = {"A_T1": (1.0592, 3.032), "B_T1": (1.0998, 4.397), "C_T1": (1.0525, 3.142), "D_T1": (1.0398, 5.246),
         "A_T2": (1.1884, 3.969), "B_T2": (1.1899, 7.295), "C_T2": (1.1154, 3.969), "D_T2": (1.2021, 5.766)}
GROUP = {"A1": "A_T1", "A2": "A_T1", "B1": "B_T1", "C1": "C_T1", "D1": "D_T1", "A_T2": "A_T2",
         "B_T2s1": "B_T2", "B_T2r1": "B_T2", "B_T2r2": "B_T2", "C_T2": "C_T2", "D_T2": "D_T2"}
# Future (UNSEEN) archives: properties ASSUMED equal to the published type x tension row (POMA lists
# type x tension, not specimen); this is an assumption, not a measurement of the new specimen.
FUTURE = {"B_T2s2": ("2024-07-03_TypeB_T2_sample2", "B_T2"), "A3": ("2024-04-22_TypeA_sample3", "A_T1"),
          "A4": ("2024-04-24_TypeA_sample4", "A_T1"), "D2": ("2024-04-26_TypeD_sample2", "D_T1")}

CTRL_FIELDS = ["folder", "trial", "beta", "window_s", "t", "force", "speed", "error"]
LABEL_FIELDS = ["amplitude_std", "periodicity", "f0_hz", "flybacks_per_period"]
FLOOR = 1e-3          # legacy clamp (N) used by all earlier boundary code
K = 2                 # neighbouring *different strokes* in force order
T_EVAL = 0.2          # window start of the primary label (0.2 s window)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read(stem, fields, path=None):
    path = Path(path) if path else FEAT / f"{stem}.csv.gz"
    t = pc.read_csv(path, convert_options=pc.ConvertOptions(include_columns=fields))
    return {k: t[k].to_numpy(zero_copy_only=False) for k in fields}


def _assemble(d, rep, path_hint):
    """Per-stroke control arrays (label free)."""
    err = np.array([x not in (None, "") for x in d["error"]])
    if err.any():
        raise ValueError(f"{path_hint}: {err.sum()} feature rows carry errors")
    keep = np.ones(len(err), bool)
    if rep:
        keep = np.array([f"/{rep}/" in f + "/" for f in d["folder"]])
    d = {k: v[keep] for k, v in d.items()}
    folders = sorted(set(d["folder"]))
    if len(folders) != 3:
        raise ValueError(f"{path_hint}: expected 3 speed folders, got {folders}")
    spd = {f: float(np.median(d["speed"][d["folder"] == f])) for f in folders}
    folders.sort(key=lambda f: spd[f])
    fi = np.array([folders.index(f) for f in d["folder"]])
    trial = d["trial"].astype(int)
    key = fi * 2001 + trial
    w = d["window_s"].astype(float)
    t = np.round(d["t"].astype(float), 2)
    nkey = 3 * 2001

    def pick(mask):
        out = np.full(nkey, np.nan)
        kk = key[mask]
        if len(np.unique(kk)) != len(kk):
            raise ValueError(f"{path_hint}: duplicate stroke rows")
        out[kk] = d["force"][mask]
        return out
    F_on = pick(np.isclose(w, .1) & (t == 0.0))
    F_01 = pick(np.isclose(w, .1) & (t == 0.1))
    F_ev = pick(np.isclose(w, .2) & (t == T_EVAL))
    # 5-cm window: first 0.1 s window with t>0 and window speed * t >= 0.05 m (Astra's dec5_pl rule)
    m5 = np.isclose(w, .1) & (t > 0) & (d["speed"] * t >= .05)
    order = np.lexsort((t[m5], key[m5]))
    k5 = key[m5][order]
    first = np.r_[True, k5[1:] != k5[:-1]]
    F_5 = np.full(nkey, np.nan)
    F_5[k5[first]] = d["force"][m5][order][first]
    beta = np.full(nkey, np.nan)
    m0 = np.isclose(w, .2) & (t == T_EVAL)
    beta[key[m0]] = d["beta"][m0]
    strokes = np.array([f * 2001 + tr for f in range(3) for tr in range(1, 2001)])
    out = dict(folder=strokes // 2001, trial=strokes % 2001, F_on=F_on[strokes], F_01=F_01[strokes],
               F_ev=F_ev[strokes], F_5=F_5[strokes], beta=beta[strokes],
               speed=np.array([spd[folders[f]] for f in strokes // 2001]))
    for k in ("F_on", "F_01", "F_ev", "F_5", "beta"):
        if not np.isfinite(out[k]).all():
            raise ValueError(f"{path_hint}: missing {k} for {np.sum(~np.isfinite(out[k]))} strokes")
    # beta columns: within each folder sort by beta and cut into 40 groups of 50 (verify_astra_boundaries.columns)
    col = np.empty(len(strokes), int)
    for f in range(3):
        ii = np.flatnonzero(out["folder"] == f)
        o = ii[np.argsort(out["beta"][ii], kind="stable")]
        col[o] = np.arange(len(o)) // 50
        lb = np.log(out["beta"][o]).reshape(40, 50)
        if not np.all(lb[1:, 0] > lb[:-1, -1]):
            raise ValueError(f"{path_hint}: beta columns overlap")
    out["column"] = col
    # nominal rank; reversal detected label-free from onset force vs trial order (A1 is reversed)
    r = (out["trial"] - 1) % 50 + 1
    rho = [spearmanr(r[out["folder"] == f], out["F_on"][out["folder"] == f])[0] for f in range(3)]
    reversed_ = bool(np.median(rho) < 0)
    out["rank"] = 51 - r if reversed_ else r
    out["F_nom"] = .1 * 40.0 ** ((out["rank"] - 1) / 49)
    out["meta"] = dict(folders=folders, speed=[spd[f] for f in folders], rank_rho=[float(x) for x in rho],
                       rank_reversed=reversed_)
    return out, (fi, trial, w, t, key, d)


def controls_only(stem, rep=None, path=None):
    """Label-free per-stroke controls for any archive (reads ONLY control columns)."""
    d = _read(stem, CTRL_FIELDS, path)
    out, _ = _assemble(d, rep, str(path or stem))
    return out


def labels_after_claim(path, rep, claim_path):
    """H labels (t=0.2 s, 0.2-s window) of a NON-exposed target; refuses unless the single-use score claim exists."""
    if not Path(claim_path).exists():
        raise PermissionError("score claim missing: labels of an unseen target may only be read after the claim")
    sys.path.insert(0, str(BASE / "claude_extra"))
    import verify_astra_boundaries as V
    d = _read(None, CTRL_FIELDS + LABEL_FIELDS, path)
    out, (fi, trial, w, t, key, dd) = _assemble(d, rep, str(path))
    strokes = out["folder"] * 2001 + out["trial"]
    m = np.flatnonzero(np.isclose(w, .2) & (t == T_EVAL))
    lab = np.zeros(3 * 2001, bool)
    for i in m:
        fpp = dd["flybacks_per_period"][i]
        fpp = None if fpp is None or not np.isfinite(fpp) else int(fpp)
        lab[key[i]] = V.rm.classify(float(dd["amplitude_std"][i]), float(dd["periodicity"][i]), float(dd["f0_hz"][i]),
                                    fpp, 98.08, V.APPLIED) == "helmholtz"
    out["H2"] = lab[strokes]
    return out


def load_exposed(tag):
    """Controls + H labels (t = 0.1/0.2/0.3, 0.2-s window) for an EXPOSED tag; cached in ./cache."""
    if tag not in EXPOSED:
        raise PermissionError(f"{tag} is not an exposed diagram; labels may not be read")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
        return out
    stem, rep = EXPOSED[tag]
    sys.path.insert(0, str(BASE / "claude_extra"))
    import verify_astra_boundaries as V  # classifier + A2 thresholds (same call as final/data.py)
    d = _read(stem, CTRL_FIELDS + LABEL_FIELDS)
    out, (fi, trial, w, t, key, dd) = _assemble(d, rep, stem)
    strokes = out["folder"] * 2001 + out["trial"]
    for tt in (0.1, 0.2, 0.3):
        m = np.flatnonzero(np.isclose(w, .2) & (t == tt))
        lab = np.zeros(3 * 2001, bool)
        for i in m:
            fpp = dd["flybacks_per_period"][i]
            fpp = None if fpp is None or not np.isfinite(fpp) else int(fpp)
            lab[key[i]] = V.rm.classify(float(dd["amplitude_std"][i]), float(dd["periodicity"][i]), float(dd["f0_hz"][i]),
                                        fpp, 98.08, V.APPLIED) == "helmholtz"
        out[f"H{int(round(tt * 10))}"] = lab[strokes]
    meta = json.dumps(out.pop("meta"))
    np.savez_compressed(path, meta=np.array(meta), **out)
    return load_exposed(tag)


# ----------------------------------------------------------------------------------------- boundaries
def intervals(d, axis="F_ev", H="H2", fix="none", delta=0.01):
    """Lower-boundary interval per (folder, column) on log force, legacy K=2 single-band rule.

    axis : per-stroke force used for ordering and for the interval ends (label free control)
    fix  : 'none'  legacy  log(max(F,1e-3)) ends (zero-width intervals possible at the clamp)
           'drop'  legacy ends, zero-width/reversed columns removed
           'floor' ends at/below the 1 mN clamp become -inf (left-censored at the floor)
           'delta' measurement-resolution widening in force units: [log(F_lo-delta), log(F_hi+delta)],
                   F_lo-delta <= 1 mN -> -inf.   (POST HOC fix of the D_T2 failure mode)
    """
    rows = []
    F = d[axis]
    for f in range(3):
        for c in range(40):
            ii = np.flatnonzero((d["folder"] == f) & (d["column"] == c))
            if len(ii) != 50:
                raise ValueError("incomplete force grid")
            ii = ii[np.argsort(F[ii], kind="stable")]
            Fr = F[ii]
            h = d[H][ii].astype(int) if H else None
            if h is not None:
                runs = np.flatnonzero(np.convolve(h, np.ones(K, int), "valid") == K)
                if len(runs):
                    s = int(runs[0]); flo = -np.inf if s == 0 else Fr[s - 1]; fhi = Fr[s]
                else:
                    flo, fhi = Fr[-1], np.inf
                band = len(runs) > 0
            else:
                flo = fhi = np.nan; band = None
            lo, hi = _ends(flo, fhi, fix, delta)
            rows.append(dict(folder=f, column=c, lv=float(np.log(np.median(d["speed"][ii]))),
                             lb=float(np.log(np.median(d["beta"][ii]))), lo=lo, hi=hi, band=band,
                             s=(int(runs[0]) if h is not None and len(runs) else None),
                             floor1=float(np.log(max(Fr[1], FLOOR))), strokes=ii))
    if fix == "drop":
        rows = [r for r in rows if r["lo"] < r["hi"]]
    return rows


def _ends(flo, fhi, fix, delta):
    lg = lambda x: np.log(max(x, FLOOR))
    if np.isnan(flo) and np.isnan(fhi):
        return np.nan, np.nan
    if fix in ("none", "drop"):
        return (-np.inf if flo == -np.inf else lg(flo)), (np.inf if fhi == np.inf else lg(fhi))
    if fix == "floor":
        lo = -np.inf if (flo == -np.inf or flo <= FLOOR) else np.log(flo)
        hi = np.inf if fhi == np.inf else lg(fhi)
        return lo, hi
    if fix == "delta":
        lo = -np.inf if (flo == -np.inf or flo - delta <= FLOOR) else np.log(flo - delta)
        hi = np.inf if fhi == np.inf else np.log(max(fhi, 0.0) + delta)
        return lo, hi
    raise ValueError(fix)


# ----------------------------------------------------------------------------------------- likelihood
def log_int_prob(lo, hi, mu, s):
    """log P(lo < X <= hi), X ~ N(mu, s^2); stable in both tails. Zero width -> -inf (never clipped)."""
    zl = (lo - mu) / s; zh = (hi - mu) / s
    flip = zl > 0
    a = np.where(flip, log_ndtr(-zl), log_ndtr(zh))
    b = np.where(flip, log_ndtr(-zh), log_ndtr(zl))
    with np.errstate(divide="ignore", invalid="ignore"):
        return a + np.log1p(-np.exp(b - a))


def _grad_terms(lo, hi, mu, s):
    """d logP / d mu and d logP / d log s."""
    lp = log_int_prob(lo, hi, mu, s)
    zl = (lo - mu) / s; zh = (hi - mu) / s
    def lphi(z):
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(np.isfinite(z), -0.5 * z * z - 0.9189385332046727, -np.inf)
    el = np.exp(lphi(zl) - lp); eh = np.exp(lphi(zh) - lp)
    gm = (el - eh) / s
    zl_ = np.where(np.isfinite(zl), zl, 0.0); zh_ = np.where(np.isfinite(zh), zh, 0.0)
    gs = zl_ * el - zh_ * eh
    return lp, gm, gs


def check_rows(rows):
    lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    ok = ~(np.isnan(lo) & np.isnan(hi))
    bad = ok & ~(lo < hi)
    return lo, hi, ok, bad


def fit_offset(rows, mu0, sigma=None):
    """One common offset delta (and sigma unless given) for target intervals around column predictions mu0.
    Same statistic as the D_T2 primary lower fit; multi-start L-BFGS; fails loudly on zero-width input."""
    lo, hi, ok, bad = check_rows(rows)
    if bad.any():
        return dict(success=False, reason="invalid_or_zero_width_interval", n_bad=int(bad.sum()))
    lo, hi, mu0 = lo[ok], hi[ok], np.asarray(mu0, float)[ok]
    if not (np.isfinite(lo) & np.isfinite(hi)).any() and (np.isfinite(lo).all() or np.isfinite(hi).all()):
        return dict(success=False, reason="one_sided_only")
    def f(p):
        s = np.exp(p[1]) if sigma is None else sigma
        lp, gm, gs = _grad_terms(lo, hi, mu0 + p[0], s)
        if not np.isfinite(lp).all():
            return 1e100, np.zeros(2)
        return -lp.sum(), -np.array([gm.sum(), gs.sum() if sigma is None else 0.0])
    best = None
    for s0 in (.15, .5, 1.5):
        r = minimize(f, [0.0, np.log(s0)], jac=True, method="L-BFGS-B", bounds=[(-5, 5), (-6, 3)])
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None or best.fun >= 1e90:
        return dict(success=False, reason="optimizer_failed")
    return dict(success=True, level=float(best.x[0]), sigma=float(np.exp(best.x[1]) if sigma is None else sigma),
                nll=float(best.fun), sigma_bound=bool(sigma is None and best.x[1] <= -5.99), n=int(ok.sum()))


def fit_sessions(rows_by_tag, shape=None):
    """Stage 1: interval-censored ML with one intercept per (diagram, speed-session) + common beta slope + sigma.
    rows_by_tag: {tag: rows}.  Returns alpha[(tag, folder)], b, sigma and per-session curvature SE."""
    keys = sorted({(t, r["folder"]) for t, rs in rows_by_tag.items() for r in rs})
    kidx = {k: i for i, k in enumerate(keys)}
    lo, hi, ok, bad = check_rows([r for rs in rows_by_tag.values() for r in rs])
    if bad.any():
        raise ValueError(f"{bad.sum()} zero-width intervals; use a fix")
    allr = [r for rs in rows_by_tag.values() for r in rs]
    sid = np.array([kidx[(t, r["folder"])] for t, rs in rows_by_tag.items() for r in rs])
    shape = shape or (lambda r: np.array([r["lb"]]))
    Phi = np.array([shape(r) for r in allr])
    lo, hi, sid, Phi = lo[ok], hi[ok], sid[ok], Phi[ok]
    n, q = len(keys), Phi.shape[1]
    proxy = np.where(np.isfinite(lo), np.where(np.isfinite(hi), .5 * (lo + hi), lo), hi)
    a0 = np.array([np.median(proxy[sid == i]) for i in range(n)])
    def f(p):
        a, th, ls = p[:n], p[n:n + q], p[n + q]
        lp, gm, gs = _grad_terms(lo, hi, a[sid] + Phi @ th, np.exp(ls))
        if not np.isfinite(lp).all():
            return 1e100, np.zeros(n + q + 1)
        g = np.r_[np.bincount(sid, gm, n), gm @ Phi, gs.sum()]
        return -lp.sum(), -g
    best = None
    for b0 in (-1.0, -0.5):
        th0 = np.zeros(q); th0[0] = b0
        r = minimize(f, np.r_[a0 - Phi.mean(0) @ th0, th0, np.log(.4)], jac=True, method="L-BFGS-B",
                     options=dict(maxiter=8000, ftol=1e-13, gtol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    a, th, s = best.x[:n], best.x[n:n + q], np.exp(best.x[n + q])
    # per-session SE from the 1-D curvature (shape, sigma fixed)
    se = np.empty(n)
    for i in range(n):
        m = sid == i
        h = 1e-4
        fn = lambda x: -log_int_prob(lo[m], hi[m], x + Phi[m] @ th, s).sum()
        c2 = (fn(a[i] + h) - 2 * fn(a[i]) + fn(a[i] - h)) / h ** 2
        se[i] = 1 / np.sqrt(c2) if c2 > 0 else np.inf
    return dict(keys=keys, alpha=a, b=float(th[0]), theta=th, sigma=float(s), se=se, success=bool(best.success),
                nll=float(best.fun))


def reml_fit(y, X, groups, se):
    """Stage 2: y_ds = X beta + u_d + w_ds + e_ds, u~N(0,tau^2) by diagram, w~N(0,omega^2), e~N(0,se^2) known.
    REML over (log tau, log omega); returns beta, cov(beta), tau, omega."""
    y = np.asarray(y, float); X = np.asarray(X, float); se = np.asarray(se, float)
    G = np.array([g for g in groups]); ug = sorted(set(G))
    def parts(lt, lw):
        tau2, om2 = np.exp(2 * lt), np.exp(2 * lw)
        V = np.diag(om2 + se ** 2) + tau2 * (G[:, None] == G[None, :])
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
        r = y - X @ beta
        return V, Vi, XtViX, beta, r
    def nreml(p):
        V, Vi, XtViX, beta, r = parts(*p)
        s1 = np.linalg.slogdet(V)[1]; s2 = np.linalg.slogdet(XtViX)[1]
        return 0.5 * (s1 + s2 + r @ Vi @ r)
    best = None
    for st in ([np.log(.1), np.log(.1)], [np.log(.3), np.log(.03)], [np.log(.03), np.log(.3)]):
        r = minimize(nreml, st, method="L-BFGS-B", bounds=[(-7, 1), (-7, 1)])
        if best is None or r.fun < best.fun:
            best = r
    V, Vi, XtViX, beta, res = parts(*best.x)
    return dict(beta=beta, cov=np.linalg.inv(XtViX), tau=float(np.exp(best.x[0])), omega=float(np.exp(best.x[1])),
                success=bool(best.success), tau_at_bound=bool(best.x[0] <= -6.99), omega_at_bound=bool(best.x[1] <= -6.99))


# ----------------------------------------------------------------------------------------- covariates
def session_covariates(d):
    """Label-free session (speed-folder) covariates from low nominal ranks 1-20 (800 strokes per folder)."""
    out = {}
    for f in range(3):
        m = (d["folder"] == f) & (d["rank"] <= 20)
        out[f] = dict(dec5=float(np.median(d["F_5"][m] - d["F_on"][m])),            # N, Astra's dec5_pl per folder
                      dec_ev=float(np.median(d["F_ev"][m] - d["F_on"][m])),         # N, onset -> evaluation window
                      cal=float(np.median(np.log(np.maximum(d["F_on"][m], FLOOR) / d["F_nom"][m]))),  # onset/nominal
                      n=int(m.sum()))
    out["diagram"] = {k: float(np.mean([out[f][k] for f in range(3)])) for k in ("dec5", "dec_ev", "cal")}
    return out


# ----------------------------------------------------------------------------------------- transport
def transport_maps(d, src, dst="F_ev"):
    """Label-free per-(folder,column) map from a source force axis to the measured evaluation axis.
    Model in force units: F_dst = F_src - (p + q F_src), Theil-Sen over the column's 50 strokes,
    restricted to the lower half of the column's source forces (where lower boundaries lie)."""
    maps = {}
    for f in range(3):
        for c in range(40):
            ii = np.flatnonzero((d["folder"] == f) & (d["column"] == c))
            x = d[src][ii]; y = d[dst][ii]
            o = np.argsort(x)[:30]
            q, p, *_ = theilslopes(x[o] - y[o], x[o])
            if not (1 - q) > 0.2:   # degenerate -> pure offset
                q = 0.0; p = float(np.median(x[o] - y[o]))
            maps[(f, c)] = (float(p), float(q))
    return maps


def fwd(lx, pq):
    """log F_src -> log F_dst."""
    p, q = pq
    return np.log(np.maximum(np.exp(lx) * (1 - q) - p, FLOOR))


def inv(ly, pq):
    """log F_dst -> log F_src (interval ends; +-inf preserved)."""
    p, q = pq
    ly = np.asarray(ly, float)
    with np.errstate(over="ignore", invalid="ignore"):
        v = (np.exp(ly) + p) / (1 - q)
    out = np.where(np.isfinite(ly), np.log(np.maximum(v, 1e-9)), ly)
    return out
