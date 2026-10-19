"""Null models WITHOUT force-history hysteresis for the A2 v=0.05 m/s maintenance statistic.

All nulls keep every floor stroke's OBSERVED force trajectory (0.2 s windows every 0.05 s) and replace only the
labels.  The marginal P(H | log F_now, t, column) is a penalised logistic regression with B-spline terms in log F
and t, a log F x t interaction, column intercepts and column-specific log F slopes, fitted on the A2 v1 floor
windows (stroke-grouped CV picks the penalty).

Schemes
  indep        : independent Bernoulli(pi_t) windows (no persistence; reference only)
  copula(w,rho): latent z_t = sqrt(w) u_stroke + sqrt(1-w) e_t, e_t AR(1) with lag-1 correlation rho;
                 H_t = [z_t < Phi^-1(pi_t)].  Marginal exactly pi_t, persistence from u (stroke heterogeneity) and
                 e (temporal), no dependence on whether the stroke was H before at a higher force.
                 w = 1 is the 'settling-only' null (b): every stroke has a fixed threshold in pi-space and pi_t
                 depends on current force and exposure time only.
  markov       : two-state Markov chain; P(H_t | H_{t-1}=0, F_t, t, col) and P(H_t | H_{t-1}=1, F_t, t, col) fitted
                 by logistic regression on observed transitions, initial state from P(H_0 | F_0, col).
  hyst(delta)  : POSITIVE CONTROL with explicit hysteresis: copula(w=1) entry rule, but once H the stroke stays H
                 until z_t > Phi^-1(pi_t) + delta (maintenance threshold delta latent SDs more permissive).
Statistics (identical code for data and nulls)
  ratio   : entry_vs_low.py statistic, median over columns of q10(low)/q10(entry), K=2; also share of cols < 1
  lowent  : median per-stroke low/entry
  late_below_FM : share of late entries (t_entry >= 0.3 s, i.e. established during the plateau from a non-H state)
                  whose force is below the column's q10(low)  (literal contradiction of 'never established there')
  MT      : matched-time ratio, t >= 0.5 s: q10(maintained K-run-end forces) / q10(entry forces), median over cols
  up/down : per-window switching rates nonH->H, H->nonH
"""
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

from common import load, runs_of

K = 2
RNG = np.random.default_rng(20261018)


# ---------------------------------------------------------------- data
def prepare(tag="A2", folder=0, win=0.2):
    S, speed = load(tag, win)
    fl = [s for s in S if s.folder == folder and s.floor]
    return fl, speed[folder]


class Design:
    def __init__(self, fl, cols_onehot=True, slopes=True, nk_f=6, nk_t=5, use_t=True):
        self.use_t = use_t
        lf = np.concatenate([np.log(np.maximum(s.F, 1e-3)) for s in fl])
        tt = np.concatenate([s.t for s in fl])
        self.sf = SplineTransformer(n_knots=nk_f, degree=3, knots="quantile", extrapolation="linear").fit(lf[:, None])
        self.st = SplineTransformer(n_knots=nk_t, degree=3, knots="quantile", extrapolation="linear").fit(tt[:, None])
        self.cols = sorted({s.col for s in fl})
        self.lfm, self.tm = lf.mean(), tt.mean()
        self.cols_onehot, self.slopes = cols_onehot, slopes
        self.lbm = np.mean([np.log(s.beta) for s in fl])

    def X(self, lf, t, col, lbeta):
        parts = [self.sf.transform(lf[:, None])]
        if self.use_t:
            parts += [self.st.transform(t[:, None]), ((lf - self.lfm) * (t - self.tm))[:, None]]
        if self.cols_onehot:
            oh = (col[:, None] == np.array(self.cols)[None, :]).astype(float)
            parts.append(oh)
            if self.slopes:
                parts.append(oh * (lf - self.lfm)[:, None])
        else:
            lb = lbeta - self.lbm
            parts += [lb[:, None], lb[:, None] ** 2, (lb * (lf - self.lfm))[:, None]]
        return np.hstack(parts)


def stroke_X(D, s):
    lf = np.log(np.maximum(s.F, 1e-3))
    return D.X(lf, s.t, np.full(len(s.t), s.col), np.full(len(s.t), np.log(s.beta)))


class ConstModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def fit_logit(X, y, groups, Cs=(0.03, 0.3, 3.0, 30.0)):
    if y.min() == y.max() or min(np.sum(y == 1), np.sum(y == 0)) < 5 or len(np.unique(groups)) < 5:
        return ConstModel((np.sum(y) + 0.5) / (len(y) + 1.0)), None
    best = None
    for C in Cs:
        ll = 0.0
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
            p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            ll += -np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, C)
    return LogisticRegression(C=best[1], max_iter=5000).fit(X, y), best[1]


def fit_marginal(fl, D):
    X = np.vstack([stroke_X(D, s) for s in fl])
    y = np.concatenate([s.H.astype(int) for s in fl])
    g = np.concatenate([np.full(len(s.t), i) for i, s in enumerate(fl)])
    m, C = fit_logit(X, y, g)
    pis = [m.predict_proba(stroke_X(D, s))[:, 1] for s in fl]
    return m, C, pis


def fit_markov(fl, D_up, D_stay):
    """transition models on observed lag-1 transitions; initial-state model on t=0 windows"""
    rows_up, rows_st = [], []
    for i, s in enumerate(fl):
        Xs = stroke_X(D_up, s); Xs2 = stroke_X(D_stay, s)
        h = s.H.astype(int)
        for j in range(1, len(h)):
            (rows_st if h[j - 1] else rows_up).append((Xs[j] if not h[j - 1] else Xs2[j], h[j], i))
    out = []
    for rows in (rows_up, rows_st):
        X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])
        out.append(fit_logit(X, y, g))
    X0 = np.array([stroke_X(D_up, s)[0] for s in fl]); y0 = np.array([int(s.H[0]) for s in fl])
    m0 = fit_logit(X0, y0, np.arange(len(fl)))
    p_up = [out[0][0].predict_proba(stroke_X(D_up, s))[:, 1] for s in fl]
    p_st = [out[1][0].predict_proba(stroke_X(D_stay, s))[:, 1] for s in fl]
    p0 = m0[0].predict_proba(X0)[:, 1]
    return p_up, p_st, p0, (out[0][1], out[1][1], m0[1])


# ---------------------------------------------------------------- simulation
def sim_indep(pis, rng):
    return [rng.random(len(p)) < p for p in pis]


def sim_copula(pis, rng, w, rho, delta=None):
    labs = []
    for p in pis:
        n = len(p)
        u = rng.standard_normal()
        e = np.empty(n)
        e[0] = rng.standard_normal()
        if rho > 0:
            eta = rng.standard_normal(n)
            a = np.sqrt(1 - rho * rho)
            for j in range(1, n):
                e[j] = rho * e[j - 1] + a * eta[j]
        else:
            e[1:] = rng.standard_normal(n - 1)
        z = np.sqrt(w) * u + np.sqrt(1 - w) * e
        thr = norm.ppf(np.clip(p, 1e-9, 1 - 1e-9))
        if delta is None:
            labs.append(z < thr)
        else:
            h = np.zeros(n, bool)
            state = False
            for j in range(n):
                state = (z[j] < thr[j] + delta) if state else (z[j] < thr[j])
                h[j] = state
            labs.append(h)
    return labs


def sim_markov(p_up, p_st, p0, rng):
    labs = []
    for pu, ps, q0 in zip(p_up, p_st, p0):
        n = len(pu)
        r = rng.random(n)
        h = np.zeros(n, bool)
        h[0] = r[0] < q0
        for j in range(1, n):
            h[j] = r[j] < (ps[j] if h[j - 1] else pu[j])
        labs.append(h)
    return labs


# ---------------------------------------------------------------- statistics
def stats(fl, labs, q=0.1, min_n=5, late_t=0.3, mt_t=0.5):
    cells = defaultdict(list)
    up = down = nH = nN = 0
    for s, h in zip(fl, labs):
        hi = h.astype(int)
        d = np.diff(hi)
        up += np.sum(d == 1); down += np.sum(d == -1); nH += np.sum(hi[:-1] == 1); nN += np.sum(hi[:-1] == 0)
        r = runs_of(h, K)
        if r.size == 0:
            continue
        ends = r + K - 1
        ie = ends[0]
        jl = ends[np.argmin(s.F[ends])]
        maint = [(s.t[j], s.F[j]) for j in ends[1:]]
        mdict = {round(s.t[j], 3): s.F[j] for j in ends[1:]}
        cells[s.col].append((s.F[ie], s.F[jl], s.t[ie], maint, mdict))
    ratios, lowent, lb_num, lb_den, mt, allc = [], [], 0, 0, [], 0
    loo_n = loo_d = 0
    pair = []
    for c, g in cells.items():
        if len(g) < min_n:
            continue
        e = np.array([x[0] for x in g]); lo = np.array([x[1] for x in g]); te = np.array([x[2] for x in g])
        qe, ql = np.quantile(e, q), np.quantile(lo, q)
        ratios.append(ql / qe)
        lowent.extend(lo / e)
        late = te >= late_t - 1e-9
        lb_num += np.sum(e[late] < ql); lb_den += np.sum(late)
        e_mt = e[te >= mt_t - 1e-9]
        m_mt = [f for x in g for (tt, f) in x[3] if tt >= mt_t - 1e-9]
        if len(e_mt) >= 3 and len(m_mt) >= 3:
            mt.append(np.quantile(m_mt, q) / np.quantile(e_mt, q))
        # leave-one-out: low of stroke i below the lowest entry of every OTHER stroke in the column
        order = np.sort(e)
        for i in range(len(g)):
            other_min = order[1] if e[i] == order[0] else order[0]
            loo_n += lo[i] < other_min; loo_d += 1
        # matched-time pairs: for each late entry, share of strokes maintained at that same window time whose
        # force is below the entry force
        for i in np.flatnonzero(late):
            key = round(te[i], 3)
            fm = [x[4][key] for k2, x in enumerate(g) if k2 != i and key in x[4]]
            if fm:
                pair.append(np.mean(np.array(fm) < e[i]))
    ratios = np.array(ratios)
    return dict(ratio=float(np.median(ratios)) if ratios.size else np.nan,
                frac_lt1=float(np.mean(ratios < 1)) if ratios.size else np.nan, ncols=int(ratios.size),
                lowent=float(np.median(lowent)) if lowent else np.nan,
                late_below_FM=lb_num / lb_den if lb_den else np.nan, n_late=int(lb_den),
                MT=float(np.median(mt)) if mt else np.nan, n_MT=len(mt),
                loo=loo_n / loo_d if loo_d else np.nan,
                pair=float(np.mean(pair)) if pair else np.nan, n_pair=len(pair),
                up=up / max(nN, 1), down=down / max(nH, 1),
                nostitch=float(np.mean([np.all(h == h[0]) for h in labs])),
                entered=float(np.mean([runs_of(h, K).size > 0 for h in labs])))


def summarize(name, obs, sims, keys=("ratio", "frac_lt1", "lowent", "loo", "late_below_FM", "MT", "pair", "up", "down", "nostitch", "entered")):
    lines = [f"[{name}] n_sims={len(sims)}"]
    for k in keys:
        v = np.array([s[k] for s in sims], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            lines.append(f"   {k:14s} obs {obs[k]:.4f}  null: none finite")
            continue
        p_lo = np.mean(v <= obs[k]); p_hi = np.mean(v >= obs[k])
        lines.append(f"   {k:14s} obs {obs[k]:.4f}  null median {np.median(v):.4f} [2.5% {np.percentile(v, 2.5):.4f}, 97.5% {np.percentile(v, 97.5):.4f}]"
                     f"  P(null<=obs) {p_lo:.3f}  P(null>=obs) {p_hi:.3f}")
    return "\n".join(lines)


def main(tag="A2", folder=0, nsim=1000, grid_sims=12):
    t0 = time.time()
    fl, v = prepare(tag, folder)
    obs = stats(fl, [s.H for s in fl])
    print(f"{tag} folder {folder} v={v:.3f}: floor strokes {len(fl)}; observed: " +
          ", ".join(f"{k}={val:.4f}" if isinstance(val, float) else f"{k}={val}" for k, val in obs.items()), flush=True)

    D = Design(fl)
    m, C, pis = fit_marginal(fl, D)
    print(f"marginal logistic: C={C}, n features {D.X(np.zeros(1), np.zeros(1), np.zeros(1, int), np.zeros(1)).shape[1]}", flush=True)
    # calibration: observed vs mean predicted P(H) by time bin x rank band
    tb = [(0, .3), (.3, .8), (.8, 1.3), (1.3, 2.2)]; rb = [(1, 10), (11, 20), (21, 30), (31, 50)]
    print("calibration obs/pred P(H) [rows rank bands 1-10, 11-20, 21-30, 31-50; cols t bins 0-.3, .3-.8, .8-1.3, 1.3+]")
    for r0, r1 in rb:
        cells = []
        for a, b in tb:
            o = [h for s, p in zip(fl, pis) for tt, h in zip(s.t, s.H) if r0 <= s.rank <= r1 and a <= tt < b]
            pp = [x for s, p in zip(fl, pis) for tt, x in zip(s.t, p) if r0 <= s.rank <= r1 and a <= tt < b]
            cells.append(f"{np.mean(o):.2f}/{np.mean(pp):.2f}")
        print("   ranks", r0, r1, " ".join(cells))

    rng = RNG
    res = {}
    # independent windows
    sims = [stats(fl, sim_indep(pis, rng)) for _ in range(min(nsim, 300))]
    print(summarize("indep (no persistence)", obs, sims), flush=True)
    res["indep"] = sims

    # copula grid
    print("\ncopula grid (w, rho): mean over sims of ratio / up / down / nostitch / entered", flush=True)
    grid = []
    for w in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0):
        for rho in ((0.0,) if w == 1.0 else (0.0, 0.8, 0.95, 0.99, 0.998)):
            ss = [stats(fl, sim_copula(pis, rng, w, rho)) for _ in range(grid_sims)]
            mean = {k: float(np.nanmean([s[k] for s in ss])) for k in ("ratio", "up", "down", "nostitch", "entered", "late_below_FM", "MT")}
            err = sum(((mean[k] - obs[k]) / max(obs[k], 1e-3)) ** 2 for k in ("up", "down", "nostitch", "entered"))
            grid.append((err, w, rho, mean))
            print(f"   w={w:.2f} rho={rho:.3f}: ratio {mean['ratio']:.3f} up {mean['up']:.4f} down {mean['down']:.4f} "
                  f"nostitch {mean['nostitch']:.3f} entered {mean['entered']:.3f} late<FM {mean['late_below_FM']:.3f} MT {mean['MT']:.2f}  fit-err {err:.3f}", flush=True)
    grid.sort(key=lambda x: x[0])
    _, wb, rb_, _ = grid[0]
    sims = [stats(fl, sim_copula(pis, rng, wb, rb_)) for _ in range(nsim)]
    print(summarize(f"copula best-matching persistence w={wb} rho={rb_}", obs, sims), flush=True)
    res["copula_best"] = sims
    sims = [stats(fl, sim_copula(pis, rng, 1.0, 0.0)) for _ in range(nsim)]
    print(summarize("settling-only null (b): copula w=1 (fixed per-stroke threshold, pi depends on F_now and t only)", obs, sims), flush=True)
    res["settling"] = sims

    # Markov
    Du = Design(fl, cols_onehot=False); Ds = Design(fl, cols_onehot=False)
    p_up, p_st, p0, Cs = fit_markov(fl, Du, Ds)
    print(f"\nMarkov transition fits (C up/stay/init = {Cs})", flush=True)
    sims = [stats(fl, sim_markov(p_up, p_st, p0, rng)) for _ in range(nsim)]
    print(summarize("markov (memory = previous window only)", obs, sims), flush=True)
    res["markov"] = sims

    # positive control: explicit hysteresis
    for delta in (0.5, 1.0, 2.0):
        sims = [stats(fl, sim_copula(pis, rng, 1.0, 0.0, delta=delta)) for _ in range(min(nsim, 300))]
        print(summarize(f"POSITIVE CONTROL hysteresis delta={delta} (w=1)", obs, sims), flush=True)
        res[f"hyst{delta}"] = sims
    for delta in (1.0,):
        sims = [stats(fl, sim_copula(pis, rng, wb, rb_, delta=delta)) for _ in range(min(nsim, 300))]
        print(summarize(f"POSITIVE CONTROL hysteresis delta={delta} (w={wb}, rho={rb_})", obs, sims), flush=True)
    # worlds WITHOUT settling (pi depends on current force and column only): does the statistic discriminate
    # hysteresis there?  (positive/negative controls for power; not fitted to reproduce the time trend)
    D0 = Design(fl, use_t=False)
    m0, C0, pis0 = fit_marginal(fl, D0)
    for delta in (None, 0.5, 1.0, 2.0):
        sims = [stats(fl, sim_copula(pis0, rng, 1.0, 0.0, delta=delta)) for _ in range(min(nsim, 300))]
        print(summarize(f"NO-SETTLING world (pi(F_now, col) only, w=1), hysteresis delta={delta}", obs, sims), flush=True)
        res[f"nosettle_{delta}"] = sims
    np.save(f"null_results_{tag}_f{folder}.npy", {"obs": obs, "res": res}, allow_pickle=True)
    print(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "A2"
    folder = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nsim = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    main(tag, folder, nsim)
