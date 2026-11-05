# IDEA: Plan an explicit refutation witness per live rule (a high-left column whose early starts are closed by cheap zero reads, plus a low-right column closed by one forced positive pair or by a zero cover of its late starts, or one column with no run), price every witness as expected-reads-per-success under a within-case kernel label model, commit to the cheapest one, and when no refutation is plausible plan the cheapest pass certificate instead.
# NOVELTY: Instead of one-step counterfactual shrinkage of intervals, it looks many reads ahead by pricing whole multi-read proofs (domino cover of early starts + positive pair) with the cost and probability of every needed outcome, so it commits to the cheapest refutation per rule rather than spreading reads; probabilities come from cross-column intercept alignment of already-read waveforms, and a fail-fast ordering reads the least-certain element of the chosen proof first.
import math

import numpy as np

from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns

LAM = 2.0
SIG = 0.12
CXMAX = 0.4
KAPPA = 1.5
C0 = 0.12
PFLOOR = 0.03
BONUS = 0.15
MAXQ = 400
K0 = 6
KS = 3
DEBUG = []
STRICT = False
TH_K = 400.0

_WCACHE = {}
_CACHE = {"n": -5, "last": None, "rest": []}
CACHE_FROM = 60


def _fallback(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


def _wmat(N):
    if N not in _WCACHE:
        idx = np.arange(N)
        _WCACHE[N] = np.exp(-np.abs(idx[:, None] - idx[None, :]) / LAM)
    return _WCACHE[N]


def _predict(view, P, known):
    C, R, N = P.shape
    idx = np.arange(N)
    W = _wmat(N)
    lab = ((P == 1) & known[:, None, :]).astype(float)
    S0 = known.astype(float) @ W
    S1 = lab @ W
    a = np.log(np.maximum(view.force, 1e-9)) + 2.0 * view.log_beta[:, None]
    af = a.reshape(-1)
    kidx = np.flatnonzero(known.reshape(-1))
    col = np.repeat(np.arange(C), N)
    Wx = np.exp(-np.abs(af[:, None] - af[kidx][None, :]) / SIG)
    Wx *= (col[:, None] != col[kidx][None, :])
    kx = Wx.sum(axis=1)
    labK = lab.transpose(0, 2, 1).reshape(C * N, R)[kidx]
    px = (Wx @ labK) / np.maximum(kx, 1e-12)[:, None]
    cx = CXMAX * (1.0 - np.exp(-kx / KAPPA))
    px = px.reshape(C, N, R).transpose(0, 2, 1)
    cx = cx.reshape(C, N)[:, None, :]
    prior = 0.1 + 0.4 * np.exp(-((idx - 26.0) / 16.0) ** 2)
    pi1 = (S1 + cx * px + C0 * prior) / (S0[:, None, :] + cx + C0)
    return np.where(known[:, None, :], lab, pi1)


def _ecov(n, pc):
    """Expected reads of n serial reads whose joint success probability is pc (equal per-read odds)."""
    nn = np.maximum(n, 1.0)
    q = np.power(np.maximum(pc, 1e-300), 1.0 / nn)
    gap = 1.0 - q
    e = np.where(gap < 1e-6, nn, (1.0 - pc) / np.maximum(gap, 1e-12))
    return np.where(n > 0, e, 0.0)


def _cover_tables(need_set, unk, pi1):
    """Forward domino cover: zero reads that eliminate the starts in need_set (prefix-cumulative)."""
    C, R, N = need_set.shape
    p0 = 1.0 - pi1
    cum_n = np.zeros((C, R, N))
    cum_lp = np.zeros((C, R, N))
    pick = np.full((C, R, N), -1, dtype=np.int64)
    n = np.zeros((C, R))
    lp = np.zeros((C, R))
    precov = np.zeros((C, R), dtype=bool)
    for s in range(0, N - 1):
        need = need_set[:, :, s] & ~precov
        pA = np.where(unk[:, s + 1][:, None], p0[:, :, s + 1], -1.0)
        pB = np.where(unk[:, s][:, None], p0[:, :, s], -1.0)
        useA = (pA + BONUS) >= pB
        pbest = np.where(useA, pA, pB)
        jbest = np.where(useA, s + 1, s)
        ok = pbest >= 0
        take = need & ok
        n = n + take
        lp = lp + np.where(need, np.where(ok, np.log(np.maximum(pbest, PFLOOR)), -60.0), 0.0)
        pick[:, :, s] = np.where(take, jbest, -1)
        precov = take & useA
        cum_n[:, :, s] = n
        cum_lp[:, :, s] = lp
    cum_n[:, :, N - 1] = n
    cum_lp[:, :, N - 1] = lp
    return cum_n, cum_lp, pick


def _suffix_tables(cand, unk, pi1):
    """Backward domino cover: zero reads eliminating all candidate starts >= s (suffix-cumulative)."""
    C, R, N = cand.shape
    p0 = 1.0 - pi1
    suf_n = np.zeros((C, R, N + 1))
    suf_lp = np.zeros((C, R, N + 1))
    n = np.zeros((C, R))
    lp = np.zeros((C, R))
    precov = np.zeros((C, R), dtype=bool)
    for s in range(N - 2, 0, -1):
        need = cand[:, :, s] & ~precov
        pB = np.where(unk[:, s][:, None], p0[:, :, s], -1.0)
        pA = np.where(unk[:, s + 1][:, None], p0[:, :, s + 1], -1.0)
        useB = (pB + BONUS) >= pA
        pbest = np.where(useB, pB, pA)
        ok = pbest >= 0
        take = need & ok
        n = n + take
        lp = lp + np.where(need, np.where(ok, np.log(np.maximum(pbest, PFLOOR)), -60.0), 0.0)
        precov = take & useB
        suf_n[:, :, s] = n
        suf_lp[:, :, s] = lp
    suf_n[:, :, 0] = n
    suf_lp[:, :, 0] = lp
    return suf_n, suf_lp


def _calibration(view, known, K0):
    C, R, N = view.partial.shape
    f = view.force
    a = np.where(f > 0, np.log(np.maximum(f, 1e-9)) + 2.0 * view.log_beta[:, None], np.nan)
    lo, hi = np.nanmin(a), np.nanmax(a)
    k = int(known.sum())
    targets = np.linspace(lo, hi, K0 + 2)[1:-1]
    centre = np.argsort(np.abs(np.arange(K0) - (K0 - 1) / 2.0), kind="stable")
    t = targets[centre[k]]
    used = known.sum(axis=1)
    cost = np.abs(np.nan_to_num(a, nan=1e9) - t) + 0.08 * used[:, None]
    cost[known] = np.inf
    c, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return int(c), int(j)


def _kill(ctx):
    """Best refutation attempt over live rules: (score, reads)."""
    C, R, N = ctx["cand"].shape
    cand, unk, pi1 = ctx["cand"], ctx["unk"], ctx["pi1"]
    lower, upper, valid = ctx["lower"], ctx["upper"], ctx["valid"]
    cum_n, cum_lp, pick = _cover_tables(cand, unk, pi1)
    suf_n, suf_lp = _suffix_tables(cand, unk, pi1)

    Rt = np.maximum.accumulate(np.where(cand, upper[:, None, :], -np.inf), axis=2)  # (C,R,N)
    # family A: positive pair at (s2, s2+1); index N-1 = nothing to read
    nA = np.zeros((C, R, N))
    pA = np.ones((C, R, N))
    nA[:, :, :N - 1] = (unk[:, :N - 1].astype(float) + unk[:, 1:].astype(float))[:, None, :]
    pA[:, :, :N - 1] = pi1[:, :, :N - 1] * pi1[:, :, 1:]
    pu1 = np.where(unk[:, :N - 1][:, None, :], pi1[:, :, :N - 1], 1.0)
    pu2 = np.where(unk[:, 1:][:, None, :], pi1[:, :, 1:], 1.0)
    pfirst = np.ones((C, R, N))
    pfirst[:, :, :N - 1] = np.minimum(pu1, pu2)
    EA = np.where(nA > 1.5, 1.0 + pfirst, np.where(nA > 0.5, 1.0, 0.0))
    # family B: zero cover of every candidate start > t in column c2 (right bound = Rt[t])
    nB = suf_n[:, :, 1:N + 1][:, :, :N]  # suffix from t+1 -> index t+1
    nB = suf_n[:, :, 1:N + 1]
    lpB = suf_lp[:, :, 1:N + 1]
    pB = np.exp(lpB)
    EB = _ecov(nB, pB)

    nOpt = np.concatenate([nA, nB], axis=2).transpose(1, 0, 2)  # (R,C2,2N)
    pOpt = np.concatenate([pA, pB], axis=2).transpose(1, 0, 2)
    EOpt = np.concatenate([EA, EB], axis=2).transpose(1, 0, 2)
    TOpt = np.concatenate([Rt, Rt], axis=2).transpose(1, 0, 2)
    S2 = 2 * N

    # keep only options not dominated in (right bound, c2-side cost per success)
    gOpt = np.where(pOpt > 1e-9, EOpt / np.maximum(pOpt, 1e-12), np.inf)
    order = np.argsort(TOpt, axis=2, kind="stable")
    gs = np.take_along_axis(gOpt, order, 2)
    rm = np.minimum.accumulate(gs, axis=2)
    prev = np.concatenate([np.full(rm.shape[:2] + (1,), np.inf), rm[:, :, :-1]], axis=2)
    keep = gs < prev
    Kmax = max(1, int(keep.sum(axis=2).max()))
    comp = np.argsort(~keep, axis=2, kind="stable")[:, :, :Kmax]
    sel = np.take_along_axis(order, comp, 2)            # original option index, (R,C2,K)
    kk = np.take_along_axis(keep, comp, 2)
    Tt = np.where(kk, np.take_along_axis(TOpt, sel, 2), np.nan)
    nK = np.take_along_axis(nOpt, sel, 2)
    pK = np.where(kk, np.take_along_axis(pOpt, sel, 2), 0.0)
    EK = np.take_along_axis(EOpt, sel, 2)

    Tflat = np.ascontiguousarray(np.nan_to_num(Tt, nan=-np.inf)).reshape(-1)
    M1 = np.zeros((R, C, C, Kmax), dtype=np.int64)
    for c1 in range(C):
        vi = np.flatnonzero(valid[c1])
        if vi.size == 0:
            continue
        lows = lower[c1, vi]
        ss = np.searchsorted(lows, Tflat, side="right") - 1
        m = np.where(ss >= 0, vi[np.maximum(ss, 0)], 0)
        M1[:, c1] = m.reshape(R, C, Kmax)
    r_ix = np.arange(R)[:, None, None, None]
    c1_ix = np.arange(C)[None, :, None, None]
    pc_tab = np.exp(cum_lp)
    ecov_tab = _ecov(cum_n, pc_tab)
    cn = cum_n[c1_ix, r_ix, M1]
    pc = pc_tab[c1_ix, r_ix, M1]
    ecov = ecov_tab[c1_ix, r_ix, M1]
    nT = nK[:, None, :, :]
    pT = pK[:, None, :, :]
    ET = EK[:, None, :, :]
    ntot = nT + cn
    ptot = pT * pc
    etot = ET + pT * ecov
    bad = np.eye(C, dtype=bool)[None, :, :, None] & np.isfinite(Tt)[:, None, :, :]
    ok = (ntot > 0) & (ptot > 1e-9) & ~bad & kk[:, None, :, :]
    score = np.where(ok, etot / np.maximum(ptot, 1e-12), np.inf)
    flat = score.reshape(R, -1)
    am = flat.argmin(axis=1)
    best_pair = flat[np.arange(R), am]

    sn = cum_n[:, :, N - 1]
    sp = np.exp(cum_lp[:, :, N - 1])
    ssc = np.where((sn > 0) & (sp > 1e-9), _ecov(sn, sp) / np.maximum(sp, 1e-12), np.inf)
    sam = ssc.argmin(axis=0)
    best_single = ssc[sam, np.arange(R)]

    best = np.minimum(best_pair, best_single)
    r = int(best.argmin())
    if not np.isfinite(best[r]):
        return np.inf, None
    reads = []
    if best_single[r] <= best_pair[r]:
        c = int(sam[r])
        for s in range(0, N - 1):
            j = int(pick[c, r, s])
            if j >= 0:
                reads.append((c, j, float(1 - pi1[c, r, j]), 0, r))
    else:
        c1, c2, kq = np.unravel_index(int(am[r]), (C, C, Kmax))
        c1, c2, kq = int(c1), int(c2), int(kq)
        s2 = int(sel[r, c2, kq])
        m1 = int(M1[r, c1, c2, kq])
        if s2 < N - 1:
            for j in (s2, s2 + 1):
                if unk[c2, j]:
                    reads.append((c2, j, float(pi1[c2, r, j]), 1, r))
        elif s2 >= N:
            t = s2 - N
            # zero cover of candidate starts > t in column c2, recomputed explicitly
            need = cand[c2, r].copy()
            need[:t + 1] = False
            s = N - 2
            precov = False
            while s >= 1:
                if need[s] and not precov:
                    pb = float(1 - pi1[c2, r, s]) if unk[c2, s] else -1.0
                    pa = float(1 - pi1[c2, r, s + 1]) if unk[c2, s + 1] else -1.0
                    if pb + BONUS >= pa:
                        if pb >= 0:
                            reads.append((c2, s, pb, 0, r))
                        precov = True
                    else:
                        reads.append((c2, s + 1, pa, 0, r))
                        precov = False
                else:
                    precov = False
                s -= 1
        for s in range(0, m1 + 1):
            j = int(pick[c1, r, s])
            if j >= 0:
                reads.append((c1, j, float(1 - pi1[c1, r, j]), 0, r))
    if not reads:
        return np.inf, None
    return float(best[r]), reads


def _pass(ctx):
    """Cheapest pass certificate over live rules: (log-score, reads)."""
    C, R, N = ctx["cand"].shape
    unk, pi1, possible = ctx["unk"], ctx["pi1"], ctx["possible"]
    lower, upper, valid = ctx["lower"], ctx["upper"], ctx["valid"]
    ones = pi1 >= 0.5
    pm = ones[:, :, :N - 1] & ones[:, :, 1:]
    has = pm.any(axis=2)
    shat = np.where(has, pm.argmax(axis=2), N - 1)
    cidx = np.arange(C)[:, None]
    lo_hat = np.where(has, lower[cidx, shat], np.nan)
    up_hat = np.where(has, upper[cidx, shat], np.nan)
    colok = has & np.isfinite(lo_hat)
    feas = colok.all(axis=0)
    if not feas.any():
        return np.inf, None
    Lstar = np.where(colok, lo_hat, -np.inf).max(axis=0)  # (R,)
    feas &= np.where(colok, up_hat >= Lstar[None, :], True).all(axis=0)
    if not feas.any():
        return np.inf, None
    sidx = np.arange(N)
    need = possible & (~valid[:, None, :] | (upper[:, None, :] < Lstar[None, :, None]))
    need &= sidx[None, None, :] < shat[:, :, None]
    cum_n, cum_lp, pick = _cover_tables(need, unk, pi1)
    sh = np.minimum(shat, N - 2)
    prev = np.maximum(sh - 1, 0)
    rr = np.arange(R)[None, :]
    cn = cum_n[cidx, rr, prev]
    cl = cum_lp[cidx, rr, prev]
    n2 = unk[cidx, sh].astype(float) + unk[cidx, sh + 1].astype(float)
    p2 = pi1[cidx, rr, sh] * pi1[cidx, rr, sh + 1]
    ntot = (cn + n2).sum(axis=0)
    lp = cl.sum(axis=0) + np.log(np.maximum(p2, 1e-12)).sum(axis=0)
    ptot = np.exp(np.maximum(lp, -300))
    E = _ecov(ntot, ptot)
    ls = np.where(feas & (ntot > 0), np.log(np.maximum(E, 1e-9)) - lp, np.inf)
    r = int(ls.argmin())
    if not np.isfinite(ls[r]):
        return np.inf, None
    reads = []
    for c in range(C):
        s = int(shat[c, r])
        for j in (s, s + 1):
            if unk[c, j]:
                reads.append((c, j, float(pi1[c, r, j]), 1, r))
        for q in range(0, max(s, 0)):
            j = int(pick[c, r, q])
            if j >= 0:
                reads.append((c, j, float(1 - pi1[c, r, j]), 0, r))
    if not reads:
        return np.inf, None
    return float(ls[r]), reads


def _plan(view):
    P = view.partial
    C, R, N = P.shape
    known = P[:, 0, :] != -1
    unk = ~known
    if view.queries_so_far < KS:
        return None
    if view.queries_so_far >= CACHE_FROM and _CACHE["n"] == view.queries_so_far - 1 and _CACHE["last"] is not None:
        lc, lj, _, lpl, lrule = _CACHE["last"]
        if int(P[lc, lrule, lj]) == lpl:
            rest = [t for t in _CACHE["rest"] if P[t[0], 0, t[1]] == -1]
            if rest:
                _CACHE["n"] = view.queries_so_far
                _CACHE["last"] = rest[0]
                _CACHE["rest"] = rest[1:]
                return rest[0][0], rest[0][1]
    lower = view.lower[:, 0, :]
    upper = view.upper[:, 0, :]
    valid = np.isfinite(lower)
    possible = view.possible[..., :N]
    cand = possible & valid[:, None, :]

    cb = cand.any(axis=2).all(axis=0)
    left = np.where(cand, lower[:, None, :], np.inf).min(axis=2)
    right = np.where(cand, upper[:, None, :], -np.inf).max(axis=2)
    live = cb & (left.max(axis=0) <= right.min(axis=0))
    ix = np.flatnonzero(live)
    if ix.size == 0:
        return None

    pi1 = _predict(view, P, known)[:, ix]
    ctx = dict(cand=cand[:, ix], unk=unk, pi1=pi1, lower=lower, upper=upper, valid=valid,
               possible=possible[:, ix])
    ks, kreads = _kill(ctx)
    qs, qreads = _pass(ctx)
    ps = math.exp(min(qs, 600.0)) if np.isfinite(qs) else np.inf
    DEBUG.append((round(ks, 2) if np.isfinite(ks) else None, round(ps, 2) if np.isfinite(ps) else None))
    if kreads is None and qreads is None:
        return None
    if kreads is None or (qreads is not None and (ps < ks or ks > TH_K)):
        reads = qreads
    else:
        reads = kreads
    seen = set()
    uniq = []
    for c, j, p, pl, rr in reads:
        if (c, j) not in seen:
            seen.add((c, j))
            uniq.append((c, j, p, pl, int(ix[rr])))
    uniq.sort(key=lambda t: t[2])
    _CACHE["n"] = view.queries_so_far
    _CACHE["last"] = uniq[0]
    _CACHE["rest"] = uniq[1:]
    return uniq[0][0], uniq[0][1]


def choose(view):
    res = None
    if view.queries_so_far < MAXQ:
        if STRICT:
            res = _plan(view)
        else:
            try:
                res = _plan(view)
            except Exception:
                res = None
    if res is None:
        res = _fallback(view)
    c, j = int(res[0]), int(res[1])
    if view.partial[c, 0, j] != -1:
        c, j = (int(x[0]) for x in np.nonzero(view.partial[:, 0, :] == -1))
        c, j = int(c), int(j)
    return c, j
