# IDEA: Witness commit-and-verify (from t21) with three structural changes: (1) an A column is ruled out only
#   by a confirmed positive pair below the threshold, and its cost is a greedy zero-cover count that steps over
#   isolated positives; (2) the (A, B) witness pair is chosen jointly from a prefix-cover table; (3) a pass-proof
#   mode that pins every column inside a common window, entered by a ski-rental rule (reads spent so far >=
#   optimistic remaining pass cost of the cheapest alive rule).
# NOVELTY: t21 judged A feasibility from the first "real" positive, so one isolated positive vetoed a column for
#   the whole run; it fixed B to the single lowest attainable ceiling; and it proved passes with the same bottom-up
#   sweeps as refutations. Labels only (view.revealed is never read); no per-case constants.
import numpy as np
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns

KSEED = 8        # opening reads taken by the published one-step counterfactual lookahead
COVER = True     # A feasibility = no confirmed pair below the threshold; cost = greedy zero cover
JOINT = True     # choose the (A, B) witness pair jointly instead of fixing B to the lowest ceiling
SKI = 1.0        # pass-proof switch: reads so far >= SKI * remaining pass cost (None disables the mode)
FAILFIRST = True  # with a B pair still to complete, open the side with the larger start domain first
ORACLE_T = [None]
SHARE = True     # amortise a witness column over the rules that would use it


def _fallback(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


def choose(view):
    res = _choose(view)
    c, r = int(res[0]), int(res[1])
    if view.partial[c, 0, r] != -1:
        return _fallback(view)
    return c, r


def _prefix_cover(cand, unread, N):
    """PC[c, r, s] = zero reads needed to kill every candidate start <= s if the unread labels are zero
    (greedy interval stabbing: a zero at k kills starts k-1 and k; known positives are stepped over)."""
    C, R, _ = cand.shape
    cov = np.full((C, R), -1)
    cnt = np.zeros((C, R), int)
    PC = np.zeros((C, R, N), int)
    for s in range(N - 1):
        need = cand[:, :, s] & (cov < s)
        if need.any():
            hi_ok = unread[:, s + 1][:, None]
            lo_ok = unread[:, s][:, None]
            use_hi = need & hi_ok
            use_lo = need & ~hi_ok & lo_ok
            cnt = cnt + (use_hi | use_lo)
            cov = np.where(use_hi, s + 1, np.where(use_lo, s, cov))
        PC[:, :, s] = cnt
    PC[:, :, N - 1] = cnt
    return PC


def _pass_cost(P, poss, fin, Lc, Hc, unread, N):
    """Optimistic remaining reads to prove each rule passes: a pair in every column without one (2 reads)
    plus zero reads that kill the invalid starts and the starts whose upper end lies below the current
    common-point floor x (the largest lower end among columns that already have a confirmed pair)."""
    C, R, _ = P.shape
    S = poss & fin[:, None, :]
    pair = (P[..., :-1] == 1) & (P[..., 1:] == 1)
    pinned = pair.any(-1)
    gl = np.where(S, Lc[:, None, :], -np.inf).max(-1)
    x = np.where(pinned, gl, -np.inf).max(0)
    need = (poss & ~fin[:, None, :]) | (S & (Hc[:, None, :] < x[None, :, None]))
    need[:, :, N - 1] = False
    cov = np.full((C, R), -1)
    cnt = np.zeros((C, R))
    for s in range(N - 1):
        nd = need[:, :, s] & (cov < s)
        if not nd.any():
            continue
        hi_ok = unread[:, s + 1][:, None]
        lo_ok = unread[:, s][:, None]
        use_hi = nd & hi_ok
        use_lo = nd & ~hi_ok & lo_ok
        cnt += use_hi | use_lo
        cov = np.where(use_hi, s + 1, np.where(use_lo, s, cov))
    return (cnt + 2 * ~pinned).sum(0)


def _pin_read(r, P, poss, fin, Lc, Hc, unread, N):
    """One read of the pass proof for rule r: pin each column's first run inside the common window.
    Columns whose floor is above the current floor x are pinned exactly (bottom-up, they raise x); the
    others look for a pair top-down inside their window [lower <= x]; once every column has a pair, the
    invalid starts and the starts whose upper end is below x are killed by zero reads."""
    idx = np.arange(N)
    pr = P[:, r, :]
    S = poss[:, r, :] & fin
    pair = (pr[:, :-1] == 1) & (pr[:, 1:] == 1)
    fp = np.where(pair, idx[:-1], N).min(-1)
    pinned = fp < N
    G = np.where(S, Lc, np.inf).min(-1)
    gl = np.where(S, Lc, -np.inf).max(-1)
    x = gl[pinned].max() if pinned.any() else -np.inf

    def first_pos_read(c, lo):
        for s in np.flatnonzero(S[c] & (idx >= lo)):
            if s + 1 < N and unread[c, s + 1]:
                return c, int(s + 1)
            if unread[c, s]:
                return c, int(s)
        return None

    if (~pinned).any():
        un = np.flatnonzero(~pinned)
        high = un[G[un] > x]
        if len(high):
            c = int(high[np.argmax(G[high])])
            rd = first_pos_read(c, 0)
            if rd is not None:
                return rd
        for c in un[np.argsort(-G[un], kind="stable")]:
            c = int(c)
            win = np.flatnonzero(S[c] & (Lc[c] <= x))
            if len(win) == 0:
                rd = first_pos_read(c, 0)
            else:
                w1 = int(win[-1])
                rd = None
                for s in win[::-1]:
                    s = int(s)
                    a, b = pr[c, s], pr[c, s + 1] if s + 1 < N else 0
                    if a == 0 or b == 0:
                        continue
                    if a == -1 and b == -1:
                        rd = (c, s + 1) if s == w1 else (c, s)
                    elif a == -1:
                        rd = (c, s)
                    else:
                        rd = (c, s + 1)
                    break
                if rd is None:
                    rd = first_pos_read(c, w1 + 1)
            if rd is not None:
                return rd
        return None
    need = (poss[:, r, :] & ~fin) | (S & (Hc < x))
    need[:, N - 1] = False
    for c in np.flatnonzero(need.any(-1)):
        s = int(np.flatnonzero(need[c])[0])
        if s + 1 < N and unread[c, s + 1] and pr[c, s + 1] != 1:
            return int(c), s + 1
        if unread[c, s]:
            return int(c), s
        if s + 1 < N and unread[c, s + 1]:
            return int(c), s + 1
        if s >= 1 and unread[c, s - 1]:
            return int(c), s - 1
    return None


def _choose(view):
    if view.queries_so_far < KSEED:
        return _fallback(view)
    P = view.partial
    C, R, N = P.shape
    idx = np.arange(N)
    rows = np.arange(C)
    unread = P[:, 0, :] == -1
    L = view.lower[:, 0, :]
    H = view.upper[:, 0, :]
    fin = np.isfinite(L) & np.isfinite(H)
    Lc = np.where(fin, L, -np.inf)
    Hc = np.where(fin, H, np.inf)
    poss = view.possible
    cand = poss & fin[:, None, :]
    left = np.where(cand, Lc[:, None, :], np.inf).min(-1)    # floor G of each column and rule
    right = np.where(cand, Hc[:, None, :], -np.inf).max(-1)  # ceiling H
    alive = cand.any(-1).all(0) & (left.max(0) <= right.min(0))
    if not alive.any() or C < 2:
        return _fallback(view)
    alive_rules = [int(r) for r in np.flatnonzero(alive)]

    # ---- pass proof (ski rental) -------------------------------------------------------------------
    if SKI is not None:
        pc = _pass_cost(P, poss, fin, Lc, Hc, unread, N)
        best = min(alive_rules, key=lambda r: (pc[r], r))
        if view.queries_so_far >= SKI * pc[best]:
            rd = _pin_read(best, P, poss, fin, Lc, Hc, unread, N)
            if rd is not None:
                return rd

    # ---- attainable ceilings: a known positive that can still complete into a pair -------------------
    one = P == 1
    zero = P == 0
    pad = np.ones((C, R, 1), bool)
    zl = np.concatenate([pad, zero[..., :-1]], -1)
    zr = np.concatenate([zero[..., 1:], pad], -1)
    ro = one & ~(zl & zr)                      # positive not already isolated by two zeros
    fo = np.where(ro, idx, N).min(-1)
    hi = np.where(fo < N, np.minimum(fo, N - 2), N - 2)
    Hh = np.where(fo < N, np.take_along_axis(np.broadcast_to(Hc[:, None, :], (C, R, N)), hi[..., None], -1)[..., 0], np.inf)
    fo1 = np.minimum(fo + 1, N - 1)
    fom = np.maximum(fo - 1, 0)
    nxt = np.take_along_axis(P, fo1[..., None], -1)[..., 0]
    forced_fo = (fo < N - 1) & (nxt == 1)
    ub = np.broadcast_to(unread[:, None, :], (C, R, N))
    un_next = (fo + 1 < N) & np.take_along_axis(ub, fo1[..., None], -1)[..., 0]
    un_prev = (fo >= 1) & np.take_along_axis(ub, fom[..., None], -1)[..., 0]
    comp = (fo < N) & (forced_fo | un_next | un_prev)
    Hatt = np.where(comp, Hh, right)

    def pair_read(b, r):
        k = int(fo[b, r])
        if k + 1 < N and unread[b, k + 1]:
            return b, k + 1
        if k >= 1 and unread[b, k - 1]:
            return b, k - 1
        return None

    def sweep_read(c, r):
        nz = np.flatnonzero(cand[c, r])
        if len(nz) == 0:
            return None
        f = int(nz[0])
        if f + 1 < N and unread[c, f + 1]:
            return c, f + 1
        if unread[c, f]:
            return c, f
        return None

    # A feasibility barrier: lower end at the first confirmed pair (two adjacent known positives)
    fpair = (P[..., :-1] == 1) & (P[..., 1:] == 1)
    fps = np.where(fpair, idx[:-1], N).min(-1)
    Lfp = np.where(fps < N, np.take_along_axis(np.broadcast_to(Lc[:, None, :], (C, R, N)), np.minimum(fps, N - 1)[..., None], -1)[..., 0], np.inf)
    if not COVER:  # t21's barrier: the start just above the last zero below the first possibly-real positive
        lz = np.where(zero & (idx < fo[..., None]), idx, -1).max(-1)
        lo = np.clip(lz + 1, 1, N - 2)
        Lfp = np.take_along_axis(np.broadcast_to(Lc[:, None, :], (C, R, N)), lo[..., None], -1)[..., 0]
    PC = _prefix_cover(cand, unread, N) if COVER else np.cumsum(cand, -1) // 2 + (np.cumsum(cand, -1) % 2)

    # ---- refutation witnesses -----------------------------------------------------------------------
    ver = []
    for r in alive_rules:
        th = Hatt[:, r]
        okb = np.isfinite(th)
        if not JOINT:  # B fixed to the lowest attainable ceiling
            okb = okb & (rows == int(np.argmin(th)))
        if not okb.any():
            continue
        m = cand[None, :, r, :] & (Lc[None] <= th[:, None, None])
        last = np.where(m, idx, -1).max(-1)
        ca = np.where(last >= 0, PC[:, r, :][rows[None, :], np.maximum(last, 0)], 0)
        feas = (Lfp[None, :, r] > th[:, None]) & (rows[None, :] != rows[:, None]) & okb[:, None]
        if ORACLE_T[0] is not None:  # analysis only: true boundary intercept of each column
            feas = feas & (ORACLE_T[0][None, :, r] > th[:, None])
        cbv = np.where(right[:, r] <= th + 1e-9, 0, 1)
        tot = np.where(feas, ca + cbv[:, None], 10 ** 6)
        b, a = divmod(int(tot.argmin()), C)
        if tot[b, a] >= 10 ** 6:
            continue
        ver.append((int(tot[b, a]), r, a, b, int(cbv[b])))
    share = {}
    for cost, r, a, b, cb in ver:
        key = ("B", b) if cb else ("A", a)
        share[key] = share.get(key, 0) + 1
    ver2 = sorted((cost / float(share[("B", b) if cb else ("A", a)] if SHARE else 1), cost, r, a, b, cb)
                  for cost, r, a, b, cb in ver)
    for _, _, r, a, b, cb in ver2:
        if cb:
            if FAILFIRST and int(cand[a, r].sum()) > int(cand[b, r].sum()):
                rd = sweep_read(a, r)
                if rd is None:
                    rd = pair_read(b, r)
            else:
                rd = pair_read(b, r)
                if rd is None:
                    rd = sweep_read(a, r)
        else:
            rd = sweep_read(a, r)
        if rd is not None:
            return rd
    return _fallback(view)
