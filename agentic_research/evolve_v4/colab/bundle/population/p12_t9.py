# IDEA: Shared-witness routing: verification candidates are ranked by cost divided by the number of nested rules sharing the witness column, and at wide tolerance (eps>=0.3) a tolerance-free no-valid-start cover is taken ahead of pair verification whenever its read count is below twice the cheapest pair verification.
# NOVELTY: Parent only uses the no-valid-start cover after pair margins are exhausted; this prices it against pair verification up front and amortises costs across nested rules, instead of any information-gain or lookahead score.
import numpy as np
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns


def _fallback(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


KSEED = 8
DEAD_ON = True
DEADMUL = 2.0


def choose(view):
    res = _choose(view)
    c, r = int(res[0]), int(res[1])
    if view.partial[c, 0, r] != -1:
        return _fallback(view)
    return c, r


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
    left = np.where(cand, Lc[:, None, :], np.inf).min(-1)
    right = np.where(cand, Hc[:, None, :], -np.inf).max(-1)
    has = cand.any(-1)
    alive = has.all(0) & (left.max(0) <= right.min(0))
    if not alive.any() or C < 2:
        return _fallback(view)
    alive_rules = [int(r) for r in np.flatnonzero(alive)]

    s0 = (P[:, :, 0] == 1) & (P[:, :, 1] == -1) & alive[None, :]
    if s0.any():
        return int(np.flatnonzero(s0.any(1))[0]), 1

    one = P == 1
    zero = P == 0
    pad = np.ones((C, R, 1), bool)
    zl = np.concatenate([pad, zero[..., :-1]], -1)
    zr = np.concatenate([zero[..., 1:], pad], -1)
    ro = one & ~(zl & zr)
    fo = np.where(ro, idx, N).min(-1)
    lz = np.where(zero & (idx < fo[..., None]), idx, -1).max(-1)
    lo = np.clip(lz + 1, 1, N - 2)
    hi = np.where(fo < N, np.minimum(fo, N - 2), N - 2)
    hi = np.maximum(hi, lo)

    def at(A, k):
        return np.take_along_axis(np.broadcast_to(A[:, None, :], (C, R, N)), k[..., None], -1)[..., 0]

    Lh, Lpot = at(Lc, lo), at(Lc, hi)
    Hpot = at(Hc, lo)
    Hh = np.where(fo < N, at(Hc, hi), np.inf)
    cap = Hc[:, N - 2][:, None]

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

    def sweep_read(c, r, include_invalid=False):
        m = poss[c, r] if include_invalid else cand[c, r]
        nz = np.flatnonzero(m)
        if len(nz) == 0:
            return None
        f = int(nz[0])
        if f + 1 < N and unread[c, f + 1]:
            return c, f + 1
        if unread[c, f]:
            return c, f
        return None

    def lower_read(c, r):
        k = int(fo[c, r])
        ranks = np.flatnonzero(unread[c] & (idx > lz[c, r]) & (idx < k))
        if len(ranks):
            return c, int(ranks[len(ranks) // 2])
        if k < N:
            return pair_read(c, r)
        return None

    # ---- 1. verification, amortised over rules sharing a witness column ------
    T = Hatt.min(0)
    Bc = Hatt.argmin(0)
    ver = []
    for r in alive_rules:
        t = T[r]
        b = int(Bc[r])
        if not np.isfinite(t):
            continue
        ok = (Lh[:, r] > t) & (rows != b)
        if not ok.any():
            continue
        nb = (cand[:, r, :] & (Lc <= t)).sum(-1)
        cost_a = np.where(ok, (nb + 1) // 2, 10 ** 6)
        a = int(cost_a.argmin())
        cb = 0 if right[b, r] <= t + 1e-9 else 1
        ver.append((int(cost_a[a]) + cb, r, a, b, cb))
    share = {}
    for cost, r, a, b, cb in ver:
        key = ("B", b) if cb else ("A", a)
        share[key] = share.get(key, 0) + 1
    ver2 = []
    for cost, r, a, b, cb in ver:
        key = ("B", b) if cb else ("A", a)
        ver2.append((cost / float(share[key]), cost, r, a, b, cb))
    ver2.sort()
    if DEAD_ON and ver2:
        dr = _dead_route(view, P, poss, cand, unread, alive_rules, C, R, N, True)
        if dr is not None and dr[1] < ver2[0][1] * (DEADMUL if view.epsilon >= 0.3 else 0.0):
            return dr[0]
    for _, _, r, a, b, cb in ver2:
        rd = pair_read(b, r) if cb else sweep_read(a, r)
        if rd is not None:
            return rd

    refs = []
    for r in alive_rules:
        oa = np.argsort(-Lpot[:, r], kind="stable")
        ob = np.argsort(Hpot[:, r], kind="stable")
        a, b = int(oa[0]), int(ob[0])
        if a == b:
            m1 = Lpot[a, r] - Hpot[int(ob[1]), r]
            m2 = Lpot[int(oa[1]), r] - Hpot[b, r]
            if m1 >= m2:
                b = int(ob[1])
            else:
                a = int(oa[1])
        m = Lpot[a, r] - Hpot[b, r]
        if m > 0:
            refs.append((-m, r, a, b))
    refs.sort()
    for _, r, a, b in refs:
        hb = Hh[b, r] if np.isfinite(Hh[b, r]) else cap[b, 0]
        ov_lo = max(Lh[a, r], Hpot[b, r])
        ov_hi = min(Lpot[a, r], hb)
        tc = 0.5 * (ov_lo + ov_hi) if ov_hi > ov_lo else ov_lo
        wa = Lpot[a, r] - Lh[a, r]
        wb = hb - Hpot[b, r]
        order = (("A", a), ("B", b)) if wa >= wb else (("B", b), ("A", a))
        for who, col in order:
            if who == "A":
                st = np.flatnonzero(Lc[col] > tc)
                q = (int(st.min()) if len(st) else N - 1) - 1
            else:
                st = np.flatnonzero(Hc[col] <= tc)
                q = int(st.max()) if len(st) else 1
            ranks = np.flatnonzero(unread[col] & (idx > lz[col, r]) & (idx < fo[col, r]))
            if len(ranks):
                return col, int(ranks[np.abs(ranks - q).argmin()])

    if DEAD_ON:
        rd = _dead_route(view, P, poss, cand, unread, alive_rules, C, R, N)
        if rd is not None:
            return rd

    best = None
    for r in alive_rules:
        g = np.where(cand[:, r, :], idx, -1).max(-1)
        f = np.where(cand[:, r, :], idx, N).min(-1)
        Lg = Lc[rows, np.maximum(g, 0)]
        Hf = Hc[rows, np.minimum(f, N - 1)]
        X, Y = int(Lg.argmax()), int(Hf.argmin())
        gap = Lg[X] - Hf[Y]
        key = (gap, r)
        if best is None or key < best[0]:
            best = (key, r, X, Y)
    (_, _), r, X, Y = best
    g = np.where(cand[:, r, :], idx, -1).max(-1)
    f = np.where(cand[:, r, :], idx, N).min(-1)
    gap = Lc[X, max(int(g[X]), 0)] - Hc[Y, min(int(f[Y]), N - 1)]
    tries = []
    if gap > 0:
        tries = [("L", X), ("S", Y), ("S", X), ("L", Y)]
    bad = np.flatnonzero((poss[:, r, :] & ~fin).any(-1))
    for c in bad:
        c = int(c)
        if poss[c, r, N - 1]:
            tries.append(("L", c))
            tries.append(("S", c))
        else:
            tries.append(("I", c))
    for kind, c in tries:
        if kind == "L":
            rd = lower_read(c, r)
        elif kind == "S":
            rd = sweep_read(c, r)
        else:
            rd = sweep_read(c, r, True)
        if rd is not None:
            return rd
    return _fallback(view)


def _dead_route(view, P, poss, cand, unread, alive_rules, C, R, N, withcost=False):
    u = view.rules[:, 0]
    v = view.rules[:, 1]
    stricter = (u[None, :] >= u[:, None]) & (v[None, :] <= v[:, None])
    am = np.zeros(R, bool)
    am[alive_rules] = True
    benefit = (stricter & am[None, :]).sum(1).astype(float)
    V = cand.copy()
    V[:, :, N - 1] = False
    cov = np.full((C, R), -1)
    cnt = np.zeros((C, R))
    infeas = np.zeros((C, R), bool)
    for s in range(N - 1):
        need = V[:, :, s] & (cov < s)
        if not need.any():
            continue
        hi_ok = unread[:, s + 1][:, None]
        lo_ok = unread[:, s][:, None]
        use_hi = need & hi_ok
        use_lo = need & ~hi_ok & lo_ok
        infeas |= need & ~hi_ok & ~lo_ok
        cnt += (use_hi | use_lo)
        cov = np.where(use_hi, s + 1, np.where(use_lo, s, cov))
    npos = (P == 1).sum(-1)
    okm = (~infeas) & (npos == 0) & am[None, :] & (cnt > 0)
    if not okm.any():
        return None
    score = np.where(okm, (cnt + 1.0) / np.maximum(benefit[None, :], 1.0), np.inf)
    flat = int(np.argmin(score))
    c, r = divmod(flat, R)
    plan = []
    covered = -1
    for s in np.flatnonzero(V[c, r]):
        if s <= covered:
            continue
        if unread[c, s + 1]:
            plan.append(s + 1)
            covered = s + 1
        elif unread[c, s]:
            plan.append(s)
            covered = s
    if not plan:
        return None
    plan = np.array(plan)
    readr = np.flatnonzero(~unread[c])
    if len(readr):
        d = np.abs(plan[:, None] - readr[None, :]).min(1).astype(float)
    else:
        d = -np.abs(plan - np.median(plan)).astype(float)
    k = int(np.argmax(d + 1e-3 * plan))
    if withcost:
        return (c, int(plan[k])), float(cnt[c, r])
    return c, int(plan[k])
