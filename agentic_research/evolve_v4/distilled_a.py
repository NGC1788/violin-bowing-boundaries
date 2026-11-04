# IDEA: Witness-commit-and-verify: per live rule, commit to the column pair that can separate it,
#   verify by a zero-sweep of the high column's early starts plus one forced positive pair in the low
#   column, locate label flips by bisection, and pin the most compatible rule when no refutation is left.
# NOVELTY: Distilled from evolved program t2 (campaign a1): the seed-lookahead opening and the unused
#   rank-0 shortcut are removed, so the counterfactual baseline is used only as a last-resort fallback.
import numpy as np
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns


def _fallback(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


def choose(view):
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

    # ---- onset brackets under the single-flip hypothesis -------------------
    one = P == 1
    zero = P == 0
    pad = np.ones((C, R, 1), bool)
    zl = np.concatenate([pad, zero[..., :-1]], -1)
    zr = np.concatenate([zero[..., 1:], pad], -1)
    ro = one & ~(zl & zr)                       # non-isolated read positives
    fo = np.where(ro, idx, N).min(-1)           # (C,R) lowest such rank, N = none
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

    # ---- 1. verification of hypothesised separations -----------------------
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
    ver.sort()
    for _, r, a, b, cb in ver:
        rd = pair_read(b, r) if cb else sweep_read(a, r)
        if rd is not None:
            return rd

    # ---- 2. refinement: optimistic best pair, probe at the splitting intercept
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

    # ---- 3. pass-style pinning of the most compatible rule -----------------
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
    else:
        bad = np.flatnonzero((poss[:, r, :] & ~fin).any(-1))
        for c in bad:
            c = int(c)
            if poss[c, r, N - 1]:
                tries.append(("L", c))
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
