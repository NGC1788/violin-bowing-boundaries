# IDEA: Ablation of t16: same witness commit-and-verify with fail-first order and galloping, but the no-valid-start cover route is disabled everywhere (DEAD_ON=False).
# NOVELTY: Removes a mechanism instead of adding one: the cover route (t6's main idea) turns out to cost reads once the other mechanisms are present.
import numpy as np
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns


def _fallback(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


KSEED = 8
DEAD_ON = False
DEADMUL = 2.0
COVER = True
JOINT = True
PINFRAC = 0.5
SKI = 1.0


def _prefix_cover(cand, unread, N):
    """PC[c, r, s] = greedy zero-reads needed to kill every candidate start <= s (unread assumed zero)."""
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


def _cover(cand_r, below, unread, N):
    """Reads needed to kill every candidate start whose lower end is below the threshold, if all
    unread labels turn out zero (greedy interval stabbing; known isolated positives are stepped over)."""
    C = cand_r.shape[0]
    cov = np.full(C, -1)
    cnt = np.zeros(C, int)
    for s in range(N - 1):
        need = cand_r[:, s] & below[:, s] & (cov < s)
        if not need.any():
            continue
        hi_ok = unread[:, s + 1]
        use_hi = need & hi_ok
        use_lo = need & ~hi_ok & unread[:, s]
        cnt += use_hi | use_lo
        cov = np.where(use_hi, s + 1, np.where(use_lo, s, cov))
    return cnt



def _pin_read(r, P, poss, fin, Lc, Hc, unread, N):
    """One read of the pass proof for rule r: pin every column's first run inside the common window."""
    C = P.shape[0]
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
        # bottom-up search for the lowest pair among candidate starts >= lo
        for s in np.flatnonzero(S[c] & (idx >= lo)):
            if s + 1 < N and unread[c, s + 1]:
                return c, int(s + 1)
            if unread[c, s]:
                return c, int(s)
        return None

    if (~pinned).any():
        un = np.flatnonzero(~pinned)
        # columns whose floor is above x will raise x: pin them exactly, highest floor first
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
                # top-down pair search inside the window
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
    # all pinned: kill invalid possible starts and starts whose upper end is below x
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


def _pass_cost(P, poss, fin, Lc, Hc, unread, N):
    """Optimistic remaining reads to prove each rule passes: per column a pair (2 reads if none yet)
    plus zero reads that kill the invalid starts and the starts whose upper end lies below the
    current common-point floor x (unread labels assumed favourable)."""
    C, R, _ = P.shape
    idx = np.arange(N)
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

    if SKI:
        pc = _pass_cost(P, poss, fin, Lc, Hc, unread, N)
        best = min(alive_rules, key=lambda r: (pc[r], r))
        if view.queries_so_far >= SKI * pc[best]:
            rd = _pin_read(best, P, poss, fin, Lc, Hc, unread, N)
            if rd is not None:
                return rd
    elif PINFRAC is not None:
        pairs = (P[..., :-1] == 1) & (P[..., 1:] == 1)
        npin = pairs.any(-1).sum(0)  # columns with a forced pair, per rule
        slack = right.min(0) - left.max(0)
        best = max(alive_rules, key=lambda r: (npin[r], slack[r]))
        if npin[best] >= PINFRAC * C:
            rd = _pin_read(best, P, poss, fin, Lc, Hc, unread, N)
            if rd is not None:
                return rd

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
            if k >= N:
                zr_ = np.flatnonzero(P[c, r] == 0)
                z0 = int(zr_.min()) if len(zr_) else 0
                z1 = max(int(lz[c, r]), 0)
                step = max(1, z1 - z0 + 1)
                tgt = z1 + step
                return c, int(ranks[np.abs(ranks - tgt).argmin()])
            return c, int(ranks[len(ranks) // 2])
        if k < N:
            return pair_read(c, r)
        return None

    # first forced pair (two adjacent known positives): the floor can never rise past it
    fpair = (P[..., :-1] == 1) & (P[..., 1:] == 1)
    fps = np.where(fpair, idx[:-1], N).min(-1)
    Lfp = np.where(fps < N, np.take_along_axis(np.broadcast_to(Lc[:, None, :], (C, R, N)), np.minimum(fps, N - 1)[..., None], -1)[..., 0], np.inf)

    # ---- 1. verification, amortised over rules sharing a witness column ------
    T = Hatt.min(0)
    Bc = Hatt.argmin(0)
    ver = []
    if JOINT:
        PC = _prefix_cover(cand, unread, N)
    for r in alive_rules:
        if JOINT:
            th = Hatt[:, r]
            okb = np.isfinite(th)
            if not okb.any():
                continue
            m = cand[None, :, r, :] & (Lc[None] <= th[:, None, None])
            last = np.where(m, idx, -1).max(-1)
            ca = np.where(last >= 0, PC[:, r, :][rows[None, :], np.maximum(last, 0)], 0)
            feas = (Lfp[None, :, r] > th[:, None]) & (rows[None, :] != rows[:, None]) & okb[:, None]
            cbv = np.where(right[:, r] <= th + 1e-9, 0, 1)
            tot = np.where(feas, ca + cbv[:, None], 10 ** 6)
            flat = int(tot.argmin())
            b, a = divmod(flat, C)
            if tot[b, a] >= 10 ** 6:
                continue
            ver.append((int(tot[b, a]), r, a, b, int(cbv[b])))
            continue
        t = T[r]
        b = int(Bc[r])
        if not np.isfinite(t):
            continue
        if COVER:
            ok = (Lfp[:, r] > t) & (rows != b)
            if not ok.any():
                continue
            cost_a = np.where(ok, _cover(cand[:, r, :], Lc <= t, unread, N), 10 ** 6)
        else:
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
        if cb:
            # fail-first: open the half of the witness with the larger remaining start domain
            da = int(cand[a, r].sum())
            db = int(cand[b, r].sum())
            if da > db and view.epsilon >= 0.3:
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
