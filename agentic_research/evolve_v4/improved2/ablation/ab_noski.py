# IDEA: improved/general_policy.py with three monotone-mode changes (noisy mode, after a monotonicity violation, is
#   unchanged = improved/best_policy.py):
#   (1) barrier: while no violation has been seen, a single known positive bars an A column (monotone labels imply
#       the pair above it), so sweeps certain to fail under monotone labels are not started;
#   (2) probe: a sweep held back by the gate is tested with ONE read at the top of its region when that read is at
#       least as likely to be zero as positive, taking the boundary uniform over the column's open gap
#       (highest known zero, lowest known positive]. A zero is the evidence the gate waits for (the sweep then
#       cannot fail under monotone labels); a positive bars the column. Otherwise t21's witness bisection runs;
#   (3) when no witness, probe or bisection target is left (no alive rule can be separated under monotone
#       potentials), the read goes to the cheapest pass proof instead of the generic one-step lookahead.
# BASE (improved/general_policy.py): best_policy.py plus a label-regime switch. While every column read so far is
#   monotone along force (no zero above a positive in the same column and rule), a long A sweep (greedy cover
#   dearer than bisecting its region, ceil(log2(n+1))) must wait for evidence that the column's boundary is above
#   the threshold, and t21's witness bisection supplies that evidence.
import numpy as np
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns

KSEED = 8        # opening reads taken by the published one-step counterfactual lookahead
COVER = True     # A feasibility = no confirmed pair below the threshold; cost = greedy zero cover
JOINT = False    # True: choose the (A, B) pair jointly over all B candidates (no gain once LOOSE is on)
SKI = None        # pass-proof switch: reads so far >= SKI * remaining pass cost (None disables the mode)
FAILFIRST = True  # with a B pair still to complete, open the side with the larger start domain first
LOOSE = True     # attack the loosest alive rule first (dominance count in the (u, v) threshold order)
GATE = True      # long sweeps wait for bisection evidence while labels look monotone
ADAPT = True     # the gate is lifted for good once a zero is read above a positive (noisy labels)
MONOBAR = True   # monotone mode: the first known positive (not only a confirmed pair) bars an A column
PROBE = True     # monotone mode: probe the top of the cheapest held-back sweep before bisecting
ODDS = 1.0       # probe only when P(zero)/P(positive) >= ODDS under that prior (1: more likely than not)
FBPIN = True     # monotone mode: with nothing left to refute or bisect, read for the cheapest pass proof
SHARE = False    # True: amortise a witness column over the rules that would use it (no gain once LOOSE is on)


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
    lz = np.where(zero & (idx < fo[..., None]), idx, -1).max(-1)
    lo = np.clip(lz + 1, 1, N - 2)

    def at(A, k):
        return np.take_along_axis(np.broadcast_to(A[:, None, :], (C, R, N)), k[..., None], -1)[..., 0]

    Lh = at(Lc, lo)  # t21's evidence: floor reachable if the first possibly-real positive is real
    hib = np.maximum(hi, lo)
    Lpot, Hpot = at(Lc, hib), at(Hc, lo)
    if not COVER:  # t21's barrier: the start just above the last zero below the first possibly-real positive
        Lfp = Lh
    PC = _prefix_cover(cand, unread, N) if COVER else np.cumsum(cand, -1) // 2 + (np.cumsum(cand, -1) % 2)

    # label regime seen so far in this case: a zero read above a positive (same column and rule) means the
    # labels are not monotone along force, so bisection misses first runs and bottom-up sweeps are the right
    # tool; while every column still looks monotone, long sweeps wait for bisection evidence (t21's refs)
    first_one = np.where(P == 1, idx, N).min(-1)
    last_zero = np.where(P == 0, idx, -1).max(-1)
    noisy = bool((last_zero > first_one).any()) if ADAPT else False  # a zero read above a positive
    probes = []
    Lfo = np.where(first_one < N, at(Lc, np.minimum(first_one, N - 1)), np.inf)  # start at the first positive

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
        if MONOBAR and GATE and not noisy:
            # monotone labels: a known positive implies the pair above it, so it already bars the column
            feas = feas & (Lfo[None, :, r] > th[:, None])
        if GATE and not noisy:
            # a long sweep (cover dearer than bisecting its region) needs evidence that the column's boundary
            # lies above the threshold; otherwise locate boundaries by bisection first
            nreg = m.sum(-1)
            short = ca <= np.ceil(np.log2(nreg + 1.0))
            ok = short | (Lh[None, :, r] > th[:, None])
            if PROBE:
                tb = np.where(feas & ~ok, ca, 10 ** 6)
                pb, pa = divmod(int(tb.argmin()), C)
                if tb[pb, pa] < 10 ** 6:
                    s = int(last[pb, pa])
                    # monotone labels, boundary uniform over the column's open gap (highest zero, lowest positive]
                    z, p = int(last_zero[pa, r]), int(first_one[pa, r])
                    psucc = (p - s) / float(max(p - z, 1))
                    if psucc >= ODDS * (1.0 - psucc):
                        probes.append((int(tb[pb, pa]), r, pa, s))
            feas = feas & ok
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
    if LOOSE:
        u, v = view.rules[:, 0], view.rules[:, 1]
        dom = ((u[None, :] >= u[:, None]) & (v[None, :] <= v[:, None])).sum(1)  # rules at least as strict
        ver2.sort(key=lambda t: (-dom[t[2]], t[0]))
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
    if PROBE and probes:
        # one read at the top of a held-back sweep's region decides it if labels are monotone: a zero there
        # is the evidence the gate waits for, a positive bars the column
        if LOOSE:
            probes.sort(key=lambda t: (-dom[t[1]], t[0]))
        for _, r, a, s in probes:
            for q in (s, s + 1):
                if 0 <= q < N and unread[a, q]:
                    return a, int(q)
    if GATE and not noisy:
        Hh2 = np.where(fo < N, at(Hc, hib), np.inf)
        cap = Hc[:, N - 2]
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
            mg = Lpot[a, r] - Hpot[b, r]
            if mg > 0:
                refs.append((-mg, r, a, b))
        refs.sort()
        for _, r, a, b in refs:
            hb = Hh2[b, r] if np.isfinite(Hh2[b, r]) else cap[b]
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
        if FBPIN and SKI is not None:
            # no witness, no probe and no bisection target: under monotone potentials no alive rule can be
            # separated any more, so the read goes to the cheapest pass proof instead of the generic lookahead
            rd = _pin_read(best, P, poss, fin, Lc, Hc, unread, N)
            if rd is not None:
                return rd
    return _fallback(view)
