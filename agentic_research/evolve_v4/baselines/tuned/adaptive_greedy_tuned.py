# Tuned copy of baselines/adaptive_greedy.py (search only, labels only, 29 scorings; see BASELINES_FAIR.md).
# Best of the tuning run: GEO=16, GEO_CAP=0.3, OFFSETS=None, SMOOTH=3 -> search 4513 (eps0.2 1782 / eps0.4 2731).
"""SBFE-style adaptive greedy (Deshpande-Hellerstein-Kletenik flavour), labels only.
Tunable copy of baselines/adaptive_greedy.py.

Progress measure Phi (monotone, submodular-like): over live rules, the number of
still-possible first-run starts (version-space size) plus KILL x the number of
rules refuted. Each unread cell costs 1. Under an independent per-rule label
model (smoothed positive rate, optionally tilted by neighbouring read cells)
we pick the cell with the largest expected one-step reduction of Phi.

Change from the original: the one-step effect of a read on the possible starts is computed
in closed form (a 0 at k removes starts k-1 and k; a 1 at k removes start k+1 and, next to a
read 1, every start after the forced pair and the no-run slot). This equals recomputing
possible_starts on the hypothetical partial, only faster. Knobs:
  KILL       weight of refuting a rule, in units of eliminated starts
  NEIGHBOUR  how far a read neighbour pulls the cell's positive probability toward its label
  SMOOTH     pseudocount of the positive-rate estimate
  PRIOR      "global": per-rule rate over all columns (original); "column": per (column, rule),
             shrunk to the per-rule rate with strength COL_STRENGTH
  OFFSETS    cells considered, relative to still-possible starts; None = every unread cell
  ELIM       "count": eliminated starts (original); "frac": eliminated / possible per
             (column, rule), so a column with few candidates counts as much as a wide one
  RULES      "live": progress only for rules that can still pass (original); "all"
  GEO        weight of the smooth refutation progress: increase of the rule's outer gap
             max_c left - min_c right (log-intercept units, capped at GEO_CAP); 0 = original
  PASSW      extra weight for eliminating starts that block a pass (invalid starts, or starts
             whose interval misses the rule's current outer overlap); 0 = original
"""
import numpy as np

KILL = 8.0
NEIGHBOUR = 0.6
SMOOTH = 3.0
PRIOR = "global"
COL_STRENGTH = 4.0
OFFSETS = None
ELIM = "count"
RULES = "live"
GEO = 16
GEO_CAP = 0.3
PASSW = 0.0


def choose(view):
    partial, possible, lower, upper = view.partial, view.possible, view.lower, view.upper
    cols, nrules, ranks = partial.shape
    nstart = ranks - 1                                    # start slots; index ranks-1 = no run
    valid = np.isfinite(lower)
    cand = possible & valid
    left = np.where(cand, lower, np.inf).min(-1)
    right = np.where(cand, upper, -np.inf).max(-1)
    live = cand.any(-1).all(0) & (left.max(0) <= right.min(0))
    weight_rules = live if RULES == "live" else np.ones(nrules, bool)
    unread = partial[:, 0, :] == -1
    if OFFSETS is None:
        mask = unread.copy()
    else:
        mask = np.zeros((cols, ranks), bool)
        starts = possible[:, live, :-1].any(1) if live.any() else possible[:, :, :-1].any(1)
        for off in OFFSETS:
            lo, hi = max(0, -off), min(ranks - 1, ranks - 1 - off)
            mask[:, lo + off:hi + off] |= starts[:, lo:hi]
        mask &= unread
        if not mask.any():
            mask = unread
    cs, ks = np.nonzero(mask)
    n = len(cs)
    n1 = (partial == 1).sum((0, 2))
    n0 = (partial == 0).sum((0, 2))
    p1 = (n1 + SMOOTH) / (n1 + n0 + 2.0 * SMOOTH)                    # (rules,)
    if PRIOR == "column":
        c1 = (partial == 1).sum(2)                                   # (cols, rules)
        c0 = (partial == 0).sum(2)
        pc = (c1 + COL_STRENGTH * p1) / (c1 + c0 + COL_STRENGTH)
        p = pc[cs].copy()
    else:
        p = np.broadcast_to(p1, (n, nrules)).copy()
    if NEIGHBOUR:
        for off in (-1, 1):
            kk = np.clip(ks + off, 0, ranks - 1)
            nb = partial[cs, :, kk]
            seen = (nb != -1) & (ks + off >= 0)[:, None] & (ks + off < ranks)[:, None]
            p = np.where(seen, (1 - NEIGHBOUR) * p + NEIGHBOUR * (nb == 1), p)
    # other-column outer bounds (max / min over the other columns, via top-2)
    if cols > 1:
        o = np.argsort(-left, axis=0, kind="stable")
        top1, top2 = np.take_along_axis(left, o[:1], 0)[0], np.take_along_axis(left, o[1:2], 0)[0]
        l_col = np.where(np.arange(cols)[:, None] == o[0][None], top2[None], top1[None])
        o = np.argsort(right, axis=0, kind="stable")
        bot1, bot2 = np.take_along_axis(right, o[:1], 0)[0], np.take_along_axis(right, o[1:2], 0)[0]
        r_col = np.where(np.arange(cols)[:, None] == o[0][None], bot2[None], bot1[None])
    else:
        l_col = np.full((cols, nrules), -np.inf)
        r_col = np.full((cols, nrules), np.inf)
    l_other, r_other = l_col[cs], r_col[cs]
    # closed-form one-step effect per (candidate, rule); intercept bounds of the surviving valid
    # starts from the three extreme candidates (at most two are removed by one read) or prefixes.
    Lm = np.where(cand, lower, np.inf)                       # (cols, rules, ranks)
    Um = np.where(cand, upper, -np.inf)
    oL = np.argsort(Lm, axis=-1, kind="stable")[..., :3]
    oU = np.argsort(-Um, axis=-1, kind="stable")[..., :3]
    vL = np.take_along_axis(Lm, oL, -1)
    vU = np.take_along_axis(Um, oU, -1)
    pminL = np.minimum.accumulate(Lm, -1)
    pmaxU = np.maximum.accumulate(Um, -1)
    ncand = cand.sum(-1)
    pcand = np.cumsum(cand, -1)
    suffix = np.cumsum(possible[..., ::-1], -1)[..., ::-1]   # possible starts at slot >= j
    R = np.arange(nrules)[None, :]
    C = cs[:, None]
    K = ks[:, None]

    def at(A, k, ok):
        kk = np.clip(k, 0, ranks - 1)
        return np.where(ok, A[C, R, kk], 0)

    def extreme(v, o, removed):
        """first of the three extremes whose position is not removed (removed: list of (pos, ok))."""
        vv, oo = v[cs], o[cs]                                 # (n, rules, 3)
        keep = np.ones(vv.shape, bool)
        for pos, ok in removed:
            keep &= ~((oo == np.broadcast_to(pos, ok.shape)[..., None]) & ok[..., None])
        first = keep.argmax(-1)
        return np.take_along_axis(vv, first[..., None], -1)[..., 0]

    total = possible[cs].sum(-1)
    okm1 = np.broadcast_to(K >= 1, (n, nrules))
    ok0 = np.broadcast_to(K <= nstart - 1, (n, nrules))
    okp1 = np.broadcast_to(K + 1 <= nstart - 1, (n, nrules))
    # outcome 0: starts k-1 and k go
    elim0 = at(possible, K - 1, okm1) + at(possible, K, ok0)
    rm0 = [(K - 1, okm1), (K, ok0)]
    ln0, rn0 = extreme(vL, oL, rm0), extreme(vU, oU, rm0)
    cnt0 = ncand[cs] - at(cand, K - 1, okm1) - at(cand, K, ok0)
    # outcome 1: start k+1 goes; a read 1 next to k forces the run, removing every later slot
    prev1 = okm1 & (partial[C, R, np.clip(K - 1, 0, ranks - 1)] == 1)
    next1 = (K + 1 < ranks) & (partial[C, R, np.clip(K + 1, 0, ranks - 1)] == 1)
    forced = prev1 | next1
    thr = np.where(prev1, K, K + 1)
    elim1 = np.where(forced, at(suffix, thr, forced), at(possible, K + 1, okp1))
    rm1 = [(K + 1, okp1)]
    ln1 = np.where(forced, at(pminL, thr - 1, forced), extreme(vL, oL, rm1))
    rn1 = np.where(forced, at(pmaxU, thr - 1, forced), extreme(vU, oU, rm1))
    ln1 = np.where(forced & (thr < 1), np.inf, ln1)
    cnt1 = np.where(forced, at(pcand, thr - 1, forced), ncand[cs] - at(cand, K + 1, okp1))
    if PASSW:
        Lr, Rr = left.max(0), right.min(0)                   # (rules,)
        bad = possible & (~valid | (lower > Rr[None, :, None]) | (upper < Lr[None, :, None]))
        bsuffix = np.cumsum(bad[..., ::-1], -1)[..., ::-1]
        bad0 = at(bad, K - 1, okm1) + at(bad, K, ok0)
        bad1 = np.where(forced, at(bsuffix, thr, forced), at(bad, K + 1, okp1))
    else:
        bad0 = bad1 = 0
    gap0 = left.max(0) - right.min(0)                         # (rules,) current outer gap
    progress = []
    for elim, ln, rn, cnt, bd in ((elim0, ln0, rn0, cnt0, bad0), (elim1, ln1, rn1, cnt1, bad1)):
        if ELIM == "frac":
            elim = elim / np.maximum(total, 1)
        gap = np.maximum(ln, l_other) - np.minimum(rn, r_other)
        killed = (cnt == 0) | (gap > 0)
        val = elim + KILL * killed + PASSW * bd
        if GEO:
            grow = np.where(np.isfinite(gap), gap, gap0 + GEO_CAP) - gap0[None, :]
            val = val + GEO * np.clip(np.nan_to_num(grow, nan=0.0, posinf=GEO_CAP), 0, GEO_CAP)
        progress.append(val * weight_rules)
    expected = ((1 - p) * progress[0] + p * progress[1]).sum(-1)
    best = np.lexsort((cs, ks, -np.round(expected, 9)))[0]
    return int(cs[best]), int(ks[best])
