"""SBFE-style adaptive greedy (Deshpande-Hellerstein-Kletenik flavour), labels only.

Progress measure Phi (monotone, submodular-like): over live rules, the number of
still-possible first-run starts (version-space size) plus KILL x the number of
rules refuted. Each unread cell costs 1. Under an independent per-rule label
model (smoothed global positive rate, optionally tilted by revealed neighbours)
we pick the cell with the largest expected one-step reduction of Phi.
"""
import numpy as np

from agentic_research.evolve_v4.helpers import possible_starts

KILL = 8.0
NEIGHBOUR = 0.6


def choose(view):
    partial, possible, lower, upper = view.partial, view.possible, view.lower, view.upper
    cols, nrules, ranks = partial.shape
    valid = np.isfinite(lower)
    cand = possible & valid
    left = np.where(cand, lower, np.inf).min(-1)
    right = np.where(cand, upper, -np.inf).max(-1)
    live = cand.any(-1).all(0) & (left.max(0) <= right.min(0))
    unread = partial[:, 0, :] == -1
    mask = np.zeros((cols, ranks), bool)
    starts = possible[:, live, :-1].any(1) if live.any() else possible[:, :, :-1].any(1)
    for off in (-1, 0, 1, 2):
        lo, hi = max(0, -off), min(ranks - 1, ranks - 1 - off)
        mask[:, lo + off:hi + off] |= starts[:, lo:hi]
    mask &= unread
    if not mask.any():
        mask = unread
    cs, ks = np.nonzero(mask)
    n = len(cs)
    n1 = (partial == 1).sum((0, 2))
    n0 = (partial == 0).sum((0, 2))
    p1 = (n1 + 1.0) / (n1 + n0 + 2.0)                      # (rules,)
    p = np.broadcast_to(p1, (n, nrules)).copy()
    if NEIGHBOUR:
        for off in (-1, 1):
            kk = np.clip(ks + off, 0, ranks - 1)
            nb = partial[cs, :, kk]
            seen = (nb != -1) & (ks + off >= 0)[:, None] & (ks + off < ranks)[:, None]
            p = np.where(seen, (1 - NEIGHBOUR) * p + NEIGHBOUR * (nb == 1), p)
    # other-column outer bounds
    l_other = np.empty((n, nrules))
    r_other = np.empty((n, nrules))
    for c in np.unique(cs):
        idx = cs == c
        l_other[idx] = np.delete(left, c, 0).max(0) if cols > 1 else -np.inf
        r_other[idx] = np.delete(right, c, 0).min(0) if cols > 1 else np.inf
    base_poss = possible[cs]
    progress = []
    for y in (0, 1):
        h = partial[cs].copy()
        h[np.arange(n), :, ks] = y
        newp = possible_starts(h)
        elim = (base_poss & ~newp).sum(-1)
        cn = newp & valid[cs]
        ln = np.where(cn, lower[cs], np.inf).min(-1)
        rn = np.where(cn, upper[cs], -np.inf).max(-1)
        killed = ~cn.any(-1) | (np.maximum(ln, l_other) > np.minimum(rn, r_other))
        progress.append((elim + KILL * killed) * live)
    expected = ((1 - p) * progress[0] + p * progress[1]).sum(-1)
    best = np.lexsort((cs, ks, -np.round(expected, 9)))[0]
    return int(cs[best]), int(ks[best])
