"""Practitioner baseline: measure every column's break point by bisection, one column at a time.

This is how a robustness (or playability) study is usually run by hand: find each item's threshold,
then check the law. Labels only. For the first column whose first-run start is still ambiguous for
some rule, read the unread rank nearest the middle of its candidate starts.
"""
import numpy as np


def choose(view):
    partial, possible = view.partial, view.possible
    unread = partial[:, 0, :] == -1
    ranks = partial.shape[2]
    for c in range(partial.shape[0]):
        ambiguous = possible[c].sum(-1) > 1
        if not ambiguous.any():
            continue
        starts = np.flatnonzero(possible[c][ambiguous][:, :-1].any(0))
        if len(starts) == 0:
            continue
        cand = np.unique(np.concatenate([starts, np.minimum(starts + 1, ranks - 1)]))
        cand = cand[unread[c, cand]]
        if len(cand) == 0:
            continue
        mid = (starts.min() + starts.max() + 1) / 2
        return c, int(cand[np.argmin(np.abs(cand - mid))])
    c, r = np.argwhere(unread)[0]
    return int(c), int(r)
