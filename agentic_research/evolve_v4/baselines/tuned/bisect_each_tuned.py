# Tuned copy of baselines/bisect_each.py (search only, labels only, 30 scorings; see BASELINES_FAIR.md).
# Best of the tuning run: LIVE_ONLY, ORDER=bound, BOUND=left, STEP=1 -> search 5550 (eps0.2 1789 / eps0.4 3761).
"""Practitioner baseline: measure every column's break point by bisection (tunable copy).

Same idea as baselines/bisect_each.py: find each column's first-run start by bisection over its
candidate starts, then let the frozen certificate check the law. Labels only.
Knobs (defaults reproduce the original file):
  ORDER     which ambiguous column to measure next: "index" (original), "fewest", "most",
            "round" (breadth-first: the column with the widest candidate span),
            "bound" (the column that sets the most live rules' outer overlap bound first)
  FRAC      bisection point inside [first, last] candidate start (0.5 = midpoint)
  LIVE_ONLY only measure rules that can still pass (practitioner stops measuring refuted rules)
  STEP      None: bisection point at FRAC of the candidate span (original); an integer: fixed-step
            staircase, probe STEP ranks above the lowest candidate start
  SPAN      "union": bisect the union of ambiguous rules' starts (original); "widest": bisect the
            single rule with the widest candidate span in that column
"""
import numpy as np

ORDER = "bound"
FRAC = 0.5
LIVE_ONLY = True
SPAN = "union"
STEP = 1
BOUND = "left"
                    # "sum" = left hits + right hits, "left" / "right" = only that side
TIE = "fewest"      # "bound" order tie break: "fewest" candidate starts, or "index"


def _outer(view):
    possible, lower, upper = view.possible, view.lower, view.upper
    cand = possible & np.isfinite(lower)
    left = np.where(cand, lower, np.inf).min(-1)
    right = np.where(cand, upper, -np.inf).max(-1)
    return left, right, cand.any(-1).all(0) & (left.max(0) <= right.min(0))


def _live(view):
    return _outer(view)[2]


def choose(view):
    partial, possible = view.partial, view.possible
    unread = partial[:, 0, :] == -1
    cols, nrules, ranks = partial.shape
    rules = _live(view) if LIVE_ONLY else np.ones(nrules, bool)
    if not rules.any():
        rules = np.ones(nrules, bool)
    options = []
    if ORDER == "bound":
        left, right, live = _outer(view)
        lh, rh = (left == left.max(0)) & live, (right == right.min(0)) & live
        hits_all = {"any": lh | rh, "sum": lh.astype(int) + rh, "left": lh, "right": rh}[BOUND].sum(1)
    for c in range(cols):
        ambiguous = (possible[c].sum(-1) > 1) & rules
        if not ambiguous.any():
            continue
        sub = possible[c][ambiguous][:, :-1]
        if SPAN == "widest":
            spans = [(np.flatnonzero(s).max() - np.flatnonzero(s).min()) if s.any() else -1 for s in sub]
            sub = sub[[int(np.argmax(spans))]]
        starts = np.flatnonzero(sub.any(0))
        if len(starts) == 0:
            continue
        cand = np.unique(np.concatenate([starts, np.minimum(starts + 1, ranks - 1)]))
        cand = cand[unread[c, cand]]
        if len(cand) == 0:
            continue
        if STEP is None:
            mid = starts.min() + FRAC * (starts.max() + 1 - starts.min())
        else:
            mid = starts.min() + STEP
        rank = int(cand[np.argmin(np.abs(cand - mid))])
        if ORDER == "index":
            return c, rank
        width = starts.max() - starts.min() + 1
        if ORDER == "bound":
            key = (-int(hits_all[c]), len(starts) if TIE == "fewest" else 0, c)
        else:
            key = {"fewest": (len(starts), c), "most": (-len(starts), c), "round": (-width, c)}[ORDER]
        options.append((key, c, rank))
    if options:
        _, c, rank = min(options)
        return int(c), rank
    c, r = np.argwhere(unread)[0]
    return int(c), int(r)
