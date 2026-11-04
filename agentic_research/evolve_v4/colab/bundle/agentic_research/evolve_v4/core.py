"""Frozen certificate, counterfactual baseline and replay, copied verbatim from the repo."""

import numpy as np


def possible_starts(partial):
    """First adjacent positive pair, with final slot for no run. Unknown is -1."""
    forced = (partial[..., :-1] == 1) & (partial[..., 1:] == 1)
    earlier = np.cumsum(forced, axis=-1) - forced
    possible = (partial[..., :-1] != 0) & (partial[..., 1:] != 0) & (earlier == 0)
    possible[..., 1:] &= partial[..., :-2] != 1
    return np.concatenate([possible, ~forced.any(axis=-1, keepdims=True)], axis=-1)


def certificates(possible, lower, upper):
    """Return decision, whether all boundaries are known, and geometric rejection."""
    valid = np.isfinite(lower)
    candidates = possible & valid
    outer_low = np.where(candidates, lower, np.inf).min(axis=-1).max(axis=0)
    outer_high = np.where(candidates, upper, -np.inf).max(axis=-1).min(axis=0)
    can_be_valid = candidates.any(axis=-1).all(axis=0)
    could_pass = can_be_valid & (outer_low <= outer_high)
    all_valid = ~(possible & ~valid).any(axis=(0, 2))
    inner_low = np.where(candidates, lower, -np.inf).max(axis=-1).max(axis=0)
    inner_high = np.where(candidates, upper, np.inf).min(axis=-1).min(axis=0)
    guaranteed_pass = all_valid & (inner_low <= inner_high)
    decision = True if guaranteed_pass.any() else False if not could_pass.any() else None
    geometric_only = bool((can_be_valid & (outer_low > outer_high)).all())
    return decision, bool((possible.sum(axis=-1) == 1).all()), geometric_only


def response_patterns(rules):
    u, v = rules.T
    masks = [(u <= a) & (v >= b) for a in np.unique(u) for b in np.unique(v)]
    return np.vstack([np.zeros(len(rules), dtype=bool), masks]).astype(np.int8)


def outer(possible, lower, upper):
    valid = possible & np.isfinite(lower)
    left = np.where(valid, lower, np.inf).min(axis=-1)
    right = np.where(valid, upper, -np.inf).max(axis=-1)
    return left, right


def shortlist(partial, possible, lower, upper, left, right, live):
    unobserved = partial[:, 0] == -1
    weights = live * (1 + (left == left.max(0)).astype(int)
                      + (right == right.min(0)).astype(int))
    weights *= possible.sum(-1) > 1
    starts = (possible * weights[..., None]).sum(axis=1)
    starts[:, -1] = 0
    score = starts.copy()
    score[:, 1:] += starts[:, :-1]
    score[:, :-1] += starts[:, 1:]
    score[~unobserved] = -1
    order = np.argsort(-score.T.ravel(), kind="stable")
    top = order[:min(6, int(unobserved.sum()))]
    flat = list(top) + [int(np.flatnonzero(unobserved.T)[0])]
    for endpoints, intervals, bound, select in (
        (left, lower, left.max(0), np.argmin),
        (right, upper, right.min(0), np.argmax),
    ):
        for column, rule in np.argwhere((endpoints == bound) & live & (weights > 0)):
            valid = possible[column, rule] & np.isfinite(intervals[column, 0])
            fill = np.inf if select is np.argmin else -np.inf
            start = int(select(np.where(valid, intervals[column, 0], fill)))
            for rank in range(max(0, start - 1), start + 2):
                if unobserved[column, rank]:
                    flat.append(rank * len(partial) + column)
                    break
    rank, column = np.unravel_index(np.unique(flat), score.T.shape)
    return column, rank, score[column, rank]


def counterfactual_choose(partial, possible, lower, upper, patterns, probabilities):
    left, right = outer(possible, lower, upper)
    max_left, min_right = left.max(0), right.min(0)
    live = max_left <= min_right
    column, rank, priority = shortlist(partial, possible, lower, upper, left, right, live)
    hypothetical = np.broadcast_to(
        partial[column, None], (len(column), len(patterns), *partial.shape[1:])
    ).copy()
    hypothetical[np.arange(len(column)), :, :, rank] = patterns
    l_new, r_new = outer(possible_starts(hypothetical), lower[column, None], upper[column, None])
    others = np.arange(len(partial))[None, :] != column[:, None]
    l_other = np.where(others[..., None], left, -np.inf).max(axis=1)
    r_other = np.where(others[..., None], right, np.inf).min(axis=1)
    l_total = np.maximum(l_new, l_other[:, None])
    r_total = np.minimum(r_new, r_other[:, None])
    remaining = (l_total <= r_total) & live
    width = np.where(remaining, np.maximum(r_total - l_total, 0), 0).sum(axis=-1)
    rules = remaining.sum(axis=-1) @ probabilities
    slack = width @ probabilities
    # The shortlist is already ordered by rank, then column.
    selected = np.lexsort((np.arange(len(column)), -priority,
                           np.round(slack, 12), np.round(rules, 12)))[0]
    return int(column[selected]), int(rank[selected])


class View:
    """Everything a policy may see. Arrays are copies; mutating them has no effect."""

    def __init__(self, partial, possible, case, revealed, count):
        self.partial = partial.copy()            # (columns, 9 rules, 50 ranks): -1 unknown, 0/1 seen
        self.possible = possible.copy()          # possible first-run starts, last slot = no run
        self.lower = case["lower"].copy()        # (columns, 1, 50) intercept interval ends, NaN invalid
        self.upper = case["upper"].copy()
        self.force = case["public"]["force"].copy()        # (columns, 50) measured force, sorted
        self.log_beta = case["public"]["log_beta"].copy()  # (columns,)
        self.rules = case["public"]["rules"].copy()        # (9, 2) periodicity floor, ratio ceiling
        self.epsilon = case["epsilon"]
        self.revealed = {k: v.copy() for k, v in revealed.items()}  # NaN where unqueried
        self.queries_so_far = count


def replay(case, choose):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    revealed = {k: np.full(v.shape, np.nan) for k, v in case["hidden"].items()}
    columns, _, ranks = labels.shape
    for count in range(columns * ranks + 1):
        possible = possible_starts(partial)
        decision, _, _ = certificates(possible, lower, upper)
        if decision is not None:
            if decision != case["expected"]:
                raise AssertionError("Certificate disagrees with full decision")
            return count
        col, rank = choose(View(partial, possible, case, revealed, count))
        col, rank = int(col), int(rank)
        if not (0 <= col < columns and 0 <= rank < ranks) or partial[col, 0, rank] != -1:
            raise ValueError(f"Invalid or repeated query {(col, rank)}")
        partial[col, :, rank] = labels[col, :, rank]
        for key in revealed:
            revealed[key][col, rank] = case["hidden"][key][col, rank]
    raise AssertionError("No terminal certificate")
