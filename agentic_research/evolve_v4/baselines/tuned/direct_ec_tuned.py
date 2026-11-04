# Tuned copy of baselines/direct_ec.py (search only, labels only, 23 scorings; see BASELINES_FAIR.md).
# Best of the tuning run: PERSIST, EMPTY=split, SAMPLES=192, COPY=0.97, EMPTYCOL=borrow, REGION=rules
# -> search 8945 (eps0.2 3630 / eps0.4 5315). Same config with SEED 7 / 11: 11516 / 10344 (sampling noise is large).
"""Literature baseline: DIRECT-style edge-cutting greedy for decision region determination.
Tunable copy of baselines/direct_ec.py.

Hypotheses are sampled completions of the unknown labels (labels only, never the hidden features).
Decision regions are the certificate outcomes (accept / reject). Reading a cell cuts every
hypothesis pair in different regions that disagree on that cell; pick the cell with the
largest expected cut weight.

Changes from the original: the sampler and the region test are vectorised over samples (same
hypothesis model, different random stream), so the method fits the 3 s per case limit. Knobs:
  SAMPLES  number of sampled hypotheses per decision
  COPY     probability an unknown cell copies the left/right observed pattern (else uniform)
  FLIP     "rank": an independent flip point per unknown rank (original); "gap": one flip point
           per gap between observed cells, so each sample is monotone inside the gap
  REGION   "decision": accept/reject (original); "rules": equivalence class = which rules pass
  TOP      cells taken from the shortlist score ranking (original helper uses 6)
  EMPTY    what to read when all samples fall in one region: "shortlist" (original, best
           shortlist cell), "split" (cell whose sampled patterns disagree most) or "rulevar"
           (largest sum over live rules of p(1-p), p = share of samples reading 1 there)
  EMPTYCOL prior for a column with nothing read: "block" (original: one random block of one
           random pattern) or "borrow" (copy the same sample of a random column that has reads;
           columns are exchangeable a priori). Borrowed columns are redrawn every step.
  PERSIST  False: fresh samples every step (original). True: keep each column's samples and
           redraw a column only when one of its cells is read. Columns are independent in the
           model, so each step's sample has the same distribution; classic EC2 keeps one
           hypothesis set too. The cache holds only samples drawn from labels already read.
"""
import numpy as np

from agentic_research.evolve_v4.helpers import response_patterns

SAMPLES = 192
COPY = 0.97
FLIP = "rank"
REGION = "rules"
TOP = 6
EMPTY = "split"
PERSIST = True
EMPTYCOL = "borrow"
SEED = 1000003      # random stream base (original value); varied only to measure sampling noise

_STATE = {}


def outer(possible, lower, upper):
    valid = possible & np.isfinite(lower)
    left = np.where(valid, lower, np.inf).min(axis=-1)
    right = np.where(valid, upper, -np.inf).max(axis=-1)
    return left, right


def shortlist(partial, possible, lower, upper, left, right, live):
    """Same as helpers.shortlist, with the number of top-scored cells as a knob."""
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
    top = order[:min(TOP, int(unobserved.sum()))]
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


def sample_hypotheses(pat_idx, known, rng, n, n_pat):
    """Vectorised sampler. Per column, unknown ranks copy the left/right observed pattern around a
    random flip point; with probability 1-COPY a cell takes a uniform random pattern."""
    cols, ranks = pat_idx.shape
    idx = np.arange(ranks)
    out = np.broadcast_to(pat_idx, (n, cols, ranks)).copy()
    # nearest observed rank to the left / right (-1 if none)
    lpos = np.where(known, idx, -1)
    lpos = np.maximum.accumulate(lpos, axis=1)
    rpos = np.where(known, idx, ranks)
    rpos = np.minimum.accumulate(rpos[:, ::-1], axis=1)[:, ::-1]
    has_obs = known.any(1)
    lp = np.where(lpos >= 0, np.take_along_axis(pat_idx, np.maximum(lpos, 0), 1), 0)
    hp = np.where(rpos < ranks, np.take_along_axis(pat_idx, np.minimum(rpos, ranks - 1), 1), lp)
    left_none = lpos < 0
    coin = rng.random((cols, ranks)) < .5                   # shared over samples, as the original
    lp = np.where(left_none, np.where(coin, hp, 0), lp)
    lo_edge = np.where(lpos >= 0, lpos, -1)
    span = np.where(rpos < ranks, rpos, ranks) - lo_edge
    if FLIP == "gap":
        u = rng.random((n, cols, ranks + 1))
        u = np.take_along_axis(u, np.broadcast_to(lo_edge + 1, (n, cols, ranks)), 2)
    else:
        u = rng.random((n, cols, ranks))
    flip = u * span + lo_edge
    val = np.where(idx > flip, hp, lp)
    noise = rng.random((n, cols, ranks)) > COPY
    val = np.where(noise, rng.integers(0, n_pat, (n, cols, ranks)), val)
    unk = ~known & has_obs[:, None]
    out = np.where(unk[None], val, out)
    # columns with nothing observed: one random block of one random pattern
    empty = np.flatnonzero(~has_obs)
    if len(empty):
        ends = np.sort(rng.integers(0, ranks + 1, (n, len(empty), 2)), -1)
        pat = rng.integers(0, n_pat, (n, len(empty)))
        block = (idx >= ends[..., :1]) & (idx < ends[..., 1:])
        out[:, empty, :] = np.where(block, pat[..., None], 0)
    return out


def first_starts(hyp, bits, nrules, ranks):
    """First-run start per (sample, column, rule) of fully observed hypotheses; ranks-1 = no run."""
    hb = bits[hyp]                                          # (S, cols, ranks) rule bitmasks
    pair = hb[..., :-1] & hb[..., 1:]
    has = ((pair[..., None] >> np.arange(nrules)) & 1).astype(bool)   # (S, cols, ranks-1, rules)
    return np.where(has.any(-2), has.argmax(-2), ranks - 1)


def bounds(start, lower, upper, cols_idx):
    """intercept interval of each sampled start: (S, cols, rules); an invalid start gets
    lower -inf / upper +inf (it can never certify a pass, see passes_from)."""
    lo = lower[:, 0, :][cols_idx[None, :, None], start]
    up = upper[:, 0, :][cols_idx[None, :, None], start]
    ok = np.isfinite(lo)
    return np.where(ok, lo, -np.inf), np.where(ok, up, np.inf)


def passes_from(lo, up):
    """guaranteed pass per (sample, rule): every column's start valid and the intervals overlap."""
    return (lo > -np.inf).all(1) & (lo.max(1) <= up.min(1))   # (S, rules)


def choose(view):
    partial, patterns = view.partial, response_patterns(view.rules)
    cols, _, ranks = partial.shape
    n_pat = len(patterns)
    known = partial[:, 0, :] != -1
    # pattern index of each observed cell = first pattern equal to its 9 labels (lookup by bits)
    bits = (patterns.astype(np.int64) << np.arange(patterns.shape[1])).sum(1)
    lut = np.zeros(1 << patterns.shape[1], dtype=np.int64)
    for i in range(n_pat - 1, -1, -1):                     # duplicates resolve to the first pattern
        lut[bits[i]] = i
    cell_bits = ((partial == 1).astype(np.int64) << np.arange(patterns.shape[1])[None, :, None]).sum(1)
    pat_idx = np.where(known, lut[cell_bits], 0)
    rng = np.random.default_rng(SEED + view.queries_so_far)
    nrules = patterns.shape[1]
    st = _STATE
    fresh = (not PERSIST or st.get("shape") != partial.shape or st.get("n") != SAMPLES
             or view.queries_so_far == 0 or (st["known"] & ~known).any())
    if fresh:
        hyp = sample_hypotheses(pat_idx, known, rng, SAMPLES, n_pat)
        start = first_starts(hyp, bits, nrules, ranks)
        lo, up = bounds(start, view.lower, view.upper, np.arange(cols))
    else:
        hyp, start, lo, up = st["hyp"], st["start"], st["lo"], st["up"]
        ch = np.flatnonzero((known != st["known"]).any(1))
        if len(ch):
            hyp[:, ch] = sample_hypotheses(pat_idx[ch], known[ch], rng, SAMPLES, n_pat)
            start[:, ch] = first_starts(hyp[:, ch], bits, nrules, ranks)
            lo[:, ch], up[:, ch] = bounds(start[:, ch], view.lower, view.upper, ch)
    if EMPTYCOL == "borrow" and known.any() and (~known.any(1)).any():
        emp, obs = np.flatnonzero(~known.any(1)), np.flatnonzero(known.any(1))
        src = obs[rng.integers(0, len(obs), (SAMPLES, len(emp)))]
        rows = np.arange(SAMPLES)[:, None]
        hyp[:, emp] = hyp[rows, src]
        start[:, emp] = start[rows, src]                    # same labels -> same first-run start
        lo[:, emp], up[:, emp] = bounds(start[:, emp], view.lower, view.upper, emp)
    if PERSIST:
        _STATE.update(shape=partial.shape, n=SAMPLES, known=known.copy(), hyp=hyp, start=start,
                      lo=lo, up=up)
    passes = passes_from(lo, up)
    key = passes if REGION == "rules" else passes.any(1, keepdims=True)
    _, region = np.unique(key, axis=0, return_inverse=True)
    region = region.ravel()                                   # 0..(classes present - 1)
    left, right = outer(view.possible, view.lower, view.upper)
    live = left.max(0) <= right.min(0)
    col, rank, priority = shortlist(partial, view.possible, view.lower, view.upper, left, right, live)
    v = hyp[:, col, rank].T                                   # (K, S)
    nK = len(col)
    cnt = np.bincount((np.arange(nK)[:, None] * n_pat + v).ravel(),
                      minlength=nK * n_pat).reshape(nK, n_pat).astype(float)
    all_dis = SAMPLES ** 2 - (cnt ** 2).sum(-1)
    nreg = region.max() + 1
    if nreg == 1:
        if EMPTY in ("split", "rulevar"):
            if EMPTY == "rulevar":
                lab = patterns[v].astype(float)                   # (K, S, rules)
                pr = lab.mean(1)
                score = (pr * (1 - pr) * live[None, :]).sum(-1)
            else:
                score = all_dis
            key = np.lexsort((priority, np.round(score, 9)))
            return int(col[key[-1]]), int(rank[key[-1]])
        return int(col[0]), int(rank[0])
    ckv = np.bincount(((np.arange(nK)[:, None] * nreg + region[None, :]) * n_pat + v).ravel(),
                      minlength=nK * nreg * n_pat).astype(float)
    nk = np.bincount(region, minlength=nreg).astype(float)
    within = (nk ** 2).sum() - (ckv.reshape(nK, -1) ** 2).sum(-1)
    gain = (all_dis - within) / (2.0 * SAMPLES ** 2)
    best, best_key = 0, None
    for i in range(len(col)):
        key = (round(float(gain[i]), 12), priority[i])
        if best_key is None or key > best_key:
            best, best_key = i, key
    return int(col[best]), int(rank[best])
