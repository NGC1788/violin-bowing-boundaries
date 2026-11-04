"""Literature baseline: DIRECT-style edge-cutting greedy for decision region determination.

Hypotheses are sampled completions of the unknown labels (labels only, never view.revealed).
Decision regions are the certificate outcomes (accept / reject). Reading a cell cuts every
hypothesis pair in different regions that disagree on that cell; pick the cell with the
largest expected cut weight.
"""
import numpy as np

from agentic_research.evolve_v4.helpers import certificates, outer, possible_starts, response_patterns, shortlist

SAMPLES = 96
COPY = .9


def sample_hypotheses(view, pat_idx, known, rng, n):
    """Per column, unknown ranks take the left/right observed pattern with one monotone flip."""
    cols, ranks = pat_idx.shape
    out = np.broadcast_to(pat_idx, (n, cols, ranks)).copy()
    n_pat = int(pat_idx.max()) + 1
    for c in range(cols):
        obs = np.flatnonzero(known[c])
        unk = np.flatnonzero(~known[c])
        if len(unk) == 0:
            continue
        if len(obs) == 0:
            for s in range(n):
                lo, hi = sorted(rng.integers(0, ranks + 1, 2))
                out[s, c, :] = 0
                out[s, c, lo:hi] = rng.integers(0, n_pat)
            continue
        # vectorised over unknown ranks per sample using gaps
        left = np.full(ranks, -1)
        right = np.full(ranks, -1)
        last = -1
        for r in range(ranks):
            if known[c, r]:
                last = r
            left[r] = last
        last = -1
        for r in range(ranks - 1, -1, -1):
            if known[c, r]:
                last = r
            right[r] = last
        for r in unk:
            l, h = left[r], right[r]
            lp = pat_idx[c, l] if l >= 0 else 0
            hp = pat_idx[c, h] if h >= 0 else lp
            if l < 0:
                lp = hp if rng.random() < .5 else 0
            span = (h if h >= 0 else ranks) - (l if l >= 0 else -1)
            flip = rng.random(n) * span + (l if l >= 0 else -1)
            use_high = r > flip
            val = np.where(use_high, hp, lp)
            noise = rng.random(n) > COPY
            val = np.where(noise, rng.integers(0, n_pat, n), val)
            out[:, c, r] = val
    return out


def choose(view):
    partial, patterns = view.partial, response_patterns(view.rules)
    cols, _, ranks = partial.shape
    known = partial[:, 0, :] != -1
    # pattern index of each observed cell
    pat_idx = np.zeros((cols, ranks), dtype=np.int64)
    obs_c, obs_r = np.nonzero(known)
    if len(obs_c):
        cells = partial[obs_c, :, obs_r]
        pat_idx[obs_c, obs_r] = (cells[:, None, :] == patterns[None]).all(-1).argmax(1)
    rng = np.random.default_rng(1000003 + view.queries_so_far)
    hyp = sample_hypotheses(view, pat_idx, known, rng, SAMPLES)
    # decision region of each hypothesis
    region = np.empty(SAMPLES, dtype=bool)
    for s in range(SAMPLES):
        full = patterns[hyp[s]].transpose(0, 2, 1)  # (cols, rules, ranks)
        region[s] = bool(certificates(possible_starts(full), view.lower, view.upper)[0])
    left, right = outer(view.possible, view.lower, view.upper)
    live = left.max(0) <= right.min(0)
    col, rank, priority = shortlist(partial, view.possible, view.lower, view.upper, left, right, live)
    if region.all() or (~region).all():
        return int(col[0]), int(rank[0])
    w = 1.0 / SAMPLES
    best, best_key = 0, None
    n_pat = len(patterns)
    for i, (c, r) in enumerate(zip(col, rank)):
        v = hyp[:, c, r]
        t = np.bincount(v[region], minlength=n_pat) * w
        f = np.bincount(v[~region], minlength=n_pat) * w
        gain = t.sum() * f.sum() - float(t @ f)  # weight of cross-region pairs disagreeing on the cell
        key = (round(gain, 12), priority[i])
        if best_key is None or key > best_key:
            best, best_key = i, key
    return int(col[best]), int(rank[best])
