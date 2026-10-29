"""Exact first-K-run distributions/certificates; no monotonic force-label assumption."""
from __future__ import annotations
import numpy as np


def first_run(labels, k=2):
    a = np.asarray(labels, dtype=bool)
    n = a.shape[-1]
    out = np.full(a.shape[:-1], n, dtype=int)
    for s in range(n-k+1):
        hit = a[..., s:s+k].all(axis=-1)
        out = np.where((out == n) & hit, s, out)
    return out


def run_distribution(probabilities, k=2):
    """Independent Bernoulli prior, exact pushforward. Last category n means no run.

    Category n-k+1 through n-1 have zero mass. Known labels use probabilities 0/1.
    Independence is a predictive approximation, not a physical assumption or guarantee.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim < 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probabilities in [0,1]')
    n = p.shape[-1]
    if not 1 <= k <= n:
        raise ValueError('Invalid K')
    shape = p.shape[:-1]
    p = p.reshape(-1, n)
    state = np.zeros((len(p), k), dtype=float)
    state[:, 0] = 1.
    out = np.zeros((len(p), n+1), dtype=float)
    for t in range(n):
        q = p[:, t]
        if t >= k-1:
            out[:, t-k+1] = state[:, k-1] * q
        nxt = np.zeros_like(state)
        nxt[:, 0] = state.sum(axis=1) * (1-q)
        if k > 1:
            nxt[:, 1:] = state[:, :-1] * q[:, None]
        state = nxt
    out[:, n] = state.sum(axis=1)
    return out.reshape(shape + (n+1,))


def certificate(observed, k=2):
    """Necessary/sufficient for fixed first-run outcome with unrestricted unknown bits."""
    obs = np.asarray(observed)
    if not np.isin(obs, [-1, 0, 1]).all():
        raise ValueError('Observed state must use -1/0/1')
    earliest = first_run(obs != 0, k)  # all unknowns=1
    latest = first_run(obs == 1, k)    # all unknowns=0
    return earliest == latest, earliest, latest


def entropy(distribution):
    p = np.asarray(distribution)
    return -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1)


def information_gain(probabilities, unknown, k=2):
    """Sum of exact per-rule marginal entropy reductions for each shared query.

    This is NOT joint entropy over correlated rule outputs; each rule uses the
    independent Bernoulli prior and already acquired observations.
    """
    p = np.asarray(probabilities, dtype=float)
    ix = np.flatnonzero(unknown)
    if p.ndim != 2 or not len(ix):
        raise ValueError('Expected rules×positions and some unknown positions')
    variants = np.broadcast_to(p, (len(ix), 2) + p.shape).copy()
    for j, pos in enumerate(ix):
        variants[j, 0, :, pos] = 0.
        variants[j, 1, :, pos] = 1.
    h = entropy(run_distribution(variants, k))  # [queries,2,rules]
    prob = p[:, ix].T
    base = entropy(run_distribution(p, k)).sum()
    expected = ((1-prob)*h[:, 0] + prob*h[:, 1]).sum(axis=1)
    result = np.full(p.shape[-1], -np.inf)
    result[ix] = np.maximum(0., base-expected)
    return result


def certificate_lower_bound(labels, k=2):
    """Optimal per-rule certificate size; max across rules is a valid joint lower bound.

    Oracle-only hindsight comparator, not an executable acquisition policy.
    Greedy rightmost actual zero hits each earlier run window; K positive labels
    certify a finite first run. The joint multi-rule optimum is not claimed.
    """
    y = np.atleast_2d(np.asarray(labels, dtype=bool))
    n = y.shape[-1]
    costs = []
    for row in y:
        start = int(first_run(row, k))
        chosen = set()
        for s in range(start if start < n else n-k+1):
            if any(s <= q < s+k for q in chosen):
                continue
            candidates = np.flatnonzero(~row[s:s+k]) + s
            if not len(candidates):
                raise AssertionError('Earlier first run contradicts true start')
            chosen.add(int(candidates[-1]))
        if start < n:
            chosen.update(range(start, start+k))
        costs.append(len(chosen))
    return max(costs), costs
