"""Exact first-K-run parameter intervals for one shared scalar gate.

Independent structural implementation: no imports from the production sweep.
Labels are fixed_i OR (guard_i AND compare(value_i, threshold)); fixed/guard
must be disjoint. The parameter domain is a closed subset of [0,+inf], with
literal +inf allowed to match threshold_profile.py. Scores are finite or NaN;
NaN requires guard=False. Returned intervals retain exact inequality closures.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Iterable


def _binary(xs: Iterable, name: str) -> list[bool]:
    try:
        a = list(xs)
    except TypeError as exc:
        raise ValueError(f'{name} must be a one-dimensional binary sequence') from exc
    out = []
    for x in a:
        try:
            if x != 0 and x != 1:
                raise ValueError(f'{name} must contain binary scalars')
            out.append(bool(x))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain binary scalars') from exc
    return out


def _inputs(values, guard, fixed, k, relation, domain):
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError('k must be a positive integer')
    if relation not in ('ge', 'lt'):
        raise ValueError("relation must be 'ge' or 'lt'")
    try:
        vals = [float(x) for x in values]
        endpoints = list(domain)
        if len(endpoints) != 2:
            raise ValueError('domain must have two endpoints')
        lo, hi = map(float, endpoints)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('values/domain must contain scalar numbers') from exc
    if math.isnan(lo) or math.isnan(hi) or lo < 0 or lo > hi:
        raise ValueError('domain must be ordered, nonnegative and non-NaN')
    g, f = _binary(guard, 'guard'), _binary(fixed, 'fixed')
    if len(vals) != len(g) or len(vals) != len(f):
        raise ValueError('values, guard and fixed lengths must match')
    if any(a and b for a, b in zip(g, f)):
        raise ValueError('guard and fixed must be disjoint')
    if any(math.isinf(v) for v in vals):
        raise ValueError('input scores must be finite or NaN')
    if any(a and math.isnan(v) for v, a in zip(vals, g)):
        raise ValueError('NaN scores require guard=False')
    return vals, g, f, int(k), lo, hi


def _intersection(lo, hi, lo_closed, hi_closed, domain_lo, domain_hi):
    """Intersection with closed domain, retaining literal +inf singleton."""
    if domain_lo > lo:
        lo, lo_closed = domain_lo, True
    if domain_hi < hi:
        hi, hi_closed = domain_hi, True
    if lo > hi or (lo == hi and not (lo_closed and hi_closed)):
        return None
    return dict(lo=float(lo), hi=float(hi), lo_closed=bool(lo_closed),
                hi_closed=bool(hi_closed))


def _activations(vals, guard, fixed, k, relation):
    # Sentinels encode fixed labels, including at the literal +inf endpoint.
    # A -inf threshold is disallowed: it would not preserve this encoding.
    if relation == 'ge':
        effective = [math.inf if f else (v if g else -math.inf)
                     for v, g, f in zip(vals, guard, fixed)]
        return [min(effective[s:s+k]) for s in range(len(vals)-k+1)]
    effective = [-math.inf if f else (v if g else math.inf)
                 for v, g, f in zip(vals, guard, fixed)]
    return [max(effective[s:s+k]) for s in range(len(vals)-k+1)]


def first_run_threshold_intervals(values, guard, fixed, k, relation, domain):
    """Return exactly the attainable first-run starts and threshold intervals.

    Each returned dict has start, lo, hi, lo_closed, hi_closed, activation and
    prefix_extreme. Impossible starts are omitted. Start zero is included and
    is the left-censored state for boundary extraction; force validity is not
    inspected here. Positive-width intervals and closed singleton endpoints
    are retained, including +inf if the caller requests it in the domain.

    For ge: q_s=min effective K scores; state s iff prefix_max<t<=q_s.
    For lt: q_s=max effective K scores; state s iff q_s<t<=prefix_min.
    Time O(nK), memory O(n). At empirical n=50 the simple independent formula
    is preferable to sharing the production event-update implementation.
    """
    vals, g, f, k, dlo, dhi = _inputs(values, guard, fixed, k, relation, domain)
    q = _activations(vals, g, f, k, relation)
    prefix = -math.inf if relation == 'ge' else math.inf
    out = []
    for s, value in enumerate(q):
        lo, hi = (prefix, value) if relation == 'ge' else (value, prefix)
        cell = _intersection(lo, hi, False, True, dlo, dhi)
        if cell is not None:
            out.append(dict(start=s, **cell, activation=float(value),
                            prefix_extreme=float(prefix)))
        prefix = max(prefix, value) if relation == 'ge' else min(prefix, value)
    return out


def no_run_threshold_interval(values, guard, fixed, k, relation, domain):
    """Return the no-K-run interval, or None when impossible in the domain."""
    vals, g, f, k, dlo, dhi = _inputs(values, guard, fixed, k, relation, domain)
    q = _activations(vals, g, f, k, relation)
    if relation == 'ge':
        # No run iff t > every K-window minimum.
        extreme = max(q, default=-math.inf)
        cell = _intersection(extreme, math.inf, False, True, dlo, dhi)
    else:
        # No run iff t <= every K-window maximum.
        extreme = min(q, default=math.inf)
        cell = _intersection(-math.inf, extreme, False, True, dlo, dhi)
    return None if cell is None else dict(start=None, **cell,
                                         extreme=float(extreme))


def contains(interval, threshold):
    """Membership helper respecting closed singleton and infinite endpoints."""
    t = float(threshold)
    if math.isnan(t):
        return False
    lo, hi = interval['lo'], interval['hi']
    return ((t > lo or (t == lo and interval['lo_closed'])) and
            (t < hi or (t == hi and interval['hi_closed'])))
