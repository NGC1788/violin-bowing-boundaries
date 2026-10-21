"""Independent, standard-library first-K-run region decomposition.

No empirical I/O, no production solver import. This maps the entire finite
closed (u,v) domain into disjoint rectangles of constant first-run state.
It deliberately does not compute edit costs, forces, or scientific verdicts.
"""
import math


def _bools(values, name):
    answer = list(values)
    if any(v not in (False, True, 0, 1) for v in answer):
        raise ValueError(name + ' must contain only booleans')
    return [bool(v) for v in answer]


def _closed_domain(pair):
    if len(pair) != 2:
        raise ValueError('domain requires two endpoints')
    low, high = map(float, pair)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError('domain must be finite, closed and ordered')
    return low, high


def _interval(low, high, low_closed=True):
    if low > high or (low == high and not low_closed):
        return None
    return {'low': low, 'high': high, 'low_closed': low_closed,
            'high_closed': True}


def _atoms(events, domain):
    low, high = domain
    points = sorted({p for p in events if low <= p < high})
    answer, previous, closed = [], low, True
    for point in points:
        answer.append(_interval(previous, point, closed))
        previous, closed = point, False
    answer.append(_interval(previous, high, closed))
    return answer


def first_run_regions(x, y, guard, fixed, k, u_domain, v_domain):
    """Return disjoint rectangles covering the supplied closed domain.

    Rows have keys u, v (interval dicts) and start (int or None for no run).
    Start zero is retained; force validity belongs to the caller.
    """
    x, y = list(x), list(y)
    guard, fixed = _bools(guard, 'guard'), _bools(fixed, 'fixed')
    if not (len(x) == len(y) == len(guard) == len(fixed)):
        raise ValueError('length mismatch')
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    if any(g and f for g, f in zip(guard, fixed)):
        raise ValueError('guard and fixed must be disjoint')
    for i, g in enumerate(guard):
        if g:
            x[i], y[i] = float(x[i]), float(y[i])
            if not (math.isfinite(x[i]) and math.isfinite(y[i])):
                raise ValueError('mutable features must be finite')
    u_domain, v_domain = _closed_domain(u_domain), _closed_domain(v_domain)
    runs = []
    for start in range(len(x) - k + 1):
        positions = range(start, start + k)
        if any(not guard[i] and not fixed[i] for i in positions):
            runs.append(None)
        else:
            mutable = [i for i in positions if guard[i]]
            runs.append((min((x[i] for i in mutable), default=math.inf),
                         max((y[i] for i in mutable), default=-math.inf)))
    regions = []
    for slab in _atoms((r[0] for r in runs if r), u_domain):
        u = slab['high']
        prior_q = math.inf
        for start, run in enumerate(runs):
            if run is None or run[0] < u:
                continue
            q = run[1]
            if q < prior_q:
                lower = max(v_domain[0], q)
                upper = min(v_domain[1], prior_q)
                vi = _interval(lower, upper, low_closed=v_domain[0] > q)
                if vi is not None:
                    regions.append({'u': dict(slab), 'v': vi, 'start': start})
                prior_q = q
        no_run = _interval(v_domain[0], min(v_domain[1], prior_q))
        if no_run is not None:
            regions.append({'u': dict(slab), 'v': no_run, 'start': None})
    return regions


def contains(interval, value):
    return ((value > interval['low'] or
             (interval['low_closed'] and value == interval['low'])) and
            (value < interval['high'] or
             (interval['high_closed'] and value == interval['high'])))


def locate(regions, u, v):
    """Return all matching states; a correct partition has exactly one."""
    return [r['start'] for r in regions if contains(r['u'], u) and contains(r['v'], v)]
