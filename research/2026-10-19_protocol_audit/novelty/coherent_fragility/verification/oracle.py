"""Independent, deliberately literal threshold and boundary reference code.

This file does not import the production threshold solver, first-run DP, or
boundary extraction modules. It evaluates every window afresh at each scalar
threshold and searches for the first run directly with Python ``all``.
"""
import math


def first_run(bits, k):
    for i in range(len(bits) - k + 1):
        if all(bits[i:i+k]):
            return i
    return None


def literal_labels(fixed, guard, values, threshold, direction):
    labels = []
    for f, g, value in zip(fixed, guard, values):
        finite = math.isfinite(value)
        varied = finite and (value >= threshold if direction == 'ge' else value < threshold)
        labels.append(int(bool(f) or (bool(g) and varied)))
    return labels


def literal_column(force, labels, beta, k, p, epsilon):
    start = first_run(labels, k)
    if start is None or start == 0:
        return {'valid': False, 'start': start, 'reason': 'no_run' if start is None else 'left_censored'}
    low, high = force[start-1], force[start]
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        return {'valid': False, 'start': start, 'reason': 'invalid_force_bracket'}
    return {'valid': True, 'start': start,
            'lo': math.log(low) - p * math.log(beta) - epsilon,
            'hi': math.log(high) - p * math.log(beta) + epsilon}


def threshold_atoms(values, domain):
    """Literal domain endpoints, finite feature endpoints and open-cell probes.

    Finite probes prefer stable half sums and fall back to nextafter. A cell
    with no representable float interior has no distinct machine threshold,
    but mathematical labels there are the right-limit labels of its left
    endpoint; the primary tests intentionally avoid such adjacent floats.
    """
    low, high = domain
    points = {low, high}
    points.update(float(x) for x in values if math.isfinite(x) and low <= x <= high)
    ordered = sorted(points)
    atoms = []
    for i, x in enumerate(ordered):
        if math.isfinite(x):
            atoms.append({'threshold': x, 'kind': 'point', 'lo': x, 'hi': x})
        if i + 1 == len(ordered):
            continue
        y = ordered[i+1]
        if not (x < y):
            continue
        if math.isinf(x) and math.isinf(y):
            probe = 0.
        elif math.isinf(x):
            probe = y - max(1., abs(y))
        elif math.isinf(y):
            probe = x + max(1., abs(x))
        else:
            probe = x / 2. + y / 2.
        if math.isfinite(probe) and x < probe < y:
            atoms.append({'threshold': probe, 'kind': 'open', 'lo': x, 'hi': y})
    return atoms


def literal_folder(columns, threshold, direction, k, p, epsilon):
    """columns use force,beta,fixed,guard,values,labels,eligible keys."""
    total_cost = eligible_cost = changed_cols = 0
    boundaries = []
    labels_by_column = []
    for column in columns:
        edited = literal_labels(column['fixed'], column['guard'], column['values'], threshold, direction)
        labels_by_column.append(edited)
        cost = sum(a != b for a, b in zip(edited, column['labels']))
        total_cost += cost
        changed_cols += bool(cost)
        if column['eligible']:
            eligible_cost += cost
            boundaries.append(literal_column(column['force'], edited, column['beta'], k, p, epsilon))
    valid = bool(boundaries) and all(b['valid'] for b in boundaries)
    lo = max((b['lo'] for b in boundaries), default=math.nan) if valid else math.nan
    hi = min((b['hi'] for b in boundaries), default=math.nan) if valid else math.nan
    return {'valid': valid, 'compatible': valid and lo <= hi,
            'lo': lo, 'hi': hi, 'gap': lo-hi if valid else math.inf,
            'cost_all': total_cost, 'cost_eligible': eligible_cost,
            'changed_columns': changed_cols, 'boundaries': boundaries,
            'labels': labels_by_column}
