"""Independent literal two-feature rule and first-run compatibility oracle.

No import of production solver or its atom construction. Each observation is
classified anew at each pair of scalar thresholds, then all K-long runs are
searched from the start of the ordered force sequence.
"""
import math


def labels(column, u, v):
    return [int(bool(f) or (bool(g) and math.isfinite(p) and math.isfinite(r)
                    and p >= u and r < v))
            for f,g,p,r in zip(column['fixed'], column['guard'],
                               column['periodicity'], column['ratio'])]


def first_run(bits, k):
    for i in range(len(bits)-k+1):
        if all(bits[i:i+k]):
            return i
    return None


def boundary(column, bits, k):
    s = first_run(bits, k)
    if s is None:
        return dict(valid=False, start=s, cause='no_run')
    if s == 0:
        return dict(valid=False, start=s, cause='left_censored')
    f0, f1 = column['force'][s-1:s+1]
    if not (math.isfinite(f0) and math.isfinite(f1) and 0 < f0 < f1):
        return dict(valid=False, start=s, cause='nonpositive_or_tied')
    return dict(valid=True, start=s, low=math.log(f0), high=math.log(f1),
                log_beta=math.log(column['beta']))


def evaluate(columns, u, v, k, slopes, epsilons):
    result = dict(n_all_edits=0, n_eligible_edits=0, n_all_changed_columns=0,
                  n_changed_columns=0, invalid_eligible_columns=0,
                  no_run_eligible_columns=0, left_censored_eligible_columns=0,
                  nonpositive_or_tied_eligible_columns=0)
    boundaries = []
    result['labels'] = []
    for c in columns:
        bits = labels(c, u, v)
        result['labels'].append(bits)
        edits = sum(a != b for a,b in zip(bits,c['baseline']))
        result['n_all_edits'] += edits
        result['n_all_changed_columns'] += bool(edits)
        if c['eligible']:
            result['n_eligible_edits'] += edits
            result['n_changed_columns'] += bool(edits)
            b = boundary(c,bits,k)
            boundaries.append(b)
            if not b['valid']:
                result['invalid_eligible_columns'] += 1
                result[b['cause']+'_eligible_columns'] += 1
    valid = bool(boundaries) and all(b['valid'] for b in boundaries)
    result['lower'], result['upper'], result['gap'], result['compatible'] = [],[],[],[]
    for p in slopes:
        lo=max((b['low'] - p*b['log_beta'] for b in boundaries),default=math.nan) if valid else math.nan
        hi=min((b['high'] - p*b['log_beta'] for b in boundaries),default=math.nan) if valid else math.nan
        result['lower'].append(lo);result['upper'].append(hi)
        result['gap'].append(lo-hi if valid else math.nan)
        result['compatible'].append([valid and lo-e <= hi+e for e in epsilons])
    return result


def independent_probes(values, domain):
    lo, hi = domain
    points = sorted({lo,hi}.union(float(x) for x in values if math.isfinite(x) and lo <= x <= hi))
    probes = list(points)
    for a,b in zip(points,points[1:]):
        mid = a/2+b/2
        if a < mid < b:
            probes.append(mid)
        right=math.nextafter(a, math.inf)
        if a < right <= b:
            probes.append(right)
    return sorted(set(probes))


def eligible(c, k=2):
    return boundary(c,c['baseline'],k)['valid']
