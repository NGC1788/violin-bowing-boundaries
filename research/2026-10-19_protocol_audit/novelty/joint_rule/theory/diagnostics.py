"""Post hoc exact fixed-pair obstruction search for the joint rule.

Prepared after the empirical joint-threshold results were exposed. A pair is
certifying only if that SAME pair rules out the candidate law on EVERY event
cell. This is not a selection of different extremizing pairs on each cell.
Only compact derived input is read; neither source/profile nor raw data edited.
"""
from pathlib import Path
import gzip
import hashlib
import itertools
import json
import math
import time
import numpy as np
from rectangle_oracle import first_run_regions, contains

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOMAIN_U = (.7, .9)
DOMAIN_V = (.3, .7)
EPSILON = .2
SLOPES = (-2., -1.2)


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def edges(columns):
    us, vs = set(), set()
    for c in columns:
        for x, y, active in zip(c['periodicity'], c['ratio'], c['guard']):
            if not active: continue
            if y < DOMAIN_V[1] and DOMAIN_U[0] <= x < DOMAIN_U[1]: us.add(x)
            if x >= DOMAIN_U[0] and DOMAIN_V[0] <= y < DOMAIN_V[1]: vs.add(y)
    return np.array(sorted(us) + [DOMAIN_U[1]]), np.array(sorted(vs) + [DOMAIN_V[1]])


def interval_mask(interval, axis):
    low = axis >= interval['low'] if interval['low_closed'] else axis > interval['low']
    return low & (axis <= interval['high'])


def column_states(c, u, v):
    regions = first_run_regions(c['periodicity'], c['ratio'], c['guard'], c['fixed'],
                                2, DOMAIN_U, DOMAIN_V)
    shape = (len(u), len(v))
    states = np.full(shape, -99, dtype=np.int16)
    coverage = np.zeros(shape, dtype=np.uint8)
    for r in regions:
        ii, jj = np.flatnonzero(interval_mask(r['u'], u)), np.flatnonzero(interval_mask(r['v'], v))
        indices = np.ix_(ii, jj)
        states[indices] = -1 if r['start'] is None else r['start']
        coverage[indices] += 1
    assert np.all(coverage == 1), c['id']
    return states, regions


def boundary_arrays(c, states):
    f = np.asarray(c['force'], float)
    lo = f[np.maximum(states - 1, 0)]
    hi = f[np.maximum(states, 0)]
    valid = (states > 0) & (lo > 0) & (hi > lo)
    with np.errstate(divide='ignore', invalid='ignore'):
        return valid, np.log(lo), np.log(hi)


def direct_states(c, u, v):
    """Independent of region construction: classify all windows anew."""
    x, y = np.asarray(c['periodicity'], float), np.asarray(c['ratio'], float)
    guard, fixed = np.asarray(c['guard'], bool), np.asarray(c['fixed'], bool)
    states = np.empty((len(u), len(v)), dtype=np.int16)
    for ui, uu in enumerate(u):
        bits = fixed[None, :] | (guard[None, :] & (x >= uu)[None, :] & (y[None, :] < v[:, None]))
        runs = bits[:, :-1] & bits[:, 1:]
        has = np.any(runs, axis=1)
        states[ui] = np.where(has, np.argmax(runs, axis=1), -1)
    return states


def intersect(a, b):
    lo, hi = max(a['low'], b['low']), min(a['high'], b['high'])
    lc = ((lo > a['low'] or a['low_closed']) and (lo > b['low'] or b['low_closed']))
    if lo > hi or (lo == hi and not lc): return None
    return dict(low=lo, high=hi, low_closed=lc, high_closed=True)


def force_witness(c, start, p):
    if start is None: return dict(valid=False, start=None, cause='no_run')
    if start == 0: return dict(valid=False, start=0, cause='left_censored')
    flo, fhi = c['force'][start - 1:start + 1]
    if not 0 < flo < fhi:
        return dict(valid=False, start=start, cause='nonpositive_or_tied', force=[flo, fhi])
    return dict(valid=True, start=start, force=[flo, fhi],
                lower=math.log(flo) - p * math.log(c['beta']),
                upper=math.log(fhi) - p * math.log(c['beta']))


def explanation(a, b, ra, rb, p):
    wa, wb = force_witness(a, ra['start'], p), force_witness(b, rb['start'], p)
    if not wa['valid']: reason, witness = 'first_invalid', a['id']
    elif not wb['valid']: reason, witness = 'second_invalid', b['id']
    elif wa['lower'] - EPSILON > wb['upper'] + EPSILON:
        reason, witness = 'first_above_second', [a['id'], b['id']]
    elif wb['lower'] - EPSILON > wa['upper'] + EPSILON:
        reason, witness = 'second_above_first', [b['id'], a['id']]
    else:
        raise AssertionError('claimed pair contains compatible cell')
    return dict(first=wa, second=wb, reason=reason, witness=witness)


def pair_cover(a, b, regions_a, regions_b, p):
    cover = []
    for ra in regions_a:
        for rb in regions_b:
            uu, vv = intersect(ra['u'], rb['u']), intersect(ra['v'], rb['v'])
            if uu is None or vv is None: continue
            cover.append(dict(u=uu, v=vv, **explanation(a, b, ra, rb, p)))
    return cover, check_cover(cover)


def check_cover(cover):
    # Endpoints plus immediate successors make strict/closed boundary checking explicit.
    uu, vv = {DOMAIN_U[0], DOMAIN_U[1]}, {DOMAIN_V[0], DOMAIN_V[1]}
    for r in cover:
        for axis, container in [('u', uu), ('v', vv)]:
            container.update((r[axis]['low'], r[axis]['high']))
            nextpoint = math.nextafter(r[axis]['low'], math.inf)
            if nextpoint <= r[axis]['high']: container.add(nextpoint)
    u, v = np.array(sorted(uu)), np.array(sorted(vv))
    covered = np.zeros((len(u), len(v)), dtype=np.uint8)
    for r in cover:
        covered[np.ix_(np.flatnonzero(interval_mask(r['u'], u)),
                       np.flatnonzero(interval_mask(r['v'], v)))] += 1
    assert np.all(covered == 1), 'certificate does not partition the closed domain'
    return int(covered.size)


def triple_cover(columns, region_lists, p):
    partial = [{'u': dict(low=DOMAIN_U[0], high=DOMAIN_U[1], low_closed=True, high_closed=True),
                'v': dict(low=DOMAIN_V[0], high=DOMAIN_V[1], low_closed=True, high_closed=True), 'starts': []}]
    for regions in region_lists:
        new = []
        for old in partial:
            for r in regions:
                uu, vv = intersect(old['u'], r['u']), intersect(old['v'], r['v'])
                if uu is not None and vv is not None:
                    new.append(dict(u=uu, v=vv, starts=old['starts'] + [r['start']]))
        partial = new
    cover = []
    for r in partial:
        witnesses = [dict(id=c['id'], **force_witness(c, start, p))
                     for c, start in zip(columns, r['starts'])]
        bad = next((b for b in witnesses if not b['valid']), None)
        if bad is not None:
            reason, ids = 'invalid', [bad['id']]
        else:
            lower = max(witnesses, key=lambda b: b['lower'])
            upper = min(witnesses, key=lambda b: b['upper'])
            assert lower['lower'] - EPSILON > upper['upper'] + EPSILON
            reason, ids = 'separated', [lower['id'], upper['id']]
        cover.append(dict(u=r['u'], v=r['v'], boundaries=witnesses, reason=reason, witness=ids))
    return cover, check_cover(cover)


def main():
    started = time.monotonic()
    path = ROOT / 'input_joint.json.gz'
    with gzip.open(path, 'rt') as fh: raw = json.load(fh)
    reference_path = ROOT / 'prior_arbitrary_reference.json'
    reference = json.loads(reference_path.read_text())
    geometry = {(r['session'] + '|' + r['folder'], float(r['p'])): bool(r['restore']['feasible'])
                for r in reference['records'] if r['k'] == 2 and r['epsilon'] == EPSILON and r['mode'] == 'both'}
    groups = sorted({c['group'] for c in raw['columns']})
    results, certificates, triple_certificates = [], [], []
    total_cells = total_pairs = total_triples = replay_cells = cover_checks = 0
    for group in groups:
        allcols = [c for c in raw['columns'] if c['group'] == group]
        cols = sorted((c for c in allcols if c['eligible']), key=lambda c: (c['column'], c['id']))
        u, v = edges(allcols)
        total_cells += len(u) * len(v)
        states_regions = [column_states(c, u, v) for c in cols]
        states = [a[0] for a in states_regions]
        bs = [boundary_arrays(c, st) for c, st in zip(cols, states)]
        valids = np.array([b[0] for b in bs])
        lowers = np.array([b[1] for b in bs])
        uppers = np.array([b[2] for b in bs])
        ub, vb = np.searchsorted(u, .8), np.searchsorted(v, .5)
        assert valids[:, ub, vb].all(), group
        # Baseline makes each individual eligible-column constraint satisfiable.
        assert np.all(lowers[:, ub, vb] <= uppers[:, ub, vb])
        for p in SLOPES:
            lb = lowers - p * np.log([c['beta'] for c in cols])[:, None, None]
            hb = uppers - p * np.log([c['beta'] for c in cols])[:, None, None]
            full_good = valids.all(axis=0) & (np.max(lb, axis=0) - EPSILON <= np.min(hb, axis=0) + EPSILON)
            pairs = []
            for i, j in itertools.combinations(range(len(cols)), 2):
                total_pairs += 1
                bothvalid = valids[i] & valids[j]
                separate = (lb[i] - EPSILON > hb[j] + EPSILON) | (lb[j] - EPSILON > hb[i] + EPSILON)
                if np.any(bothvalid & ~separate): continue
                pairs.append(dict(indices=[cols[i]['column'], cols[j]['column']],
                                  ids=[cols[i]['id'], cols[j]['id']],
                                  eligible_indices=[i, j], beta=[cols[i]['beta'], cols[j]['beta']],
                                  numerical_cells=int(np.count_nonzero(bothvalid & separate)),
                                  invalid_cells=int(np.count_nonzero(~bothvalid))))
            assert not (full_good.any() and pairs), (group, p)
            row = dict(group=group, p=p, epsilon=EPSILON, cells=int(full_good.size),
                       arbitrary_geometry_feasible=geometry[group,p],
                       eligible_columns=len(cols), compatible_cells=int(np.count_nonzero(full_good)),
                       certifying_pair_count=len(pairs), all_certifying_pairs=pairs,
                       minimum_obstruction_cardinality=2 if pairs else None,
                       obstruction_cardinality_lower_bound=3 if not pairs and not full_good.any() else None,
                       chosen_pair=None, all_certifying_triples=[], certifying_triple_count=0, chosen_triple=None)
            if pairs:
                chosen = pairs[0]
                i, j = chosen['eligible_indices']
                # Direct labels on every group event cell verify the selected fixed pair.
                for ci in (i, j):
                    direct = direct_states(cols[ci], u, v)
                    assert np.array_equal(direct, states[ci]), (group, p, ci)
                    replay_cells += direct.size
                cover, nchecks = pair_cover(cols[i], cols[j], states_regions[i][1], states_regions[j][1], p)
                cover_checks += nchecks
                baseline = [force_witness(cols[ci], int(states[ci][ub, vb]), p) for ci in (i, j)]
                cert = dict(group=group, p=p, epsilon=EPSILON, chosen_pair=chosen,
                            baseline_pair_threshold=[.8, .5], baseline_boundaries=baseline,
                            n_rectangles=len(cover), coverage_probe_points=nchecks, rectangles=cover)
                certificates.append(cert)
                row['chosen_pair'] = dict(chosen, n_certificate_rectangles=len(cover),
                                          baseline_boundaries=baseline)
            elif not full_good.any():
                triples = []
                for ii in itertools.combinations(range(len(cols)), 3):
                    total_triples += 1
                    a, b, c = ii
                    allvalid = valids[a] & valids[b] & valids[c]
                    lower = np.maximum(np.maximum(lb[a], lb[b]), lb[c])
                    upper = np.minimum(np.minimum(hb[a], hb[b]), hb[c])
                    separated = lower - EPSILON > upper + EPSILON
                    if np.any(allvalid & ~separated): continue
                    triples.append(dict(indices=[cols[i]['column'] for i in ii],
                                        ids=[cols[i]['id'] for i in ii], eligible_indices=list(ii),
                                        beta=[cols[i]['beta'] for i in ii],
                                        numerical_cells=int(np.count_nonzero(allvalid & separated)),
                                        invalid_cells=int(np.count_nonzero(~allvalid))))
                row['all_certifying_triples'] = triples
                row['certifying_triple_count'] = len(triples)
                if triples:
                    chosen = triples[0]
                    ii = chosen['eligible_indices']
                    for ci in ii:
                        direct = direct_states(cols[ci], u, v)
                        assert np.array_equal(direct, states[ci]), (group, p, ci)
                        replay_cells += direct.size
                    cover, nchecks = triple_cover([cols[i] for i in ii], [states_regions[i][1] for i in ii], p)
                    cover_checks += nchecks
                    baseline = [force_witness(cols[i], int(states[i][ub,vb]), p) for i in ii]
                    row['chosen_triple'] = dict(chosen, n_certificate_rectangles=len(cover), baseline_boundaries=baseline)
                    row['minimum_obstruction_cardinality'] = 3
                    row['obstruction_cardinality_lower_bound'] = None
                    triple_certificates.append(dict(group=group, p=p, epsilon=EPSILON,
                        chosen_triple=chosen, baseline_boundaries=baseline, n_rectangles=len(cover),
                        coverage_probe_points=nchecks, rectangles=cover))
                else:
                    row['obstruction_cardinality_lower_bound'] = 4
            results.append(row)
        print(group.split('|')[0], group.rsplit('_', 1)[-1],
              [(r['p'], r['certifying_pair_count'], r['certifying_triple_count']) for r in results[-2:]], flush=True)
    summary = {}
    for p in SLOPES:
        rr = [r for r in results if r['p'] == p]
        summary[str(p)] = dict(groups=len(rr), incompatible_groups=sum(r['compatible_cells'] == 0 for r in rr),
                               groups_with_fixed_pair=sum(r['certifying_pair_count'] > 0 for r in rr),
                               groups_with_minimum_triple=sum(r['minimum_obstruction_cardinality'] == 3 for r in rr),
                               groups_without_pair=sum(r['compatible_cells'] == 0 and r['certifying_pair_count'] == 0 for r in rr),
                               all_certifying_pairs=sum(r['certifying_pair_count'] for r in rr),
                               all_certifying_triples=sum(r['certifying_triple_count'] for r in rr),
                               chosen_pairs_always_numerical=sum(r['chosen_pair'] is not None and r['chosen_pair']['invalid_cells'] == 0 for r in rr),
                               some_pair_always_numerical=sum(any(pair['invalid_cells'] == 0 for pair in r['all_certifying_pairs']) for r in rr),
                               by_arbitrary_geometry={str(flag): dict(
                                   groups=sum(r['arbitrary_geometry_feasible'] == flag for r in rr),
                                   incompatible=sum(r['arbitrary_geometry_feasible'] == flag and r['compatible_cells'] == 0 for r in rr),
                                   minimum_pairs=sum(r['arbitrary_geometry_feasible'] == flag and r['minimum_obstruction_cardinality'] == 2 for r in rr),
                                   minimum_triples=sum(r['arbitrary_geometry_feasible'] == flag and r['minimum_obstruction_cardinality'] == 3 for r in rr))
                                   for flag in (False, True)})
    out = dict(post_hoc=True,
               status='PASS',
               scope='Explanation developed after exposed joint results; same fixed pair must obstruct every cell. Not new confirmation or a generic MUS claim.',
               domain=dict(u=DOMAIN_U, v=DOMAIN_V), epsilon=EPSILON,
               local_cells=total_cells, tested_pairs=total_pairs, tested_triples=total_triples,
               selected_subset_direct_state_comparisons=replay_cells,
               selected_certificate_cover_probes=cover_checks, summary=summary, groups=results,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [path, reference_path, HERE/'rectangle_oracle.py', Path(__file__)]},
               duration_seconds=time.monotonic()-started)
    write(HERE/'obstructions.json', out)
    write(HERE/'obstruction_certificates.json', dict(post_hoc=True, domain=out['domain'], certificates=certificates,
                                                   triple_certificates=triple_certificates))
    print(json.dumps(clean({k:out[k] for k in ('status','local_cells','tested_pairs','tested_triples','summary','duration_seconds')})))


if __name__ == '__main__':
    main()
