"""Independent finite-candidate tests for restricting open cost envelopes."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'research' / '2026-10-14' / 'width_bound_sensitivity.py'
SPEC = importlib.util.spec_from_file_location('width_bound_sensitivity', PATH)
bound = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bound)


def direct_cost(intervals, k):
    return min(cost for lower, upper, cost in intervals if lower < k < upper)


def direct_cells(intervals):
    points = sorted({x for lower, upper, _ in intervals for x in (lower, upper)})
    cells = []
    for lower, upper in zip(points, points[1:]):
        k = (0. if np.isneginf(lower) and np.isposinf(upper) else
             upper-1. if np.isneginf(lower) else lower+1. if np.isposinf(upper) else (lower+upper)/2)
        cells.append({'lower': None if np.isneginf(lower) else lower,
                      'upper': None if np.isposinf(upper) else upper,
                      'cost': direct_cost(intervals, k)})
    return cells


@pytest.mark.parametrize('intervals', [
    [(-np.inf, np.inf, 5), (-np.inf, -2., 0)],
    [(-np.inf, np.inf, 5), (2., np.inf, 0)],
    [(-np.inf, np.inf, 5), (-np.inf, -2., 0), (2., np.inf, 0)]])
def test_cheaper_outside_cell_cannot_leak_across_closed_bound(intervals):
    result = bound.restricted_minimum(direct_cells(intervals))
    assert result['cost'] == 5
    assert -2. < result['k'] < 2.
    assert direct_cost(intervals, result['k']) == 5


def test_interior_endpoint_spike_is_never_used_as_constructive_representative():
    intervals = [(-np.inf, np.inf, 5), (-np.inf, 0., 0), (0., np.inf, 0)]
    result = bound.restricted_minimum(direct_cells(intervals))
    assert direct_cost(intervals, 0.) == 5
    assert result['cost'] == 0
    assert result['k'] != 0.
    assert direct_cost(intervals, result['k']) == 0


def test_random_restricted_minima_equal_direct_search_including_domain_endpoints():
    rng = np.random.default_rng(20261015)
    for _ in range(120):
        intervals = [(-np.inf, np.inf, 9)]
        for _ in range(15):
            a, b = sorted(rng.choice(np.arange(-6., 7.), size=2, replace=False))
            intervals.append((a, b, int(rng.integers(0, 9))))
        lower, upper = sorted(rng.choice(np.arange(-5., 6.), size=2, replace=False))
        points = sorted({lower, upper} | {x for a, b, _ in intervals for x in (a, b) if lower < x < upper})
        candidates = points + [(a+b)/2 for a, b in zip(points, points[1:])]
        expected = min(direct_cost(intervals, k) for k in candidates)
        result = bound.restricted_minimum(direct_cells(intervals), lower, upper)
        assert result['cost'] == expected
        assert lower < result['k'] < upper
        assert direct_cost(intervals, result['k']) == expected


@pytest.mark.parametrize('lower,upper', [(0., 0.), (2., -2.), (-np.inf, 2.), (-2., np.nan)])
def test_invalid_or_singleton_domain_rejected(lower, upper):
    with pytest.raises(ValueError, match='finite nondegenerate'):
        bound.restricted_minimum([{'lower': None, 'upper': None, 'cost': 0}], lower, upper)


def test_no_intersecting_cell_is_not_a_feasible_zero_cost():
    with pytest.raises(ValueError, match='no open cell'):
        bound.restricted_minimum([{'lower': -5., 'upper': -3., 'cost': 0}])


def test_malformed_cost_rejected():
    with pytest.raises(ValueError, match='integer disagreement'):
        bound.restricted_minimum([{'lower': None, 'upper': None, 'cost': .5}])


def uncompressed_bands(force, labels):
    """Every interval prediction, with ties indivisible and no cost pruning."""
    levels = np.unique(np.log(force))
    log_force = np.log(force)
    bands = [(0., np.inf, int(np.sum(labels)))]
    for first, last in itertools.combinations_with_replacement(range(len(levels)), 2):
        pred = (levels[first] <= log_force) & (log_force <= levels[last])
        low = levels[last]-levels[first]
        high = (levels[last+1]-levels[first-1]
                if first > 0 and last+1 < len(levels) else np.inf)
        bands.append((low, high, int(np.sum(pred != labels))))
    return bands


def uncompressed_pairs(first, second, dt):
    result = []
    for a, b in itertools.product(uncompressed_bands(*first), uncompressed_bands(*second)):
        lo = ((np.log(b[0])-np.log(a[1]))/dt
              if b[0] > 0 and np.isfinite(a[1]) else -np.inf)
        hi = ((np.log(b[1])-np.log(a[0]))/dt
              if a[0] > 0 and np.isfinite(b[1]) else np.inf)
        result.append((lo, hi, a[2]+b[2]))
    return result


def test_bounded_full_band_problem_matches_uncompressed_search_and_attains_cost():
    """Check clipping in the actual width problem, beyond abstract envelopes."""
    rng = np.random.default_rng(20261017)
    for _ in range(60):
        raw, columns = [], []
        for _ in range(rng.integers(1, 4)):
            pair = []
            for _ in range(2):
                force = np.exp(rng.integers(0, 7, size=rng.integers(1, 6)).astype(float))
                pair.append((force, rng.integers(0, 2, len(force))))
            raw.append(pair)
            columns.append(tuple(bound.cert.enumerate_bands(*side) for side in pair))
        dt = float(rng.uniform(.2, 2.))
        lower, upper = sorted(rng.choice(np.arange(-3., 4.), size=2, replace=False))
        pairs = [uncompressed_pairs(*pair, dt) for pair in raw]
        points = sorted({lower, upper} | {
            x for intervals in pairs for a, b, _ in intervals for x in (a, b)
            if lower < x < upper})
        probes = points+[(np.longdouble(a)+np.longdouble(b))/2
                         for a, b in zip(points, points[1:])]
        expected = min(sum(direct_cost(intervals, k) for intervals in pairs) for k in probes)
        original = bound.cert.solve_shared_width(columns, dt)
        restricted = bound.restricted_minimum(original['cost_cells'], lower, upper)
        assert restricted['cost'] == expected
        k = restricted['k']
        attained = 0
        for (first, second), raw_pair in zip(columns, raw):
            candidates, _ = bound.cert.pair_intervals(first, second, dt)
            chosen = min((item for item in candidates if item['lower'] < k < item['upper']),
                         key=lambda item: item['cost'])
            witness = bound.cert.pair_witness(first, second, chosen, k, dt)
            widths = []
            for side, (force, labels) in zip(('first', 'second'), raw_pair):
                saved = witness[side]
                log_force = np.log(force)
                pred = (log_force >= saved['lower_log_force']) & (log_force <= saved['upper_log_force'])
                errors = np.flatnonzero(pred != labels)
                assert errors.tolist() == sorted(saved['disagreement_original_indices'])
                assert len(errors) == saved['cost']
                attained += len(errors)
                widths.append(saved['upper_log_force']-saved['lower_log_force'])
            assert widths[1] == pytest.approx(np.exp(k*dt)*widths[0], rel=2e-10, abs=2e-13)
        assert attained == expected
