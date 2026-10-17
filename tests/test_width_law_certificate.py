"""Independent finite-enumeration checks for the retrospective width diagnostic."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import width_law_certificate as cert


def direct_bands(force, labels):
    """No envelope compression or domination: enumerate every physical prediction."""
    order = np.argsort(force, kind="stable")
    f, y = np.log(np.asarray(force)[order]), np.asarray(labels)[order]
    levels = np.unique(f)
    result = [(0., np.inf, int(y.sum()))]
    for a, b in itertools.combinations_with_replacement(range(len(levels)), 2):
        pred = (f >= levels[a]) & (f <= levels[b])
        low = levels[b] - levels[a]
        high = levels[b + 1] - levels[a - 1] if a and b + 1 < len(levels) else np.inf
        result.append((low, high, int(np.sum(pred != y))))
    return result


def direct_pairs(first, second, dt):
    result = []
    for a, b in itertools.product(first, second):
        lo = (np.log(b[0]) - np.log(a[1])) / dt if b[0] > 0 and np.isfinite(a[1]) else -np.inf
        hi = (np.log(b[1]) - np.log(a[0])) / dt if a[0] > 0 and np.isfinite(b[1]) else np.inf
        result.append((lo, hi, a[2] + b[2]))
    return result


def direct_shared(raw_columns, dt):
    pairs = [direct_pairs(direct_bands(*a), direct_bands(*b), dt) for a, b in raw_columns]
    points = sorted(set(x for ps in pairs for p in ps for x in p[:2]))
    probes = []
    for a, b in zip(points, points[1:]):
        if np.isneginf(a):
            probes.append(0. if np.isposinf(b) else b - max(1., abs(b)))
        elif np.isposinf(b):
            probes.append(a + max(1., abs(a)))
        else:
            probes.append((np.longdouble(a) + np.longdouble(b)) / 2)
    def cost(k):
        return sum(min(c for a, b, c in ps if a < k < b) for ps in pairs)
    return min(map(cost, probes)), cost(0.)


def verify_witness(result, raw_columns, dt):
    total = 0
    for witness, raw_pair in zip(result["witnesses"], raw_columns):
        for side, (force, labels) in zip(("first", "second"), raw_pair):
            w = witness[side]
            f = np.log(force)
            pred = (f >= w["lower_log_force"]) & (f <= w["upper_log_force"])
            wrong = np.flatnonzero(pred != labels)
            assert len(wrong) == w["cost"]
            assert sorted(w["disagreement_original_indices"]) == wrong.tolist()
            assert w["upper_log_force"] > w["lower_log_force"]
            total += len(wrong)
        assert witness["second"]["width"] == pytest.approx(
            witness["first"]["width"] * np.exp(result["witness_k"] * dt))
    assert total == result["shared_width_cost"]


def test_random_small_global_optima_equal_uncompressed_exhaustive_search():
    rng = np.random.default_rng(20261014)
    for _ in range(120):
        raw, columns = [], []
        for _ in range(rng.integers(1, 4)):
            pair = []
            for _ in range(2):
                # Integer log levels deliberately create exact force ties.
                f = np.exp(rng.integers(0, 7, size=rng.integers(1, 6)).astype(float))
                y = rng.integers(0, 2, len(f))
                pair.append((f, y))
            raw.append(pair)
            columns.append(tuple(cert.enumerate_bands(*v) for v in pair))
        dt = float(rng.uniform(.2, 2))
        result = cert.solve_shared_width(columns, dt)
        expected, fixed = direct_shared(raw, dt)
        assert result["shared_width_cost"] == expected
        assert result["constant_width_cost"] == fixed
        verify_witness(result, raw, dt)


def test_single_column_every_binary_pattern_has_no_transport_penalty():
    # Free k makes any two strictly positive widths compatible in one column.
    f = np.exp(np.arange(4, dtype=float))
    for a, b in itertools.product(itertools.product((0, 1), repeat=4), repeat=2):
        result = cert.solve_shared_width([(cert.enumerate_bands(f, a), cert.enumerate_bands(f, b))], 1.)
        assert result["extra_disagreements"] == 0


def test_finite_width_upper_is_strict_and_lower_is_attainable():
    c = cert.enumerate_bands(np.exp(np.arange(4, dtype=float)), [0, 1, 1, 0])
    b = next(b for b in c["bands"] if b["cost"] == 0)
    assert b["width_lower"] == pytest.approx(1.)
    assert b["width_upper"] == pytest.approx(3.)
    # Width 3 would include one adjacent negative however the center is placed.
    with pytest.raises(ArithmeticError, match="width outside"):
        cert._band_witness(c, c["bands"].index(b), 3.)
    # The closed lower endpoint realizes just the two positive observations.
    pred = (c["log_force"] >= c["log_force"][1]) & (c["log_force"] <= c["log_force"][2])
    assert np.array_equal(pred, c["labels"])
    witness = cert._band_witness(c, c["bands"].index(b), b["width_lower"])
    assert witness["cost"] == 0


def test_opposing_width_constraints_require_a_positive_global_gap():
    f, y = np.exp(np.arange(4, dtype=float)), np.array([0, 1, 1, 0])
    a, b = cert.enumerate_bands(f, y), cert.enumerate_bands(f**3, y)
    result = cert.solve_shared_width([(a, b), (b, a)], 1.)
    assert result["independent_band_cost"] == 0
    assert result["extra_disagreements"] == 1
    assert result["constant_width_extra_disagreements"] == 2
    verify_witness(result, [[(f, y), (f**3, y)], [(f**3, y), (f, y)]], 1.)


def test_k_breakpoint_is_not_declared_feasible_from_neighbors():
    a = {"log_force": np.array([0., 1., 2., 3.]), "labels": np.array([0, 1, 1, 0]),
         "order": np.arange(4), "independent_cost": 0,
         "bands": [{"first": None, "last": None, "cost": 2, "width_lower": 0., "width_upper": np.inf},
                   {"first": 1, "last": 2, "cost": 0, "width_lower": 1., "width_upper": 3.}]}
    b = {**a, "log_force": a["log_force"] * 3,
         "bands": [{"first": None, "last": None, "cost": 2, "width_lower": 0., "width_upper": np.inf},
                   {"first": 1, "last": 2, "cost": 0, "width_lower": 3., "width_upper": 9.}]}
    pairs, _ = cert.pair_intervals(a, b, 1.)
    zero = [p for p in pairs if p["cost"] == 0]
    assert min(p["lower"] for p in zero) == 0.
    assert not any(p["lower"] < 0. < p["upper"] for p in zero)
    result = cert.solve_shared_width([(a, b)], 1.)
    assert result["shared_width_cost"] == 0
    assert result["constant_width_cost"] == 2


def test_off_grid_band_allows_arbitrary_width_and_conflicting_ties():
    c = cert.enumerate_bands([1, 2, 2, 3], [0, 0, 1, 0])
    assert c["independent_cost"] == 1
    # Tied best nonempty alternatives can be safely replaced by off-grid at
    # exactly the same loss, since it has no upper width limit.
    assert len(c["bands"]) == 1
    for width in (1e-4, 1., 1e3):
        w = cert._band_witness(c, 0, width)
        assert w["cost"] == 1
        assert w["lower_log_force"] > np.log(3)


def test_multiplicative_force_calibration_does_not_change_width_diagnostic():
    rng = np.random.default_rng(1971)
    raw = [tuple((np.exp(np.cumsum(rng.uniform(.1, .8, 7))), rng.integers(0, 2, 7))
                 for _ in range(2)) for _ in range(4)]
    make = lambda pairs: [tuple(cert.enumerate_bands(*v) for v in p) for p in pairs]
    first = cert.solve_shared_width(make(raw), .7)
    shifted = [tuple((f * np.exp(rng.uniform(-3, 3)), y) for f, y in p) for p in raw]
    second = cert.solve_shared_width(make(shifted), .7)
    for key in ("independent_band_cost", "shared_width_cost", "constant_width_cost", "extra_disagreements"):
        assert first[key] == second[key]
    assert first["witness_k"] == pytest.approx(second["witness_k"])
    verify_witness(second, shifted, .7)


@pytest.mark.parametrize("force,labels", [([], []), ([0, 1], [0, 1]), ([1, np.inf], [0, 1]),
    ([1, 2], [0, 2]), ([1, 2], [0, np.nan]), ([1, 2], [0]), ([[1, 2]], [[0, 1]])])
def test_invalid_observations_rejected(force, labels):
    with pytest.raises(ValueError):
        cert.enumerate_bands(force, labels)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_invalid_speed_separation_rejected(dt):
    c = cert.enumerate_bands([1, 2], [0, 1])
    with pytest.raises(ValueError):
        cert.solve_shared_width([(c, c)], dt)


def test_empty_column_collection_rejected():
    with pytest.raises(ValueError):
        cert.solve_shared_width([], 1.)


def load_report():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'research' / '2026-10-14' / 'width_law_report.py'
    spec = importlib.util.spec_from_file_location('width_report', path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    return report


def test_independent_report_baseline_matches_all_small_tied_force_patterns():
    report = load_report()
    force = np.array([1., 2., 2., 4.])
    for labels in itertools.product((0, 1), repeat=4):
        assert report.minimum_band_errors(force, labels) == min(b[2] for b in direct_bands(force, labels))


def test_independent_report_reclassifies_witness_and_rejects_tampering():
    import copy
    report = load_report()
    force, labels = np.exp(np.arange(4, dtype=float)), np.array([0, 1, 1, 0])
    pair = (cert.enumerate_bands(force, labels), cert.enumerate_bands(force**3, labels))
    result = cert.solve_shared_width([pair], np.log(2.))
    data = {'string': np.repeat('A', 8), 'speed': np.repeat([.1, .2], 4),
            'beta_level': np.zeros(8, dtype=int), 'ids': np.array([str(i) for i in range(8)]),
            'X': np.column_stack([np.repeat(.1, 8), np.r_[force, force**3], np.repeat([.1, .2], 4)])}
    result.update(source='A', source_speeds=[.1, .2], source_rows=8,
                  source_ids_sha256=report.bd.id_hash(data['ids']),
                  columns=[{'beta_level': 0, 'first_n': 4, 'second_n': 4}])
    checked = report.verify_task(result, data, np.tile(labels, 2))
    assert checked['attained_shared_width_cost'] == 0
    broken = copy.deepcopy(result)
    broken['witnesses'][0]['first']['lower_log_force'] = 50.
    with pytest.raises(AssertionError):
        report.verify_task(broken, data, np.tile(labels, 2))
    broken = copy.deepcopy(result)
    broken['columns'][0]['beta_level'] = 1
    with pytest.raises(AssertionError, match='incomplete nominal beta'):
        report.verify_task(broken, data, np.tile(labels, 2))


def test_report_only_removes_local_repository_prefix():
    report = load_report()
    source = {'source': str(report.ROOT / 'research' / 'input.csv'), 'hash': 'a123', 'rows': [1, 2]}
    assert report.portable_paths(source) == {'source': 'research/input.csv', 'hash': 'a123', 'rows': [1, 2]}
