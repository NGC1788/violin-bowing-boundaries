"""Check the independent A2 audit oracle against exhaustive tiny set problems."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'research/2026-10-14/lockbox_independent_audit.py'
SPEC = importlib.util.spec_from_file_location('independent_a2', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_set_metric_bounds_against_all_binary_completions():
    rng = np.random.default_rng(191)
    for n in range(1, 7):
        for _ in range(12):
            probability = rng.choice([0., .49, .5, 1.], n)
            family = rng.integers(0, 2, (n, 3), dtype=np.int8).astype(bool)
            p = probability >= .5
            lower, upper = family.all(axis=1), family.any(axis=1)
            possible_scores = []
            for values in itertools.product((False, True), repeat=n):
                truth = np.array(values)
                if np.any(lower & ~truth) or np.any(truth & ~upper):
                    continue
                union = np.count_nonzero(p | truth)
                possible_scores.append(np.count_nonzero(p & truth) / union if union else 1.)
            actual = audit.score(probability, family[:, 1], family)
            assert actual['box_lower_iou'] == min(possible_scores)
            assert actual['box_upper_iou'] == max(possible_scores)
            assert actual['predicted_positive'] == np.count_nonzero(p)


def test_empty_sets_and_threshold_tie():
    result = audit.score(np.array([0.]), np.array([False]), np.zeros((1, 15), bool))
    assert result['nominal_iou'] == result['box_lower_iou'] == result['box_upper_iou'] == 1.
    result = audit.score(np.array([.5]), np.array([False]), np.zeros((1, 15), bool))
    assert result['nominal_iou'] == result['box_lower_iou'] == result['box_upper_iou'] == 0.


def test_macro_is_unweighted_and_different_from_pooled():
    probability = np.array([1., 1., 1., 1.])
    nominal = np.array([True, False, False, False])
    family = np.repeat(nominal[:, None], 15, axis=1)
    score = audit.panels_score(probability, nominal, family, np.array(['A2/0.05', 'A2/0.1', 'A2/0.1', 'A2/0.1']))
    assert score['macro_panels']['family_min_iou'] == .5
    assert score['global_same_rule']['family_min_iou'] == .25


def test_recursive_comparison_rejects_numeric_or_schema_change():
    assert audit.compare({'a': [1., 2.], 'b': 'fixed'}, {'a': [1., 2.], 'b': 'fixed'}) == [0., 0.]
    for changed in ({'a': [1., 2.1], 'b': 'fixed'}, {'a': [1., 2.], 'b': 'changed'}, {'a': [1., 2.]}):
        with pytest.raises(AssertionError):
            audit.compare({'a': [1., 2.], 'b': 'fixed'}, changed)
