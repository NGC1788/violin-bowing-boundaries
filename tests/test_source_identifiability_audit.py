"""Tests for observational geometry, not for a physical label interpretation."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import source_identifiability_audit as audit


def test_exact_finite_brackets_and_censoring_are_not_point_estimates():
    g = audit.boundary_geometry(np.exp(np.arange(5)), [0, 1, 1, 0, 0])
    assert g["minimum_disagreement_count"] == 0
    assert g["exact_single_band"] == {
        "lower": [0.0, 1.0], "upper": [2.0, 3.0],
        "width": [1.0, 3.0], "center": [1.0, 2.0]}
    left = audit.boundary_geometry([1, 2, 3], [1, 1, 0])
    right = audit.boundary_geometry([1, 2, 3], [0, 1, 1])
    assert left["exact_single_band"]["lower"][0] == -np.inf
    assert right["exact_single_band"]["upper"][1] == np.inf
    assert not audit.finite_both(left["exact_single_band"])
    assert not audit.finite_both(right["exact_single_band"])
    assert left["exact_single_band"]["width"][1] == np.inf


def test_all_minimum_disagreement_intervals_match_exhaustive_predictions():
    n = 7
    force = np.arange(1, n + 1, dtype=float)
    for labels in itertools.product((0, 1), repeat=n):
        y = np.asarray(labels)
        alternatives = [(None, int(y.sum()))]
        for lower in range(n):
            for upper in range(lower, n):
                pred = (np.arange(n) >= lower) & (np.arange(n) <= upper)
                alternatives.append(((lower, upper), int(np.sum(pred != y))))
        best = min(score for _, score in alternatives)
        expected = {interval for interval, score in alternatives if score == best and interval is not None}
        got = audit.boundary_geometry(force, y)
        observed = {(h["first"], h["last"]) for h in got["optimal_positive_intervals"]}
        assert observed == expected
        assert got["minimum_disagreement_count"] == best
        assert got["empty_optimal"] == (int(y.sum()) == best)


def test_disconnected_observations_preserved_with_all_optimal_ties():
    result = audit.boundary_geometry([1, 2, 3, 4, 5], [0, 1, 0, 1, 0])
    assert result["component_count"] == 2
    assert result["exact_single_band"] is None
    assert result["minimum_disagreement_count"] == 1
    assert {(v["first"], v["last"]) for v in result["optimal_positive_intervals"]} == {(1, 1), (1, 3), (3, 3)}
    assert result["components"][0]["first"] == 1
    assert result["components"][1]["first"] == 3


def test_tied_force_measurements_cannot_be_separated_by_interval():
    result = audit.boundary_geometry([1, 1], [0, 1])
    assert result["force_ties_with_conflicting_labels"]
    assert result["exact_single_band"] is None
    assert result["minimum_disagreement_count"] == 1
    assert result["empty_optimal"]
    assert [(h["first"], h["last"]) for h in result["optimal_positive_intervals"]] == [(0, 1)]


def test_empty_tie_does_not_assert_finite_unconditional_boundaries():
    # Predicting just the repeated middle force and predicting nothing both
    # disagree with one label. The empty alternative cannot identify a band.
    result = audit.boundary_geometry([1, 2, 2, 3], [0, 0, 1, 0])
    assert result["minimum_disagreement_count"] == 1
    assert result["empty_optimal"]
    assert audit.finite_both(result["minimum_disagreement_nonempty_band_envelope"])
    assert result["minimum_disagreement_band_envelope"] is None


def test_no_positive_does_not_invent_a_boundary_or_global_absence():
    result = audit.boundary_geometry([1, 2, 3], [0, 0, 0])
    assert result["exact_single_band"] is None
    assert result["minimum_disagreement_band_envelope"] is None
    assert result["empty_optimal"]
    assert "does not rule out" in result["scope"]


def test_source_permutation_leaves_geometry_identical():
    f, y = np.array([1, 2, 3, 4]), np.array([0, 1, 0, 1])
    perm = [3, 1, 0, 2]
    assert audit.boundary_geometry(f, y) == audit.boundary_geometry(f[perm], y[perm])


def test_width_generator_and_source_only_extrapolation_leverage():
    assert audit.width_generator_interval([1, 2], [4, 8], np.log(2)) == pytest.approx([1, 3])
    assert audit.width_generator_interval([0, np.inf], [0, np.inf], 1) == [-np.inf, np.inf]
    assert audit.extrapolation_weights((.05, .2), .1)["absolute_weight_sum"] == pytest.approx(1)
    assert audit.extrapolation_weights((.05, .1), .2)["absolute_weight_sum"] == pytest.approx(3)
    assert audit.extrapolation_weights((.15, .2), .05)["absolute_weight_sum"] == pytest.approx(8.6376833586)
    # A quadratic reaches the textbook interpolation remainder exactly.
    pair, target = (.05, .1), .2
    t1, t2, t = np.log(pair[0]), np.log(pair[1]), np.log(target)
    w = audit.extrapolation_weights(pair, target)
    predicted = w["first"] * t1**2 + w["second"] * t2**2
    assert abs(t**2 - predicted) == pytest.approx(2 * w["curvature_remainder_factor"])


@pytest.mark.parametrize("first,second,dt", [([1, 2], [1, 2], 0),
    ([1, 2], [1, 2], np.inf), ([-1, 2], [1, 2], 1),
    ([2, 1], [1, 2], 1), ([0, 0], [1, 2], 1), ([1, np.nan], [1, 2], 1)])
def test_invalid_width_constraints_rejected(first, second, dt):
    with pytest.raises(ValueError):
        audit.width_generator_interval(first, second, dt)


@pytest.mark.parametrize("pair,target", [((.1, .1), .2), ((.2, .1), .3),
    ((0, .1), .2), ((.1, .2), 0), ((.1, .2), np.nan)])
def test_invalid_speed_geometry_rejected(pair, target):
    with pytest.raises(ValueError):
        audit.extrapolation_weights(pair, target)


def test_pair_outer_constraints_detect_incompatible_width_rates():
    def row(level, width):
        return {"beta_level": level, "band": {"lower": [0., 0.], "upper": [width, width],
                                                "width": [width, width], "center": [width/2, width/2]}}
    first = [row(0, 1), row(1, 1)]
    second = [row(0, 2), row(1, 4)]
    summary, _ = audit.pair_summary(first, second, (1, 2), [3], "band")
    assert summary["both_fully_bracketed"] == 2
    assert summary["shared_k_intersection"] == pytest.approx([2, 1])
    assert not summary["shared_k_intersection_nonempty"]
    assert summary["shared_constant_boundary_slopes"]["lower"]["contains_known_exponent"]
    assert not summary["shared_constant_boundary_slopes"]["upper"]["nonempty"]
    assert summary["shared_constant_boundary_slopes"]["upper"]["minimum_uniform_log_boundary_expansion_for_any_common_slope"] == pytest.approx(.5)


@pytest.mark.parametrize("force,labels", [([0, 1], [0, 1]), ([1, np.nan], [0, 1]),
                                        ([1, 2], [0, 2]), ([1, 2], [0, np.nan])])
def test_invalid_observations_rejected(force, labels):
    with pytest.raises(ValueError):
        audit.boundary_geometry(force, labels)
