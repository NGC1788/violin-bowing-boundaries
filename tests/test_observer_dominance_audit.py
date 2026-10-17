"""Same-observer pair differences, dense exactness, ties and panel semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import observer_dominance_audit as audit


def direct(prediction, target):
    union = np.count_nonzero(prediction | target)
    return np.count_nonzero(prediction & target) / union if union else 1.


def test_dense_counts_match_direct_labels_with_ties_empty_sets_and_subsets():
    rng = np.random.default_rng(1809)
    for repeat in range(50):
        ratio = rng.choice([.2, .3, .35, .5, .699, .7, .8, np.nan], 20)
        periodicity = rng.choice([.7, .75, .8, .85, .9, np.nan], 20)
        eligible = rng.random(20) > .2
        observer = audit.ContinuousObserver(ratio, periodicity, eligible)
        dense = audit.DenseObserver(observer)
        predictions = rng.random((6, 20)) > .5
        predictions[0] = False
        predictions[1] = True
        subset = rng.random(20) > .3
        subset[0] = True
        actual = dense.scores(predictions.astype(float), subset)
        for qi, cutoff in enumerate(dense.cutoffs):
            for fi, fraction in enumerate(dense.fractions):
                target = observer.labels(fraction, cutoff)[subset]
                expected = [direct(p[subset], target) for p in predictions]
                np.testing.assert_allclose(actual[:, qi, fi], expected, atol=1e-14, rtol=0)


def test_continuous_shared_rule_reversal_missed_by_fifteen_rule_grid():
    observer = audit.ContinuousObserver([.35, .36, .37], [.8, .8, .8])
    geometry = audit.DenseObserver(observer)
    p = np.array([[1, 0, 0], [0, 1, 1]], bool)
    dense = geometry.scores(p)
    grid = np.array([[direct(prediction, observer.labels(f, q)) for f, q in audit.GRID] for prediction in p])
    row = audit.pair_records(["A", "B"], dense, grid, geometry, task="toy", scope="pooled")[0]
    assert row["continuous_reversal"] and not row["grid_reversal"]
    assert row["grid_misses_continuous_reversal"]
    assert row["continuous_difference_min"] == pytest.approx(-1/3)
    assert row["continuous_difference_max"] == 1.
    for prefix in ("minimum", "maximum"):
        labels = observer.labels(row[prefix + "_witness_fraction"], row[prefix + "_witness_periodicity"])
        assert row[prefix + "_witness_iou_a"] == direct(p[0], labels)
        assert row[prefix + "_witness_iou_b"] == direct(p[1], labels)


def test_shared_difference_is_not_difference_of_independent_extrema():
    observer = audit.ContinuousObserver([.35, .36, .37], [.8] * 3)
    geometry = audit.DenseObserver(observer)
    same_prediction = np.array([[1, 0, 1], [1, 0, 1]], bool)
    scores = geometry.scores(same_prediction)
    independent_lower = scores[0].min() - scores[1].max()
    assert independent_lower < 0
    assert np.all(scores[0] - scores[1] == 0)
    grid = np.array([[direct(p, observer.labels(f, q)) for f, q in audit.GRID] for p in same_prediction])
    row = audit.pair_records(["A", "B"], scores, grid, geometry, task="toy", scope="pooled")[0]
    assert not row["continuous_reversal"]
    assert row["continuous_difference_min"] == row["continuous_difference_max"] == 0.


def test_macro_uses_same_operator_for_every_panel_and_differs_from_pooled():
    features = {"second_flyback_ratio": np.array([.2, .35, .36, .4, .5, .6]),
        "periodicity": np.array([.9, .8, .8, .8, .82, .78]), "f0_hz": np.full(6, 98.),
        "flybacks_per_period": np.ones(6), "amplitude_std": np.ones(6)}
    data = {"ids": np.array([str(i) for i in range(6)]), "output_features": features,
            "group": np.array(["A/.1"] * 2 + ["B/.1"] * 4), "reference_f0_hz": 98.}
    probabilities = {"A": np.array([1., 0., 0., 1., 0., 1.]), "B": np.array([0., 1., 1., 0., 1., 0.])}
    records, metadata = audit.audit_task(data, probabilities, np.ones(6, bool), "toy")
    assert len(records) == 4  # pooled, two panels, and shared-operator macro
    observer = audit.ContinuousObserver.from_features(features, 98.)
    differences = []
    for f in observer.f_states:
        for q in observer.q_states:
            labels = observer.labels(f, q)
            per_model = []
            for p in probabilities.values():
                per_model.append(np.mean([direct(p[keep].astype(bool), labels[keep])
                    for group in np.unique(data["group"]) for keep in [data["group"] == group]]))
            differences.append(per_model[0] - per_model[1])
    macro = next(r for r in records if r["scope"] == "macro_shared")
    assert macro["continuous_difference_min"] == pytest.approx(min(differences))
    assert macro["continuous_difference_max"] == pytest.approx(max(differences))
    assert metadata["panels"] == 2


def test_state_budget_and_invalid_predictions_fail_closed():
    observer = audit.ContinuousObserver([.35, .36, .37], [.8] * 3)
    with pytest.raises(ValueError, match="state budget"):
        audit.DenseObserver(observer, max_states=2)
    geometry = audit.DenseObserver(observer)
    with pytest.raises(ValueError, match="binary"):
        geometry.scores(np.array([[.5, 0., 1.]]))
    with pytest.raises(ValueError, match="subset"):
        geometry.scores(np.ones((1, 3)), np.zeros(3, bool))


def test_mean_panel_minimum_can_prefer_uniformly_dominated_shared_macro_model():
    # Two attainable maps in BOTH panels: {0} and {0,1,2,3}. Ratio ties
    # prevent impossible intermediate maps. The two models have different
    # adverse operators in different panels.
    ratio = np.tile([.2, .5, .5, .5, .8, .8], 2)
    features = {"second_flyback_ratio": ratio, "periodicity": np.full(12, .9),
                "f0_hz": np.full(12, 98.), "flybacks_per_period": np.ones(12)}
    data = {"ids": np.array([str(i) for i in range(12)]), "output_features": features,
            "group": np.repeat(["panel1", "panel2"], 6), "reference_f0_hz": 98.}
    predictions = {"A": np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
                   "B": np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])}
    records, _ = audit.audit_task(data, predictions, np.ones(12, bool), "constructed")
    macro = next(row for row in records if row["scope"] == "macro_shared")
    # Old metric: A=.25; B=(1/3 + 1/5)/2=4/15 > .25, prefers B.
    assert macro["original_grid_mean_panel_min_difference"] == pytest.approx(-1/60)
    # With any shared observer, A macro=.625 while B is 5/12 or .3.
    assert macro["continuous_difference_min"] == pytest.approx(5/24)
    assert macro["continuous_difference_max"] == pytest.approx(13/40)
    assert macro["original_grid_macro_ranking_has_opposing_shared_rule"]
    assert not macro["continuous_reversal"]  # A uniformly dominates B.
