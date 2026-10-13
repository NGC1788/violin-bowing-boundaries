import itertools
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from set_envelope import SetEnvelopeModel, iou_bounds, label_envelope, similarity_coordinates, family_ious


def test_exact_bounds_exhaustive_small_sets():
    for lo in itertools.product((False, True), repeat=3):
        lo = np.array(lo)
        for hi in itertools.product((False, True), repeat=3):
            hi = np.array(hi)
            if np.any(lo & ~hi):
                continue
            for p in itertools.product((False, True), repeat=3):
                p = np.array(p)
                for w in (np.ones(3), np.array([.2, .8, 1.7])):
                    values = []
                    for truth in itertools.product((False, True), repeat=3):
                        truth = np.array(truth)
                        if np.any(lo & ~truth) or np.any(truth & ~hi):
                            continue
                        union = np.sum(w * (p | truth))
                        values.append(float(np.sum(w * (p & truth))/union) if union else 1.)
                    assert iou_bounds(p, lo, hi, w) == pytest.approx((min(values), max(values)))


def test_scaling_covariance_and_counterexample():
    x = np.array([[.03, .2, .05], [.1, 1., .1]])
    scaled = x * np.array([1., 3., 3.])
    np.testing.assert_allclose(similarity_coordinates(x), similarity_coordinates(scaled))
    assert not np.allclose(similarity_coordinates(x, False), similarity_coordinates(scaled, False))
    with pytest.raises(ValueError):
        similarity_coordinates([[.1, 0, .1]])


def test_label_envelope_has_proper_subset_and_known_family():
    features = {"periodicity": [1., .8, .7, 1.], "f0_hz": [98., 98., 98., 40.],
                "second_flyback_ratio": [.1, .5, .1, .1], "flybacks_per_period": [1, 2, 1, 1]}
    lo, hi, mean = label_envelope(features, 98.)
    assert lo.tolist() == [True, False, False, False]
    assert hi.tolist() == [True, True, False, False]
    assert 0 < mean[1] < 1


def test_disconnected_set_can_be_learned_without_filling_gap():
    beta, force = np.meshgrid(np.geomspace(.02, .2, 12), np.geomspace(.03, 4., 35))
    X = np.column_stack((beta.ravel(), force.ravel(), np.full(beta.size, .1)))
    y = (((X[:,1]>.07)&(X[:,1]<.25))|((X[:,1]>1)&(X[:,1]<2.5))).astype(float)
    model = SetEnvelopeModel(regularization=.0001, grid_size=9).fit(X, y)
    assert np.mean(model.predict(X)==y) > .94
    gap = np.array([[.08,.5,.1]])
    assert not model.predict(gap)[0]
    before = model.coef_.copy()
    model.predict_proba(X * np.array([1, 2, 2]))
    np.testing.assert_array_equal(before, model.coef_)


def test_robust_training_is_finite_and_rejects_bad_envelope():
    X = np.column_stack((np.full(30,.1), np.geomspace(.02,4.,30), np.full(30,.1)))
    y = ((X[:,1]>.2)&(X[:,1]<2.)).astype(float)
    model = SetEnvelopeModel(objective="robust_iou").fit(X, y, lower=y, upper=y)
    assert np.isfinite(model.predict_proba(X)).all()
    with pytest.raises(ValueError):
        model.fit(X,y,lower=np.ones(30),upper=np.zeros(30))


def test_coherent_family_can_be_strictly_tighter_than_rectangular_bounds():
    family = np.array([[1,1], [1,0], [1,0], [0,1]], bool)
    prediction = np.array([1,1,0,0], bool)
    lo, hi = family.all(1), family.any(1)
    bounds = iou_bounds(prediction, lo, hi)
    values = family_ious(prediction, family)
    assert bounds[0] < values.min()
    assert bounds[1] >= values.max()


def test_family_and_box_objective_gradients_match_finite_differences(monkeypatch):
    import set_envelope as module
    original = module.minimize
    checked = []
    def verify(function, initial, **kwargs):
        point = initial + np.random.default_rng(2).normal(0,.2,len(initial))
        value, gradient = function(point)
        for j in np.linspace(0,len(point)-1,5).astype(int):
            step = np.zeros(len(point)); step[j] = 1e-5
            finite = (function(point+step)[0]-function(point-step)[0])/(2e-5)
            assert gradient[j] == pytest.approx(finite, abs=1e-7, rel=1e-4)
        checked.append(value)
        return original(function, initial, **kwargs)
    monkeypatch.setattr(module,"minimize",verify)
    X = np.column_stack((np.linspace(.02,.2,20),np.geomspace(.05,4,20),np.full(20,.1)))
    family = np.column_stack([((X[:,1]>.2)&(X[:,1]<bound)) for bound in [.7,1.2,2.]])
    for objective in ["family_iou","robust_iou","nominal_iou"]:
        model = SetEnvelopeModel(objective=objective,grid_size=4)
        model.fit(X, family.mean(1), lower=family.all(1), upper=family.any(1), family=family)
    assert len(checked) == 6


def test_zero_weight_rows_preserve_basis_but_not_labels():
    X = np.column_stack((np.linspace(.02,.2,20),np.geomspace(.05,4,20),np.full(20,.1)))
    target = (X[:,1]>.5).astype(float)
    weights = np.ones(20); weights[5:10] = 0
    first = SetEnvelopeModel(grid_size=4).fit(X,target,sample_weight=weights)
    changed = target.copy(); changed[5:10] = 1-changed[5:10]
    second = SetEnvelopeModel(grid_size=4).fit(X,changed,sample_weight=weights)
    np.testing.assert_allclose(first.coef_,second.coef_)
    np.testing.assert_array_equal(first.anchors_,second.anchors_)
    with pytest.raises(ValueError):
        SetEnvelopeModel().fit(X,target,sample_weight=np.zeros(20))
