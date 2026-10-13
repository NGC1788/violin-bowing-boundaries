"""Mathematical and inference-contract tests for fixed boundary priors."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from boundary_transport import BoundaryTransport, objective, source_boundary_audit


def known_model():
    model = BoundaryTransport()
    model.beta_center_, model.beta_scale_, model.force_center_ = 0., 1., 0.
    # At source speed: logit=1-(log F)^2, roots exp(-1), exp(1).
    model.coef_ = np.array([1., 0., 0., 0., 0., -1.])
    return model


def test_objective_gradient():
    rng = np.random.default_rng(781)
    design = rng.normal(size=(31, 6))
    target = rng.integers(0, 2, 31)
    coefficient = rng.normal(size=6)
    err = check_grad(lambda c: objective(c, design, target, .1)[0],
                     lambda c: objective(c, design, target, .1)[1], coefficient)
    assert err < 1e-6


def test_source_identity_and_fixed_root_exponents():
    model = known_model()
    X = np.array([[.1, .5, .1], [.2, 2., .1], [.3, 5., .1]])
    for alpha in ((0., 0.), (0., 1.), (1., 0.), (1., 1.)):
        np.testing.assert_allclose(model.decision_function(X, alpha), model.design(X)@model.coef_)
    X[:, 2] = .2
    for alpha in ((0., 0.), (0., 1.), (1., 0.), (1., 1.)):
        geo = model.geometry(X, alpha)
        np.testing.assert_allclose(geo["lower_log_force"], -1+alpha[0]*np.log(2))
        np.testing.assert_allclose(geo["upper_log_force"], 1+alpha[1]*np.log(2))


def test_uniform_equals_force_coordinate_shift():
    model = known_model()
    X = np.array([[.1, .5, .05], [.2, 2., .2], [.3, 5., .3]])
    expected = 1-(np.log(X[:, 1])-np.log(X[:, 2]/.1))**2
    np.testing.assert_allclose(model.decision_function(X, (1., 1.)), expected)


def test_missing_and_crossed_roots_do_not_create_a_band():
    model = known_model()
    X = np.array([[.1, 1., 1.], [.1, 4., 1.]])
    assert not model.geometry(X, (1., 0.))["has_band"].any()
    assert np.all(model.predict_proba(X, (1., 0.))[:, 1] < .5)
    model.coef_[0] = -1.
    assert not model.geometry(X)["has_band"].any()
    assert np.all(model.predict_proba(X)[:, 1] < .5)


def test_censoring_and_disconnected_labels_preserved():
    model = known_model()
    X = np.column_stack((np.full(4, .1), [.5, .7, 1., 1.5], np.full(4, .1)))
    row = source_boundary_audit(model, X, np.array([1, 0, 1, 1]), np.ones(4, int))[0]
    assert row["positive_components"] == 2
    assert row["observed_low_edge_positive"] and row["observed_high_edge_positive"]
    assert row["lower_outside_acquisition"] and row["upper_outside_acquisition"]


def test_fit_recovers_a_simple_band():
    beta, force = np.meshgrid(np.linspace(.03, .15, 9), np.geomspace(.05, 10, 45))
    X = np.column_stack((beta.ravel(), force.ravel(), np.full(beta.size, .1)))
    y = (X[:, 1] > .4) & (X[:, 1] < 3.)
    model = BoundaryTransport(.001).fit(X, y)
    assert model.coef_[-1] < 0
    assert np.mean((model.predict_proba(X, (0., 0.))[:, 1] >= .5) == y) > .97
