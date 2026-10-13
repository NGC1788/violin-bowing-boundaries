"""Mechanism, gradient and source-isolation checks for the affine-flow hypothesis."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import affine_transport as flow


def example():
    beta, force, speed = np.meshgrid(np.geomspace(.02, .2, 10), np.geomspace(.03, 5., 12), [.05, .1, .2], indexing="ij")
    X = np.column_stack((beta.ravel(), force.ravel(), speed.ravel()))
    b, u, t = flow.raw_coordinates(X)
    z = flow.transport(b, u, t, 1.1, .2, .3)
    y = abs(z + .8 * b + .7) < 1.2
    return X, y, np.broadcast_to(np.arange(10)[:, None, None], beta.shape).ravel(), speed.ravel()


@pytest.mark.parametrize("k", [0., 1e-7, -.5, .38])
def test_exact_flow_composition_and_inverse(k):
    b = np.linspace(-1, 1, 30)
    u = np.linspace(-2, 2, 30)
    first = flow.transport(b, u, .37, 1.2, .3, k)
    composed = flow.transport(b, first, -.11, 1.2, .3, k)
    np.testing.assert_allclose(composed, flow.transport(b, u, .26, 1.2, .3, k), atol=2e-14)
    np.testing.assert_allclose(flow.transport(b, first, -.37, 1.2, .3, k), u, atol=2e-14)
    if k == 0:
        np.testing.assert_allclose(first, u - (1.2 + .3 * b) * .37, atol=1e-14)


@pytest.mark.parametrize("k", [0., 1e-7, -.38, .7])
def test_objective_analytic_gradient_matches_finite_difference(k):
    X, y, _, _ = example()
    coordinates, norm = flow.raw_coordinates(X), flow.normalization(X)
    theta = np.array([.3, -.2, .1, -.4, .15, -.8, 1.2, -.3, k])
    value, analytic = flow.objective(theta, coordinates, y, norm, .01, .03)
    numeric = approx_derivative(lambda v: flow.objective(v, coordinates, y, norm, .01, .03)[0], theta,
                                method="3-point", abs_step=1e-5).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=1e-6)


def test_concave_force_band_and_training_objective_multistart():
    X, y, _, speeds = example()
    source = speeds != .2
    model = flow.fit_source(X[source], y[source], .01, .01)
    chosen = model["starts"][model["selected_start"]]
    assert chosen["objective"] == min(s["objective"] for s in model["starts"] if s["success"])
    assert model["theta"][5] <= -1e-6
    grid = np.column_stack((np.repeat(.08, 1000), np.exp(np.linspace(-8, 8, 1000)), np.repeat(.2, 1000)))
    score = flow.design(np.array(model["theta"]), flow.raw_coordinates(grid), model["normalization"])
    assert np.max(np.diff(score, n=2)) < 0
    prediction = score >= 0
    assert np.count_nonzero(np.diff(np.r_[False, prediction, False].astype(int)) == 1) <= 1
    reloaded = json.loads(json.dumps(model))
    np.testing.assert_array_equal(flow.predict_frozen(model, grid), flow.predict_frozen(reloaded, grid))


def test_target_label_poisoning_cannot_change_source_selection(tmp_path):
    X, y, levels, speeds = example()
    source = speeds != .2
    ids = np.array([f"row{i}" for i in range(len(y))])
    def select(labels, path):
        return flow.select_source(X[source], labels[source], labels[source, None], levels[source], speeds[source],
                                  ids[source], path, lambdas=(.01,), warp_lambdas=(.01,))
    first = select(y, tmp_path / "original.jsonl")
    poisoned = y.copy()
    poisoned[~source] = ~poisoned[~source]
    second = select(poisoned, tmp_path / "poisoned.jsonl")
    assert first == second
    assert (tmp_path / "original.jsonl").read_text() == (tmp_path / "poisoned.jsonl").read_text()


def test_published_panel_identifiers_support_finite_shared_evaluator():
    data = flow.load_published()
    assert all("/" in group for group in data["group"])
    score = flow.evaluate_panels(np.full(len(data["X"]), .5), data["y_fixed"] == 1, data["family"],
                                  data["group"], np.ones(len(data["X"]), bool))
    json.dumps(score, allow_nan=False)


def test_single_reference_speed_has_zero_flow_likelihood_gradient():
    X, _, _, _ = example()
    X[:, 2] = .1
    theta = np.array([.3, -.2, .1, -.4, .15, -.8, 1.2, -.3, .7])
    _, jacobian = flow.design(theta, flow.raw_coordinates(X), flow.normalization(X), derivatives=True)
    np.testing.assert_array_equal(jacobian[:, 6:], 0.)
