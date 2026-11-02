import numpy as np
from agentic_research.violin_diagnosis_v1.diagnose import macro, losses, local_directions, statistics, offset_geometry
from agentic_research.violin_diagnosis_v1.diagnose import EPS


def test_participant_macro_and_confident_wrong_loss_tail():
    ids = np.array(['a', 'a', 'a', 'b'])
    assert macro(np.array([0., 0., 0., 2.]), ids) == 1
    y = np.zeros((4, 9)); p = np.full((4, 9), .1); p[3] = .99
    s = statistics(y, p, ids)
    assert s['wrong_95_fraction'] == .5
    assert s['wrong_95_loss_share'] > .95


def test_analytic_local_directions_match_finite_differences_without_fitting():
    y = np.array([[0., 1.], [1., 0.]])
    p = np.array([[.8, .1], [.7, .3]]); b = np.full_like(p, .5)
    mix, soft = local_directions(y, p, b); eps = 1e-6
    fd = (losses(y, p + eps * (b - p)) - losses(y, p)) / eps
    np.testing.assert_allclose(mix, fd, atol=5e-5)
    logits = np.log(p / (1 - p)); changed = 1 / (1 + np.exp(-(1 - eps) * logits))
    fdsoft = (losses(y, changed) - losses(y, p)) / eps
    np.testing.assert_allclose(soft, fdsoft, atol=5e-5)


def test_geometry_is_invariant_to_softmax_row_constants():
    rng = np.random.default_rng(5); prior = rng.dirichlet(np.ones(10), 11); states = np.array([0, 1, 1, 2])
    residual = rng.normal(size=(4, 10)); logits = np.log(prior[states]) + residual
    mass = np.exp(logits - logits.max(1, keepdims=True)); mass /= mass.sum(1, keepdims=True)
    people = np.array(['a', 'a', 'b', 'b'])
    a = offset_geometry(mass, residual, states, prior, people)
    b = offset_geometry(mass, residual + np.array([[10], [-2], [5], [100]]), states, prior, people)
    for k in a: np.testing.assert_allclose(a[k], b[k], atol=1e-12)


def test_softening_one_sided_derivative_at_both_clip_boundaries():
    p = np.array([[EPS, 1-EPS]]); y = np.array([[1., 0.]])
    _, derivative = local_directions(y, p, np.full_like(p,.5))
    expected = -(p-y) * (np.log(p)-np.log1p(-p))
    np.testing.assert_allclose(derivative, expected)
    step=1e-4; logits=np.log(p)-np.log1p(-p)
    changed=1/(1+np.exp(-(1-step)*logits))
    fd=(losses(y,changed)-losses(y,p))/step
    np.testing.assert_allclose(derivative,fd,rtol=1e-6,atol=1e-6)
