"""Independent exhaustive checks of the scientific oracle, DP and query accounting."""
import itertools
import numpy as np
import pytest
from harness.data import Column, select_columns, conditional_features, rule_bank
from harness.run_distribution import run_distribution, first_run, certificate, certificate_lower_bound, information_gain, entropy
from harness.experiment import simulate, QueryOracle, even_order


def brute_first(y, k):
    for i in range(len(y)-k+1):
        if all(y[i:i+k]):
            return i
    return len(y)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_distribution_matches_independent_enumeration(k):
    p = np.array([.07, .2, .9, .5, .31, .83])
    expected = np.zeros(len(p)+1)
    for y in itertools.product([0, 1], repeat=len(p)):
        probability = np.prod(np.where(y, p, 1-p))
        expected[brute_first(y, k)] += probability
    actual = run_distribution(p, k)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(actual.sum(), 1, atol=2e-15)


def test_certificate_exact_for_every_small_partial_assignment():
    n = 6
    for raw in itertools.product([-1, 0, 1], repeat=n):
        obs = np.array(raw)
        unknown = np.flatnonzero(obs < 0)
        possible = set()
        for fill in itertools.product([0, 1], repeat=len(unknown)):
            complete = obs.copy(); complete[unknown] = fill
            possible.add(brute_first(complete.tolist(), 2))
        cert, early, late = certificate(obs)
        assert bool(cert) == (len(possible) == 1)
        assert int(early) == min(possible)
        assert int(late) == max(possible)


def test_hindsight_single_rule_bound_is_optimal_exhaustively():
    n = 6
    for raw in itertools.product([0, 1], repeat=n):
        y = np.array(raw)
        best = n
        for mask in itertools.product([False, True], repeat=n):
            if sum(mask) >= best:
                continue
            obs = np.where(mask, y, -1)
            possible = set()
            unknown = np.flatnonzero(obs < 0)
            for fill in itertools.product([0, 1], repeat=len(unknown)):
                complete = obs.copy(); complete[unknown] = fill
                possible.add(brute_first(complete.tolist(), 2))
            if len(possible) == 1:
                best = sum(mask)
        lower, each = certificate_lower_bound(y)
        assert lower == best == each[0]


def test_information_gain_matches_separate_conditionals():
    p = np.array([[.1, .8, .3, .6, .4], [.3, .2, .6, .2, .9]])
    actual = information_gain(p, np.ones(5, bool))
    initial = entropy(run_distribution(p)).sum()
    for j in range(5):
        q0, q1 = p.copy(), p.copy()
        q0[:, j], q1[:, j] = 0, 1
        expected = initial-((1-p[:, j])*entropy(run_distribution(q0))+
                            p[:, j]*entropy(run_distribution(q1))).sum()
        assert actual[j] == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize('policy', ['ascending', 'random', 'even_first', 'uncertainty', 'entropy'])
def test_queries_reconstruct_nonmonotonic_multiple_rules(policy):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, (3, 50))
    labels[0] = 0  # no run must also certify
    labels[1, :8] = [0, 1, 0, 0, 1, 1, 0, 1]
    oracle = QueryOracle(labels)
    result = simulate(np.full((3, 50), .4), oracle.query, policy, rng)
    expected = [brute_first(y.tolist(), 2) for y in labels]
    np.testing.assert_array_equal(result['certified_start'], expected)
    assert len(set(result['queries'])) == len(result['queries']) == len(oracle.queried)
    assert result['snapshots'].shape == (5, 3, 51)
    np.testing.assert_allclose(result['snapshots'].sum(axis=-1), 1., atol=1e-12)
    assert np.abs((result['snapshots'][-1]*np.arange(51)).sum(-1)-expected).max() == 0
    assert result['certificate_count'] >= certificate_lower_bound(labels)[0]


def test_oracle_rejects_duplicate_and_bad_positions():
    oracle = QueryOracle(np.zeros((9, 50)))
    oracle.query(3)
    with pytest.raises(ValueError): oracle.query(3)
    with pytest.raises(ValueError): oracle.query(50)
    with pytest.raises(TypeError): oracle.query(2.5)
    assert len(oracle.queried) == 1


def test_split_is_paired_and_features_exclude_identity_or_outcome():
    sessions = ('C1', 'C_T2', 'D1', 'D2')
    cols = [Column(f'{s}-{i}', s, s, np.zeros((50, 3)), np.zeros((9, 50)))
            for s in sessions for i in range(8)]
    a = select_columns(cols, 16, sessions)
    b = select_columns(list(reversed(cols)), 16, sessions)
    assert [c.id for c in a] == [c.id for c in b]
    assert all(sum(c.session == s for c in a) == 4 for s in sessions)
    x = conditional_features(np.zeros((2, 50, 3)), rule_bank('nine'))
    assert x.shape == (2, 9, 50, 5)
    assert sorted(even_order(50)) == list(range(50))
    assert even_order(50)[:25] == list(range(0, 50, 2))


def test_differentiable_boundary_likelihood_is_exact_and_has_finite_gradient():
    torch = pytest.importorskip('torch')
    from harness.models import torch_run_log_probs
    logits = torch.tensor([[.2, -.3, 1.5, .7, -2.]], dtype=torch.float64, requires_grad=True)
    logp = torch_run_log_probs(logits)
    expected = run_distribution(torch.sigmoid(logits).detach().numpy())
    np.testing.assert_allclose(logp.exp().detach().numpy(), expected, atol=2e-15)
    (-logp[0, 3]).backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_boundary_log_likelihood_does_not_underflow():
    torch = pytest.importorskip('torch')
    from harness.models import torch_run_log_probs
    logits = torch.full((1, 50), 30., requires_grad=True)
    logp = torch_run_log_probs(logits)
    loss = -logp[0, 50]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
