"""Synthetic verification only: no source target export or training occurs here."""
import itertools
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Tests must run from both the repository root and this standalone branch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rule_patterns as rp


def independent_scalar_rule(p, r, guard, fixed):
    return tuple(int(fixed or (guard and p >= lower and r < upper))
                 for lower in (.7, .8, .9) for upper in (.3, .5, .7))


def test_exhaustive_threshold_edges_and_shared_overrides():
    patterns = set()
    cases = 0
    for p, r, guard, fixed in itertools.product(rp.boundary_representatives((.7, .8, .9)),
                                               rp.boundary_representatives((.3, .5, .7)), (0, 1), (0, 1)):
        expected = independent_scalar_rule(p, r, guard, fixed)
        actual = rp.labels_from_features(p, r, guard, fixed)
        assert tuple(actual.astype(int)) == expected
        idx = int(rp.pattern_ids(actual))
        np.testing.assert_array_equal(rp.admissible_patterns()[idx], expected)
        patterns.add(expected)
        cases += 1
    assert cases == 324
    assert len(patterns) == 10
    assert patterns == set(map(tuple, rp.admissible_patterns()))


def test_all_512_binary_vectors_accepted_exactly_when_in_the_ten_patterns():
    accepted = set(map(tuple, rp.admissible_patterns()))
    for row in itertools.product((0, 1), repeat=9):
        if row in accepted:
            idx = int(rp.pattern_ids(row))
            assert int(rp.torch_pattern_ids(torch.tensor(row))) == idx
            np.testing.assert_array_equal(rp.admissible_patterns()[idx], row)
        else:
            with pytest.raises(ValueError): rp.pattern_ids(row)
            with pytest.raises(ValueError): rp.torch_pattern_ids(torch.tensor(row))


def test_strict_ratio_boundary_and_inclusive_periodicity_boundary():
    # At ratio .5, the .5 gate is false; at periodicity .8, the .8 gate is true.
    expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(rp.labels_from_features(.8, .5).reshape(3, 3), expected)
    assert not rp.labels_from_features(np.nan, .1).any()
    assert rp.labels_from_features(np.nan, np.nan, guard=False, fixed=True).all()


def test_broadcasting_and_returned_pattern_copy():
    out = rp.labels_from_features(np.array([.6, .75, .95]), .1)
    assert out.shape == (3, 9)
    ids = rp.pattern_ids(out)
    np.testing.assert_array_equal(rp.admissible_patterns()[ids], out)
    changed = rp.admissible_patterns(); changed[:] = 3
    assert np.isin(rp.admissible_patterns(), [0, 1]).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_normalized_joint_distribution_and_nonnegative_consistent_marginals(dtype):
    generator = torch.Generator().manual_seed(20261030)
    logits = torch.randn(4, 2, 10, generator=generator, dtype=dtype, requires_grad=True)
    probs = rp.joint_log_probabilities(logits).exp()
    marg = rp.marginal_probabilities(logits)
    atol = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(probs.sum(-1), torch.ones(4, 2, dtype=dtype), atol=atol, rtol=atol)
    assert torch.isfinite(marg).all() and ((marg >= 0) & (marg <= 1+atol)).all()
    reference = probs @ torch.tensor(rp.admissible_patterns(), dtype=dtype)
    torch.testing.assert_close(marg, reference, atol=atol, rtol=atol)
    grid = marg.reshape(4, 2, 3, 3)
    assert (grid[..., :-1, :] >= grid[..., 1:, :]-atol).all()
    assert (grid[..., :, :-1] <= grid[..., :, 1:]+atol).all()
    marg.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_losses_match_probability_definitions_and_have_finite_gradients():
    logits = torch.linspace(-3, 4, 100, dtype=torch.float64).reshape(10, 10).requires_grad_()
    labels = torch.tensor(rp.admissible_patterns())
    ids = torch.arange(10)
    marg = rp.marginal_probabilities(logits)
    expected_bce = -(labels*marg.log()+(1-labels)*torch.log1p(-marg))
    torch.testing.assert_close(rp.per_rule_bce(logits, labels, 'none'), expected_bce)
    expected_nll = -torch.log_softmax(logits, -1)[torch.arange(10), ids]
    torch.testing.assert_close(rp.categorical_nll(logits, ids, 'none'), expected_nll)
    loss = rp.per_rule_bce(logits, labels)+rp.categorical_nll(logits, ids)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_keep_log_losses_and_gradients_finite():
    logits = torch.full((10, 10), -10000., requires_grad=True)
    with torch.no_grad(): logits[:, 0] = 10000.
    labels = torch.tensor(rp.admissible_patterns())
    loss = rp.per_rule_bce(logits, labels)+rp.categorical_nll(logits, torch.arange(10))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('bad', [[], [0]*8, [0]*10, [0]*8+[2], [0]*8+[float('nan')]])
def test_invalid_numpy_labels_rejected(bad):
    with pytest.raises(ValueError): rp.pattern_ids(bad)


@pytest.mark.parametrize('bad', [torch.zeros(9), torch.zeros(2, 11), torch.ones(10, dtype=torch.int64),
                                torch.full((10,), float('nan')), torch.zeros(0, 10)])
def test_invalid_logits_rejected(bad):
    with pytest.raises(ValueError): rp.marginal_probabilities(bad)


def test_loss_target_shape_range_and_reduction_checks():
    logits = torch.zeros(2, 10)
    labels = torch.zeros(2, 9, dtype=torch.int64)
    with pytest.raises(ValueError): rp.per_rule_bce(logits, labels[:1])
    with pytest.raises(ValueError): rp.per_rule_bce(logits, labels, 'invalid')
    with pytest.raises(ValueError): rp.categorical_nll(logits, torch.tensor([0, 10]))
    with pytest.raises(ValueError): rp.categorical_nll(logits, torch.tensor([0., 1.]))
    with pytest.raises(ValueError): rp.categorical_nll(logits, torch.tensor([0]))
    torch.testing.assert_close(rp.per_rule_bce(logits, labels, 'sum'), rp.per_rule_bce(logits, labels, 'none').sum())
