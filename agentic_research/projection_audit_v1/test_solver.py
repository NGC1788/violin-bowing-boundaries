import numpy as np
import pytest
from agentic_research.projection_audit_v1.solver import (
    patterns, objective_and_gradient, gap, reference_project, all_losses, simplex)


def test_bank_independent_parity():
    from agentic_research.forecast_v3.rule_patterns import admissible_patterns
    np.testing.assert_array_equal(patterns(),admissible_patterns())


def test_gradient_and_convex_gap():
    rng=np.random.default_rng(21);bank=patterns()
    for _ in range(12):
        w=rng.dirichlet(np.ones(10)*3); q=rng.uniform(.05,.95,9)
        f,g=objective_and_gradient(w,q,bank)
        for j in range(9):
            d=np.eye(10)[j]-np.eye(10)[9];h=1e-6
            fd=(objective_and_gradient(w+h*d,q,bank)[0]-objective_and_gradient(w-h*d,q,bank)[0])/(2*h)
            assert abs(fd-g@d)<1e-7
        # Convex first-order gap upper-bounds improvement at arbitrary feasible w2.
        for _ in range(10):
            w2=rng.dirichlet(np.ones(10))
            assert f-objective_and_gradient(w2,q,bank)[0] <= gap(w,q,bank)+1e-12


def test_closed_form_two_identical_coordinates():
    bank=np.array([[0.,0.],[1.,1.]])
    for q in [np.array([.2,.8]),np.array([.1,.4]),np.array([.7,.95])]:
        logit=np.log(q)-np.log1p(-q)
        expected=1/(1+np.exp(-logit.mean()))
        out=reference_project(q,bank)
        np.testing.assert_allclose(out['probabilities'],expected,atol=1e-7)
        assert out['fw_gap']<1e-8
        assert out['accepted']
        assert np.max(all_losses(out['returned_probabilities'],bank)-all_losses(q,bank))<=0


def test_known_coherent_solution():
    bank=patterns(); w=np.arange(1,11,dtype=float);w/=w.sum();q=w@bank
    out=reference_project(q)
    assert abs(out['objective'])<1e-12
    np.testing.assert_allclose(out['probabilities'],q,atol=2e-7)
    # Tiny float score changes are allowed only by fallback, never by tolerance.
    assert np.max(all_losses(out['returned_probabilities'],bank)-all_losses(q,bank))<=0


@pytest.mark.parametrize('q',[np.zeros(9),np.ones(9),np.arange(9)%2,np.full(9,.5)])
def test_extremes_and_score_guard(q):
    out=reference_project(q)
    assert np.isfinite(out['probabilities']).all()
    assert np.min(out['weights'])>=0
    assert abs(sum(out['weights'])-1)<1e-12
    assert np.max(all_losses(out['returned_probabilities'],patterns())-all_losses(q,patterns()))<=0


@pytest.mark.parametrize('q',[np.full(9,np.nan),np.full(9,1.2),np.zeros(8)])
def test_bad_probabilities_rejected(q):
    with pytest.raises(ValueError):reference_project(q)


def test_simplex():
    w=simplex(np.array([-.8,.6,2.]));assert np.min(w)>=0
    assert abs(w.sum()-1)<1e-12


def test_failed_start_is_recorded(monkeypatch):
    import agentic_research.projection_audit_v1.solver as module
    real=module.minimize
    calls=[]
    def sometimes(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise ArithmeticError('synthetic trial failure')
        return real(*args,**kwargs)
    monkeypatch.setattr(module,'minimize',sometimes)
    result=reference_project(np.full(9,.4))
    assert not result['starts'][0]['usable']
    assert result['starts'][1]['usable']
    assert 'canonicalization_max_change' in result['starts'][1]


def test_all_failed_starts_are_not_dropped(monkeypatch):
    import agentic_research.projection_audit_v1.solver as module
    def fail(*args,**kwargs):raise ArithmeticError('synthetic all failure')
    monkeypatch.setattr(module,'minimize',fail)
    with pytest.raises(ArithmeticError,match='Both reference'):
        reference_project(np.full(9,.4))
