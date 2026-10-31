import numpy as np
import pytest
from agentic_research.projection_audit_v2.solver import (
    patterns,derivatives,fw_gap,reference_project,all_losses)


def test_gradient_hessian_and_gap():
    bank=patterns();rng=np.random.default_rng(20261031)
    for _ in range(12):
        w=rng.dirichlet(np.ones(10)*3);q=rng.uniform(.001,.999,9)
        f,g,h=derivatives(w,q,bank)
        for j in range(9):
            d=np.eye(10)[j]-np.eye(10)[9];eps=1e-6
            fp,gp,_=derivatives(w+eps*d,q,bank)
            fm,gm,_=derivatives(w-eps*d,q,bank)
            assert abs((fp-fm)/(2*eps)-g@d)<1e-7
            np.testing.assert_allclose((gp-gm)/(2*eps),h@d,rtol=1e-6,atol=1e-7)
        w2=rng.dirichlet(np.ones(10))
        assert f-derivatives(w2,q,bank)[0] <= fw_gap(w,g)+1e-12


def test_closed_form():
    bank=np.array([[0.,0.],[1.,1.]])
    for q in [np.array([.2,.8]),np.array([.1,.4]),np.array([.000001,.01])]:
        expected=1/(1+np.exp(-(np.log(q)-np.log1p(-q)).mean()))
        out=reference_project(q,bank)
        np.testing.assert_allclose(out['probabilities'],expected,atol=1e-10)
        assert out['gap_pass']
        assert out['accepted']


def test_interior_coherent_identity():
    w=np.arange(1,11,dtype=float);w/=w.sum();q=w@patterns()
    out=reference_project(q)
    assert out['identity'] and out['gap_pass']
    np.testing.assert_array_equal(q,out['returned_probabilities'])
    assert out['fw_gap']==fw_gap(out['weights'],derivatives(out['weights'],q,patterns())[1])


def test_identity_near_clip_uses_actual_represented_gradient():
    bank=patterns()
    for epsilon in [1e-6,1.001e-6,1e-5]:
        w=np.array([1-9*epsilon]+[epsilon]*9)
        q=w@bank
        out=reference_project(q)
        assert out['identity']
        _,gradient,_=derivatives(out['weights'],q,bank)
        assert out['fw_gap']==fw_gap(out['weights'],gradient)
        assert out['identity_reconstruction_residual'] == max(
            np.max(np.abs(out['weights']@bank-q)),abs(out['weights'].sum()-1))
        np.testing.assert_array_equal(q,out['returned_probabilities'])


@pytest.mark.parametrize('q',[np.zeros(9),np.ones(9),np.arange(9)%2,np.full(9,.5)])
def test_extremes(q):
    out=reference_project(q)
    assert np.isfinite(out['probabilities']).all()
    assert out['feasibility_error']<=1e-10
    assert np.max(all_losses(out['returned_probabilities'],patterns())-all_losses(q,patterns()))<=0
    assert out['identity'] or out['gap_pass']


def test_random_and_near_boundary_fixed_synthetic_inputs():
    rng=np.random.default_rng(349);bank=patterns()
    inputs=np.vstack([rng.uniform(.01,.99,(24,9)),
                      1/(1+np.exp(-rng.uniform(-12,12,(24,9))))])
    for q in inputs:
        out=reference_project(q)
        assert out['gap_pass'],out
        assert np.max(all_losses(out['returned_probabilities'],bank)-all_losses(q,bank))<=0


@pytest.mark.parametrize('q',[np.full(9,np.nan),np.full(9,1.2),np.zeros(8)])
def test_invalid_input(q):
    with pytest.raises(ValueError):reference_project(q)


def test_near_coherent_and_sparse_fixed_synthetic_inputs():
    rng=np.random.default_rng(7159);bank=patterns()
    for scale in [1e-10,1e-6,1e-3,.03]:
        for _ in range(12):
            w=rng.dirichlet(np.full(10,.3));w[w<.01]=0;w/=w.sum()
            q=np.clip(w@bank+rng.normal(0,scale,9),1e-6,1-1e-6)
            out=reference_project(q)
            assert out['gap_pass'],out
            assert np.max(all_losses(out['returned_probabilities'],bank)-all_losses(q,bank))<=0


def test_identity_outside_clip_has_no_fake_gap():
    bank=np.array([[0.,0.],[1.,1.]])
    q=np.array([1e-8,1e-8])
    out=reference_project(q,bank)
    assert out['identity'] and out['identity_outside_clip']
    assert out['fw_gap'] is None and not out['gap_pass']
    np.testing.assert_array_equal(q,out['returned_probabilities'])


def test_exact_vertex_identity():
    for q in patterns():
        out=reference_project(q)
        assert out['identity'] and out['identity_outside_clip']
        np.testing.assert_array_equal(q,out['returned_probabilities'])


def test_failed_optimization_preserves_original(monkeypatch):
    import agentic_research.projection_audit_v2.solver as module
    def fake(q,bank):
        w=np.full(len(bank),1/len(bank))
        return w,{'iterations':1000,'stop':'iteration_limit','objective':1.,
                  'fw_gap':.1,'newton_steps':1000,'vertex_steps':0,'line_search_failures':0}
    monkeypatch.setattr(module,'_optimize',fake)
    q=np.array([.9,.1,.2,.3,.2,.1,.4,.1,.1])
    out=module.reference_project(q)
    assert not out['accepted']
    np.testing.assert_array_equal(q,out['returned_probabilities'])
