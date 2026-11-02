import numpy as np
import torch
import json
import pytest
from scipy.stats import chi2
from agentic_research.discovery_v3_math.decoder import nc_cdf,decode,tensor_case,fixture,quadrature,cell_masses,tensor,path_reference
import agentic_research.discovery_v3_math.audit as audit


def test_nccdf_is_central_chisquare_at_zero_noncentrality():
    x=tensor([3.,10.,22.]);nc=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    value=nc_cdf(x,nc,10)
    np.testing.assert_allclose(value.detach().numpy(),chi2.cdf(x.numpy(),10),atol=1e-14)
    value.sum().backward();assert torch.isfinite(nc.grad).all() and torch.all(nc.grad<=0)


def test_nccdf_broadcast_gradients_match_finite_differences():
    x=tensor([[15.,25.,40.]]).requires_grad_();nc=tensor([[.5],[8.]]).requires_grad_()
    assert torch.autograd.gradcheck(lambda a,b:nc_cdf(a,b,22),(x,nc),eps=1e-5,atol=5e-6,rtol=5e-4)


def test_positive_weights_and_guard_boundary():
    _,w,meta=quadrature(64)
    assert np.all(w>0) and w.sum()<=1+1e-15 and meta['forward_normal_tail_bound']<2e-15
    bad=decode(tensor_case(fixture('guard22')),64)
    good=decode(tensor_case(fixture('guard23')),64)
    np.testing.assert_array_equal(bad['joint'].numpy(),0)
    assert float(good['valid_probability'])>.999999 and good['joint'].max()>0


def test_invalid_mask_is_not_renormalized_away():
    c=fixture('central');c['masks']=np.array([np.ones(46,bool),np.arange(46)<22]);c['logits']=np.log([.2,.8]);c['slopes']=np.zeros(2)
    out=decode(tensor_case(c),64)
    assert abs(float(out['valid_probability'])-.2)<1e-12
    assert out['joint'].max()<=.2+1e-12


def test_cell_mass_map_keeps_outside_as_allfalse():
    q=np.array([[.1,.2,.25],[.15,.3,.4],[.2,.5,.7]])
    m=cell_masses(q);assert abs(m.sum()-1)<1e-12 and m.min()>=0
    np.testing.assert_allclose(m[1:].reshape(3,3).cumsum(0).cumsum(1),q)


@pytest.mark.parametrize('name',['central','mask'])
def test_independent_reference_uses_pathwise_statistics_and_valid_shape(name):
    r=path_reference(fixture(name),3,7)
    assert r['joint'].shape==(3,3) and r['paths']==8
    assert np.all((r['joint']>=0)&(r['joint']<=1))
    assert r['valid_probability']==1 if name=='central' else 0<r['valid_probability']<1


def test_ledger_records_failed_attempt_without_allowing_duplicate(tmp_path,monkeypatch):
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps({'math_audits_started':[],'max_math_audits':2,'other':'preserve'}))
    (tmp_path/'FREEZE.json').write_text('syntheticfreeze')
    monkeypatch.setattr(audit,'LEDGER',ledger);monkeypatch.setattr(audit,'HERE',tmp_path)
    audit.ledger_update('RUNNING')
    with pytest.raises(ValueError,match='duplicate'):audit.ledger_update('RUNNING')
    audit.ledger_update('FAILED_STOP',1.2)
    d=json.loads(ledger.read_text());assert d['other']=='preserve' and len(d['math_audits_started'])==1
    assert d['math_audits_started'][0]['status']=='FAILED_STOP' and d['math_audits_started'][0]['seconds']==1.2
