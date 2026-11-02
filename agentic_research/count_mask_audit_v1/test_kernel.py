import json
import numpy as np
import torch
import pytest
from .kernel import tensor,dp,guard_dp,numpy_counts,enumerate_paths,inference,tables,direct_path_query,cells


def toy():
    return {'small_pitch_block_means':[[-.4,.6],[.7,-.2]],'small_power_block_means':[[-.3,.4],[.5,-.5]],
      'scales':[[.7,.5],[.4,.9]],'thresholds_pitch':[.3,.8,1.5],'thresholds_power':[.2,.6,1.3],
      'mixture':[.3,.7]}


def logits():return np.array([[[.4,.4],[-.7,.9],[.6,-.2],[-.1,.8]],[[.1,.1],[.8,-.6],[-.3,.7],[.5,-.4]]])


def test_dense_counts_match_binary_enumeration_and_independent_dictionary():
    for h in range(2):
        w,_=dp(tensor(logits()[h]),[2,2]);enum=np.zeros((3,3))
        for _,p,(a,b) in enumerate_paths(logits()[h],[2,2]):enum[a,b]+=p
        np.testing.assert_allclose(w,enum,atol=1e-14);np.testing.assert_allclose(w,numpy_counts(logits()[h],[2,2]),atol=1e-14)
        assert w.sum()==pytest.approx(1.)
        assert guard_dp(tensor(logits()[h]),2)==pytest.approx(enum[np.indices((3,3)).sum(0)>=2].sum())


@pytest.mark.parametrize('conditioned',[False,True])
def test_count_statistic_cdf_matches_independent_path_means(conditioned):
    plan=toy();tab,_,_=tables(plan,[2,2],2,count_conditioned=conditioned)
    q=inference(tensor(logits()),[2,2],2,tab,plan['mixture'])['q']
    ref,_=direct_path_query(plan,logits(),[2,2],2,count_conditioned=conditioned)
    np.testing.assert_allclose(q,ref,atol=1e-13)


def test_surviving_mass_brackets_and_local_logit_gradient_bound():
    plan=toy();tab,_,_=tables(plan,[2,2],2);theta=tensor(logits()).requires_grad_()
    full=inference(theta,[2,2],2,tab,plan['mixture']);fg=torch.autograd.grad(full['q'].mean(),theta)[0]
    small=tensor(logits()).requires_grad_();beam=inference(small,[2,2],2,tab,plan['mixture'],2)
    bg=torch.autograd.grad(beam['q'].mean(),small)[0];r=float(beam['residual'].detach())
    assert r>=-1e-14
    assert torch.all(beam['q']<=full['q']+1e-13)
    assert torch.all(full['q']<=beam['q']+r+1e-13)
    assert float(torch.max(torch.abs(fg-bg)))<=r+1e-13
    assert abs(float(fg.sum()-bg.sum()))<=4*r+1e-13


def test_fixed_trace_finite_difference_and_nonnegative_cells():
    plan=toy();tab,_,_=tables(plan,[2,2],2);theta=tensor(logits()).requires_grad_()
    out=inference(theta,[2,2],2,tab,plan['mixture'],3);grad=torch.autograd.grad(out['q'].mean(),theta)[0]
    up=logits();down=logits();step=1e-5;up[0,1,1]+=step;down[0,1,1]-=step
    a=inference(tensor(up),[2,2],2,tab,plan['mixture'],3,out['traces'])['q'].mean()
    b=inference(tensor(down),[2,2],2,tab,plan['mixture'],3,out['traces'])['q'].mean()
    assert float(grad[0,1,1])==pytest.approx(float((a-b)/(2*step)),abs=1e-10)
    assert cells(out['q'].detach()).min()>=-1e-13


def test_fixed_law_coarsening_encloses_arbitrary_mean_path():
    plan=toy();_,lo,hi=tables(plan,[2,2],2,coarsening=True)
    lower=inference(tensor(logits()),[2,2],2,lo,plan['mixture'])['q'].numpy()
    upper=inference(tensor(logits()),[2,2],2,hi,plan['mixture'])['q'].numpy()
    ref,_=direct_path_query(plan,logits(),[2,2],2,perturbed=True)
    assert np.all(lower<=ref+1e-13) and np.all(ref<=upper+1e-13)


def test_invalid_shapes_and_beams_fail_closed():
    with pytest.raises(ValueError):dp(tensor(logits()[0]),[1,1,2])
    with pytest.raises(ValueError):dp(tensor(logits()[0]),[2,2],0)
    bad=logits()[0];bad[0,0]=np.nan
    with pytest.raises(ValueError):dp(tensor(bad),[2,2])


def test_one_attempt_registration_and_failure_stay_consumed(tmp_path):
    from .audit import register
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'max_math_audits':2,'math_audits_started':[{'audit':'old'}]}))
    register('RUNNING',ledger=ledger,freeze_sha='synthetic')
    with pytest.raises(ValueError):register('RUNNING',ledger=ledger,freeze_sha='changed')
    register('FAILED_STOP',ledger=ledger,seconds=3.)
    d=json.loads(ledger.read_text());assert len(d['math_audits_started'])==2
    assert d['math_audits_started'][-1]['status']=='FAILED_STOP'
