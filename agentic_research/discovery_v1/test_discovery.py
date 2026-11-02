import numpy as np
import torch
import datetime
import pytest
from agentic_research.discovery_v1.data import tokens,source_folds,shuffle_tokens,fit_fold
from agentic_research.discovery_v1.models import TCN,mdn_distribution
from agentic_research.violin_raw_v1.metrics import coherence_rate
from agentic_research.discovery_v1.runner import timed_fit,job_budget


def test_prefix_tokens_ignore_future_even_nonfinite():
    for sr in [8000,44100]:
        t=np.arange(sr*2)/sr;x=.1*np.sin(2*np.pi*440*t)
        before=tokens(x,sr,'DOUBLE');changed=x.copy();changed[sr:]=np.nan
        np.testing.assert_array_equal(before,tokens(changed,sr,'DOUBLE'))
        assert before.shape==(96,4)


def test_fold_ids_never_cross_people_and_order_control_preserves_multisets():
    ids=np.array([f'p{i}_{cohort}' for cohort in ['CG','EG','FG'] for i in range(3) for _ in range(4)])
    folds=source_folds(ids)
    assert len(set(folds))==3
    for person in set(ids):assert len(set(folds[ids==person]))==1
    seq=np.arange(2*96*4).reshape(2,96,4);s=shuffle_tokens(seq,['a','b'])
    for i in range(2):assert sorted(map(tuple,seq[i]))==sorted(map(tuple,s[i]))
    np.testing.assert_array_equal(s,shuffle_tokens(seq,['a','b']))


def test_inner_fold_fit_is_unchanged_when_held_future_and_features_are_nan():
    rng=np.random.default_rng(5);n=36
    ids=np.array([f'p{i}_{c}' for c in ['CG','EG','FG'] for i in range(3) for _ in range(4)])
    x=rng.normal(size=(n,60));x[:,57:59]=rng.uniform(.1,2,(n,2));x[:,59]=1
    seq=rng.normal(size=(n,96,4));y=rng.uniform(.1,4,(n,2));guard=np.ones(n,bool);mask=source_folds(ids)!=0
    a=fit_fold(x,seq,y,guard,ids,mask)
    x[~mask]=np.nan;seq[~mask]=np.nan;y[~mask]=np.nan;guard[~mask]=False
    b=fit_fold(x,seq,y,guard,ids,mask)
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_mdn_probabilities_coherent_monotone_and_differentiable():
    torch.manual_seed(0);raw=torch.randn(12,11,requires_grad=True)
    thresholds=torch.tensor([[-1.,0.,1.],[-1.,0.,1.]])
    p,*_=mdn_distribution(raw,thresholds,torch.tensor([-3.,-3.]))
    q=p.reshape(-1,3,3)
    assert torch.all(q[:,1:]>=q[:,:-1]) and torch.all(q[:,:,1:]>=q[:,:,:-1])
    assert coherence_rate(p.detach().numpy().astype(float))==1
    loss=-(torch.log(p.clamp_min(1e-6))).mean();loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_tcn_shape_and_prefix_parameter_count_small():
    model=TCN(10);out=model(torch.zeros(3,96,4))
    assert out.shape==(3,10) and sum(p.numel() for p in model.parameters())<5000


def test_failed_fit_time_is_accounted_and_persisted():
    state={'seconds':3.,'active_job':{}};times=iter([100.,101.5]);saved=[]
    def fail(started):raise RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError,match='synthetic failure'):
        timed_fit(fail,state,lambda:saved.append(state['seconds']),clock=lambda:next(times))
    assert state['seconds']==4.5 and state['active_job']['elapsed_seconds']==1.5 and saved==[4.5]


def test_job_budget_respects_remaining_total_deadline_and_attempts():
    now=datetime.datetime(2026,11,2,tzinfo=datetime.timezone.utc)
    plan={'budget':{'end_utc':(now+datetime.timedelta(seconds=10)).isoformat(),'single_job_seconds':180,'total_training_seconds':1800,'training_jobs':12}}
    state={'seconds':1797.,'attempts':11}
    assert job_budget(plan,state,now)==3
    state['seconds']=0
    assert job_budget(plan,state,now)==10
    state['attempts']=12
    with pytest.raises(TimeoutError):job_budget(plan,state,now)


def test_truncated_mdn_has_no_mass_below_physical_zero():
    raw=torch.zeros(2,11,dtype=torch.float64)
    thresholds=torch.tensor([[0.,1.,100.],[0.,1.,100.]],dtype=torch.float64)
    p,*_=mdn_distribution(raw,thresholds,torch.zeros(2,dtype=torch.float64))
    q=p.reshape(-1,3,3)
    np.testing.assert_allclose(q[:,0,:].numpy(),0,atol=1e-12)
    np.testing.assert_allclose(q[:,:,0].numpy(),0,atol=1e-12)
    np.testing.assert_allclose(q[:,-1,-1].numpy(),.5,atol=1e-12)
